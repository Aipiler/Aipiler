#include "mix/mixDialect.h"
#include "mix/mixOps.h"
#include "mix/mixTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace mlir;

std::unique_ptr<Pass> createLowerModulePass();

auto genFFN(mlir::MLIRContext &context, mlir::OpBuilder &builder, Location loc,
            TypedValue<RankedTensorType> &hidden_states,
            TypedValue<RankedTensorType> &residual) {

  auto elementType = builder.getF32Type();

  auto linear0 = builder.create<mix::LinearOp>(loc, hidden_states,
                                               "model_parameters.mlp.linear0",
                                               2, 2, false, elementType);

  auto silu0 = builder.create<mix::SiLUOp>(loc, linear0);

  auto linear1 = builder.create<mix::LinearOp>(loc, hidden_states,
                                               "model_parameters.mlp.linear1",
                                               2, 2, false, elementType);
  auto mul0 = builder.create<mix::MulOp>(loc, silu0, linear1);

  auto linear2 = builder.create<mix::LinearOp>(
      loc, mul0, "model_parameters.mlp.linear2", 2, 2, true, elementType);
  auto output = builder.create<mix::AddOp>(loc, linear2, residual);

  return output;
}

auto genSelfAttn(mlir::MLIRContext &context, mlir::OpBuilder &builder,
                 Location loc, TypedValue<RankedTensorType> &hidden_states,
                 TypedValue<RankedTensorType> &residual,
                 TypedValue<RankedTensorType> &attention_mask) {
  auto hidden_statesType = hidden_states.getType();
  //   auto self_attn_output = builder.create<mix::SelfAttentionOp>(
  //       loc, hidden_statesType, hidden_states, residual, attention_mask);

  // hidden_states dims
  auto hidden_states_shape = hidden_statesType.getShape();
  auto batch_size = hidden_states_shape[0];
  auto seq_length = hidden_states_shape[1];
  auto hidden_size = hidden_states_shape[2];
  // RankedTensorType tensorType = RankedTensorType::get();

  /* 定义一些可重用的信息 */

  // types:
  auto type_i32 = builder.getI32Type();
  auto type_query_weight =
      RankedTensorType::get({hidden_size, hidden_size}, builder.getF32Type());
  // Attrs:
  auto attr_i0 = IntegerAttr::get(type_i32, 0);
  auto attr_i1 = IntegerAttr::get(type_i32, 1);
  auto attr_i2 = IntegerAttr::get(type_i32, 2);
  auto attr_i_batch_size = IntegerAttr::get(type_i32, batch_size);
  auto attr_i_seq_length = IntegerAttr::get(type_i32, seq_length);
  auto attr_i_hidden_size = IntegerAttr::get(type_i32, hidden_size);

  // (1, 0)
  SmallVector<Attribute> vec_1_0{attr_i1, attr_i0};
  auto attr_1_0 = mlir::ArrayAttr::get(&context, vec_1_0);
  // (seq_length, hidden_size)
  SmallVector<Attribute> vec_seq_length_hidden_size{attr_i_seq_length,
                                                    attr_i_hidden_size};
  auto attr_seq_length_hidden_size =
      mlir::ArrayAttr::get(&context, vec_seq_length_hidden_size);

  /* 定义算子 */

  // line 14: torch.aten.transpose.int
  auto transpose14 =
      builder.create<mix::PermuteOp>(loc, hidden_states, attr_1_0);

  // %arg9
  auto query_weight = builder.create<mix::WeightOp>(
      transpose14->getLoc(), type_query_weight, "Self_attn.query.weight");

  // line 16: torch.aten.t
  auto t16 = builder.create<mix::PermuteOp>(loc, query_weight, attr_1_0);

  // line 21: torch.aten.view
  auto reshape21 = builder.create<mix::ReshapeOp>(
      query_weight->getLoc(), t16.getType(), t16, attr_seq_length_hidden_size);

  return reshape21;
}

auto genRMSNorm(mlir::MLIRContext &context, mlir::OpBuilder &builder,
                Location loc, TypedValue<RankedTensorType> &hidden_states) {
  auto hidden_states_type = hidden_states.getType();
  auto hidden_size = hidden_states_type.getShape()[2];
  auto eps = APFloat(1e-6);
  auto self_attn_output = builder.create<mix::RMSNormOp>(
      loc, hidden_states_type, hidden_states, hidden_size, eps);
  return self_attn_output;
}

auto genTransformerBlock(mlir::MLIRContext &context, mlir::OpBuilder &builder,
                         Location loc,
                         TypedValue<RankedTensorType> &hidden_states,
                         TypedValue<RankedTensorType> &attention_mask) {

  // RMSNorm
  auto input_RMSNorm = genRMSNorm(context, builder, loc, hidden_states);
  auto input_RMSNorm_output = input_RMSNorm.getOutput();

  // Self_attention
  auto Self_attn =
      genSelfAttn(context, builder, input_RMSNorm->getLoc(),
                  input_RMSNorm_output, hidden_states, attention_mask);
  auto Self_attn_output = Self_attn.getOutput();

  // RMSNorm
  auto post_RMSNorm =
      genRMSNorm(context, builder, Self_attn->getLoc(), Self_attn_output);
  auto post_RMSNorm_output = post_RMSNorm.getOutput();

  // MLP
  auto FFNoutput = genFFN(context, builder, post_RMSNorm->getLoc(),
                          post_RMSNorm_output, Self_attn_output);
  return FFNoutput;
}

auto genTelechatModel(mlir::MLIRContext &context, mlir::OpBuilder &builder,
                      Location loc, int64_t hidden_size, int64_t seq_length,
                      int64_t vocab_size) {

  /* Types */
  auto F32Type = builder.getF32Type();
  auto input_ids_type =
      RankedTensorType::get({1, seq_length}, builder.getI32Type());
  auto output_type =
      RankedTensorType::get({seq_length, vocab_size}, builder.getF32Type());
  auto functionTy = builder.getFunctionType({input_ids_type}, {output_type});

  // 创建ml_program::subgraph
  auto graph = builder.create<ml_program::SubgraphOp>(
      loc, "Telechat", functionTy, ArrayAttr{}, ArrayAttr{},
      builder.getStringAttr("private"));

  auto body = graph.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto Argument0 = graph.getArgument(0);
  auto input_ids = mlir::dyn_cast<TypedValue<RankedTensorType>>(Argument0);

  /* 逻辑 */
  auto input_embedding = builder.create<mix::LinearOp>(
      loc, input_ids, "input_embedding.mlp.linear0", seq_length, hidden_size,
      false, F32Type);

  auto hidden_states = input_embedding.getOutput();

  int n_layer = 38;
  Operation *transformerBlock;
  // 循环创建N个Block
  for (int i = 0; i < n_layer; ++i) {
    // transformer block
    transformerBlock = genTransformerBlock(context, builder, graph->getLoc(),
                                           hidden_states, hidden_states);

    hidden_states = transformerBlock.getOutput();
  }
  auto last_transformerBlock = mlir::dyn_cast<mix::MLPOp>(transformerBlock);

  hidden_states = last_transformerBlock.getOutput();

  // Linear:将hidden_states映射到vocab_size上
  auto output_embedding = builder.create<mix::LinearOp>(
      loc, hidden_states, "output_embedding.mlp.linear0", hidden_size,
      vocab_size, false, F32Type);

  builder.create<ml_program::OutputOp>(loc, ValueRange{output_embedding});
  return graph;
}

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mix::MIXDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::ml_program::MLProgramDialect>();
  context.getOrLoadDialect<mix::MIXDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(theModule.getBody());

  // hidden_size
  int64_t hidden_size = 5120;
  // 输入token序列长度，静态长度
  int64_t seq_length = 40;
  // 字典长度
  int64_t vocab_size = 120000;

  auto telechat_graph = genTelechatModel(context, builder, loc, hidden_size,
                                         seq_length, vocab_size);

  theModule->dump();

  // std::cout << ShapedType::kDynamic << std::endl;

  // mlir::PassManager pm(&context);
  // pm.addPass(createLowerModulePass());
  // if (mlir::failed(pm.run(theModule))) {
  //   return 4;
  // }
  // std::cout << "==== After Lower pass GRAPH-LOWER =====" << std::endl;
  // theModule->dump();

  return 0;
}
