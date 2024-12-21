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

  auto tensorType = hidden_states.getType();

  auto mlp_output =
      builder.create<mix::MLPOp>(loc, tensorType, hidden_states, residual);
  return mlp_output;
}

auto genSelfAttn(mlir::MLIRContext &context, mlir::OpBuilder &builder,
                 Location loc, TypedValue<RankedTensorType> &hidden_states,
                 TypedValue<RankedTensorType> &residual,
                 TypedValue<RankedTensorType> &attention_mask) {
  auto hidden_statesType = hidden_states.getType();
  auto self_attn_output = builder.create<mix::SelfAttentionOp>(
      loc, hidden_statesType, hidden_states, residual, attention_mask);
  return self_attn_output;
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
                      Location loc, int64_t seq_length, int64_t vocab_size) {
  // 创建ml_program::subgraph
  auto input_ids_type =
      RankedTensorType::get({1, seq_length}, builder.getI32Type());
  auto output_type =
      RankedTensorType::get({seq_length, vocab_size}, builder.getF32Type());

  auto functionTy = builder.getFunctionType({input_ids_type}, {output_type});

  auto graph = builder.create<ml_program::SubgraphOp>(
      loc, "Telechat", functionTy, ArrayAttr{}, ArrayAttr{},
      builder.getStringAttr("private"));

  auto body = graph.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto Argument0 = graph.getArgument(0);
  auto Argument1 = graph.getArgument(1);

  auto hidden_states = mlir::dyn_cast<TypedValue<RankedTensorType>>(Argument0);
  auto attention_mask = mlir::dyn_cast<TypedValue<RankedTensorType>>(Argument1);

  int n_layer = 38;
  Operation *transformerBlock;
  // 循环创建N个Block
  for (int i = 0; i < n_layer; ++i) {
    // transformer block
    transformerBlock = genTransformerBlock(context, builder, graph->getLoc(),
                                           hidden_states, attention_mask);

    hidden_states = mlir::dyn_cast<mix::MLPOp>(transformerBlock).getOutput();
  }
  auto last_transformerBlock = mlir::dyn_cast<mix::MLPOp>(transformerBlock);

  // 加入Linear层映射过去
  auto output_embedding = builder.create<mix::LinearOp>(
      last_transformerBlock->getLoc(), output_type,
      last_transformerBlock.getOutput(), nullptr);
  builder.create<ml_program::OutputOp>(loc, ValueRange{output_embedding});
  return graph;
}

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mix::MIXDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::ml_program::MLProgramDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(theModule.getBody());

  // 输入token序列长度，静态长度
  int64_t seq_length = 40;
  // 字典长度
  int64_t vocab_size = 120000;

  auto telechat_graph =
      genTelechatModel(context, builder, loc, seq_length, vocab_size);

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
