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
#include "mlir/IR/ExtensibleDialect.h"
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
std::unique_ptr<Pass> createLowerCompositePass();
std::unique_ptr<Pass> createLowerPrimaryToTosa();

ArrayAttr createIntArrayAttr(MLIRContext &context,
                             const std::vector<int64_t> &values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());

  for (auto value : values) {
    attrs.push_back(IntegerAttr::get(IntegerType::get(&context, 64), value));
  }

  return ArrayAttr::get(&context, attrs);
}

auto genSelfAttn(mlir::MLIRContext &context, mlir::OpBuilder &builder,
                 Location loc, TypedValue<RankedTensorType> &hidden_states,
                 TypedValue<RankedTensorType> &residual,
                 TypedValue<RankedTensorType> &attention_mask) {

  //   auto hidden_states_type =
  //   mlir::dyn_cast<RankedTensorType>(hidden_states_v); auto residual_type =
  //   mlir::dyn_cast<RankedTensorType>(residual_v); auto attention_mask_type =
  //   mlir::dyn_cast<RankedTensorType>(attention_mask_v);
  auto hidden_states_type = hidden_states.getType();
  //   auto self_attn_output = builder.create<mix::SelfAttentionOp>(
  //       loc, hidden_statesType, hidden_states, residual, attention_mask);

  // hidden_states dims
  auto hidden_states_shape = hidden_states_type.getShape();
  auto batch_size = hidden_states_shape[0];
  auto seq_length = hidden_states_shape[1];
  auto hidden_size = hidden_states_shape[2];
  auto n_head = 32;
  auto head_dim = hidden_size / 32;
  auto key_value_projection_size = hidden_size * 2;
  auto key_value_projection_head_dim = key_value_projection_size / 32;
  /* 定义一些可重用的信息 */

  // types:
  auto type_i32 = builder.getI32Type();
  auto type_i64 = builder.getI64Type();
  auto type_query_weight =
      RankedTensorType::get({hidden_size, hidden_size}, builder.getF32Type());
  auto type_key_value_weight = RankedTensorType::get(
      {hidden_size * 2, hidden_size}, builder.getF32Type());
  auto type_dense_weight =
      RankedTensorType::get({hidden_size, hidden_size}, builder.getF32Type());
  auto type_dense_bias =
      RankedTensorType::get({hidden_size}, builder.getF32Type());

  /* 定义算子 */

  // %arg0
  auto query_weight = builder.create<mix::WeightOp>(loc, type_query_weight,
                                                    "Self_attn.query.weight");
  // %arg1
  auto key_value_weight = builder.create<mix::WeightOp>(
      loc, type_key_value_weight, "Self_attn.key_value.weight");

  // %arg2
  auto dense_weight = builder.create<mix::WeightOp>(loc, type_dense_weight,
                                                    "Self_attn.dense.weight");

  // %arg3
  auto dense_bias = builder.create<mix::WeightOp>(loc, type_dense_bias,
                                                  "Self_attn.dense.bias");

  // line 14: torch.aten.transpose.int
  auto transpose14 = builder.create<mix::PermuteOp>(
      loc, hidden_states, createIntArrayAttr(context, {1, 0, 2}));

  // line 16: torch.aten.t
  auto t16 = builder.create<mix::PermuteOp>(
      loc, query_weight, createIntArrayAttr(context, {1, 0}));

  // line 21: torch.aten.view
  auto reshape21 = builder.create<mix::ReshapeOp>(
      loc, transpose14, createIntArrayAttr(context, {seq_length, hidden_size}));

  // line 23: torch.aten.mm
  auto matmul23 = builder.create<mix::MatMulOp>(loc, reshape21, t16);

  // line 29: torch.aten.view
  auto reshape29 = builder.create<mix::ReshapeOp>(
      loc, matmul23,
      createIntArrayAttr(context, {seq_length, batch_size, hidden_size}));

  // line 36: torch.aten.view
  auto reshape36 = builder.create<mix::ReshapeOp>(
      loc, reshape29,
      createIntArrayAttr(context, {seq_length, batch_size, n_head, head_dim}));

  // line 38: torch.aten.t
  auto t38 = builder.create<mix::PermuteOp>(
      loc, key_value_weight, createIntArrayAttr(context, {1, 0}));

  // line 43: torch.aten.view
  auto reshape43 = builder.create<mix::ReshapeOp>(
      loc, transpose14, createIntArrayAttr(context, {seq_length, hidden_size}));

  // line 45: torch.aten.mm
  auto matmul45 = builder.create<mix::MatMulOp>(loc, reshape43, t38);

  // line 51: torch.aten.view
  auto reshape51 = builder.create<mix::ReshapeOp>(
      loc, matmul45,
      createIntArrayAttr(context,
                         {seq_length, batch_size, key_value_projection_size}));

  // line 58: torch.aten.view
  auto reshape58 = builder.create<mix::ReshapeOp>(
      loc, reshape51,
      createIntArrayAttr(context, {seq_length, batch_size, n_head,
                                   key_value_projection_head_dim}));

  // line 64: torch.aten.slice.Tensor
  auto slice64 = builder.create<mix::SliceOp>(loc, reshape58, 3, 0, 160, 1);

  // line 70: torch.aten.slice.Tensor
  auto slice70 = builder.create<mix::SliceOp>(loc, reshape58, 3, 160, 320, 1);

  // line 76: torch.aten.view
  auto reshape76 = builder.create<mix::ReshapeOp>(
      loc, reshape36, createIntArrayAttr(context, {seq_length, n_head, -1}));

  // line 82: torch.aten.view
  auto reshape82 = builder.create<mix::ReshapeOp>(
      loc, slice64, createIntArrayAttr(context, {seq_length, n_head, -1}));

  return t38;
}

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mix::MIXDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();

  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToEnd(theModule.getBody());

  auto elementType = builder.getF32Type();
  auto printInputType = UnrankedTensorType::get(elementType);
  auto printFunTy =
      builder.getFunctionType(TypeRange{printInputType}, TypeRange{});
  auto printfunc =
      builder.create<func::FuncOp>(loc, "printMemrefF32", printFunTy);
  printfunc.setPrivate();

  auto functionTy = builder.getFunctionType({}, {});
  auto graph0 = builder.create<func::FuncOp>(loc, "self_attention", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto hidden_states_type = RankedTensorType::get({1, 5, 5120}, elementType);
  auto attention_mask_type = RankedTensorType::get({1, 1, 5, 5}, elementType);

  auto hidden_states = builder.create<mix::WeightOp>(
      graph0->getLoc(), hidden_states_type, "hidden_states");
  auto residual = builder.create<mix::WeightOp>(hidden_states->getLoc(),
                                                hidden_states_type, "residual");
  auto attention_mask = builder.create<mix::WeightOp>(
      residual->getLoc(), attention_mask_type, "attention_mask");

  auto hidden_states_output = hidden_states.getOutput();
  auto residual_output = residual.getOutput();
  auto attention_mask_output = attention_mask.getOutput();

  auto self_attn =
      genSelfAttn(context, builder, attention_mask->getLoc(),
                  hidden_states_output, residual_output, attention_mask_output);

  builder.create<func::ReturnOp>(loc, ValueRange{});

  //   builder.setInsertionPointToEnd(theModule.getBody());
  //   auto mainfunc = builder.create<func::FuncOp>(loc, "main",
  //                                                builder.getFunctionType({},
  //                                                {}));
  //   mainfunc.setPrivate();
  //   auto mainbody = mainfunc.addEntryBlock();
  //   builder.setInsertionPointToEnd(mainbody);

  //   auto argAttr =
  //       DenseElementsAttr::get(tensorType, llvm::ArrayRef<float>{1, 2, 3,
  //       4});

  //   auto arg0 = builder.create<arith::ConstantOp>(loc, argAttr);
  //   auto arg1 = builder.create<arith::ConstantOp>(loc, argAttr);
  //   builder.create<func::CallOp>(loc, graph0, ValueRange{arg0, arg1});
  //   builder.create<func::ReturnOp>(loc);
  mlir::PassManager pm(&context);
  pm.addPass(createLowerModulePass());
  pm.addPass(createLowerCompositePass());
  pm.addPass(createLowerPrimaryToTosa());

  theModule->dump();

  std::cout << "-------------------------------------------------" << std::endl;

  if (mlir::failed(pm.run(theModule))) {
    return -1;
  }
  theModule->dump();

  return 0;
}