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
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
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

std::pair<Value, Value> genRotaryEmbedding(mlir::MLIRContext &context,
                                           mlir::OpBuilder &builder,
                                           Location loc) {

  // hidden_states dims
  auto hidden_size = 5120;
  auto n_head = 32;
  auto head_dim = hidden_size / n_head;
  auto key_value_projection_size = hidden_size * 2;
  auto key_value_projection_head_dim = key_value_projection_size / n_head;
  auto seq_len = 8192;

  /* 定义一些可重用的信息 */

  // types:
  auto type_i32 = builder.getI32Type();
  auto type_i64 = builder.getI64Type();
  auto type_f32 = builder.getF32Type();
  auto type_f64 = builder.getF16Type();
  auto type_query_weight =
      RankedTensorType::get({hidden_size, hidden_size}, type_f32);
  auto type_key_value_weight =
      RankedTensorType::get({key_value_projection_size, hidden_size}, type_f32);
  auto type_dense_weight =
      RankedTensorType::get({hidden_size, hidden_size}, type_f32);
  auto type_dense_bias = RankedTensorType::get({hidden_size}, type_f32);

  /* 定义算子 */

  // 下面是RotaryEmbedding 的代码，应该在genRotaryEmbedding中实现

  // line 94: torch.aten.arange.start_step
  SmallVector<int64_t> tmp94(80);
  for (int i = 0, j = 0; i < 80; i++, j += 2) {
    tmp94[i] = j;
  }
  auto dense94 = DenseElementsAttr::get(RankedTensorType::get({80}, type_i64),
                                        ArrayRef<int64_t>(tmp94));
  auto constant94 = builder.create<mix::ConstantOp>(loc, dense94);

  // line 102: torch.aten._to_copy
  auto convert102 = builder.create<mix::ConvertOp>(loc, constant94, type_f32);

  // line 104: torch.constant.int
  auto scalar101 = IntegerAttr::get(type_i64, 160);
  auto constant101 = builder.create<mix::ConstantOp>(loc, scalar101);

  // line 106: torch.aten.div.Tensor
  auto dev106 = builder.create<mix::DivOp>(loc, convert102, constant101);

  // line 108: torch.constant.float
  auto scalar108 = FloatAttr::get(type_f64, 30420.108888514722);
  auto constant108 = builder.create<mix::ConstantOp>(loc, scalar108);

  // line 110: torch.aten.pow.Scalar
  auto pow110 = builder.create<mix::PowOp>(loc, constant108, dev106);

  // line 112: torch.aten.reciprocal
  auto reciprocal112 = builder.create<mix::ReciprocalOp>(loc, pow110);

  // line 114: torch.constant.float
  auto scalar114 = FloatAttr::get(type_f64, 1.000000e+00);
  auto constant114 = builder.create<mix::ConstantOp>(loc, scalar114);

  // line 116: torch.aten.mul.Scalar
  auto mul116 = builder.create<mix::MulOp>(loc, constant114, reciprocal112);

  // line 123: torch.aten.arange
  SmallVector<float> tmp123(8192);
  for (int i = 0; i < 8192; i++) {
    tmp123[i] = i;
  }
  auto dense123 = DenseElementsAttr::get(
      RankedTensorType::get({8192}, type_f32), ArrayRef<float>(tmp123));
  auto constant123 = builder.create<mix::ConstantOp>(loc, dense123);

  // line 126: torch.aten.unsqueeze
  auto unsqueeze126 = builder.create<mix::ReshapeOp>(
      loc, constant123, createIntArrayAttr(context, {8192, 1}));

  // line 131: torch.aten.permute
  auto permute131 = builder.create<mix::PermuteOp>(
      loc, unsqueeze126, createIntArrayAttr(context, {0, 1}));

  // line 134: torch.aten.unsqueeze
  auto unsqueeze134 = builder.create<mix::ReshapeOp>(
      loc, mul116, createIntArrayAttr(context, {80, 1}));

  // line 139: torch.aten.permute
  auto permute139 = builder.create<mix::PermuteOp>(
      loc, unsqueeze134, createIntArrayAttr(context, {1, 0}));

  // line 141: torch.aten.mul.Tensor
  auto mul141 = builder.create<mix::MulOp>(loc, permute131, permute139);

  // line 145: torch.aten.cat
  SmallVector<Value> tmp145{mul141, mul141};
  auto cat145 = builder.create<mix::ConcatOp>(
      loc, tmp145, IntegerAttr::get(IntegerType::get(&context, 64), 1));

  // line 147: torch.aten.cos
  auto cos147 = builder.create<mix::CosOp>(loc, cat145);

  // line 153: torch.aten.slice.Tensor
  auto slice148 = builder.create<mix::SliceOp>(loc, cos147, 0, 0, INT32_MAX, 1);

  // line 156: torch.aten.unsqueeze
  auto unsqueeze156 = builder.create<mix::ReshapeOp>(
      loc, slice148, createIntArrayAttr(context, {8192, 1, 160}));

  // line 162: torch.aten.slice.Tensor
  auto slice162 =
      builder.create<mix::SliceOp>(loc, unsqueeze156, 2, 0, INT32_MAX, 1);

  // line 164: torch.constant.float
  auto scalar164 = FloatAttr::get(type_f64, 1.000000e+00);
  auto constant164 = builder.create<mix::ConstantOp>(loc, scalar164);

  // line 166: torch.aten.mul.Scalar
  auto mul166 = builder.create<mix::MulOp>(loc, slice162, constant164);

  // line 168: torch.aten.sin
  auto sin168 = builder.create<mix::SinOp>(loc, cat145);

  // line 174: torch.aten.slice.Tensor
  auto slice174 = builder.create<mix::SliceOp>(loc, sin168, 0, 0, INT32_MAX, 1);

  // line 177: torch.aten.unsqueeze
  auto unsqueeze177 = builder.create<mix::ReshapeOp>(
      loc, slice174, createIntArrayAttr(context, {8192, 1, 160}));

  // line 183: torch.aten.slice.Tensor
  auto slice183 =
      builder.create<mix::SliceOp>(loc, unsqueeze177, 2, 0, INT32_MAX, 1);

  // line 185: torch.constant.float
  auto scalar185 = FloatAttr::get(type_f64, 1.000000e+00);
  auto constant185 = builder.create<mix::ConstantOp>(loc, scalar185);

  // line 187: torch.aten.mul.Scalar
  auto mul187 = builder.create<mix::MulOp>(loc, slice183, constant185);

  return {mul166, mul187};
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
  auto type_f32 = builder.getF32Type();
  auto type_f64 = builder.getF16Type();
  auto type_query_weight =
      RankedTensorType::get({hidden_size, hidden_size}, type_f32);
  auto type_key_value_weight =
      RankedTensorType::get({hidden_size * 2, hidden_size}, type_f32);
  auto type_dense_weight =
      RankedTensorType::get({hidden_size, hidden_size}, type_f32);
  auto type_dense_bias = RankedTensorType::get({hidden_size}, type_f32);

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

  // 下面是RotaryEmbedding 的代码，应该在genRotaryEmbedding中实现

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