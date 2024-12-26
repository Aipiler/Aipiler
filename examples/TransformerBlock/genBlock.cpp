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

const int seq_len = 40;
const int hidden_size = 5120;

ArrayAttr createIntArrayAttr(MLIRContext &context,
                             const std::vector<int64_t> &values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());

  for (auto value : values) {
    attrs.push_back(IntegerAttr::get(IntegerType::get(&context, 64), value));
  }

  return ArrayAttr::get(&context, attrs);
}

auto genFFN(mlir::MLIRContext &context, mlir::OpBuilder &builder, Location loc,
            Value hidden_states, Value residual, const std::string) {

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

std::pair<Value, Value> genRotaryEmbedding(mlir::MLIRContext &context,
                                           mlir::OpBuilder &builder,
                                           Location loc) {

  // hidden_states dims
  auto hidden_size = 5120;
  auto n_head = 32;
  auto head_dim = hidden_size / n_head;
  auto key_value_projection_size = hidden_size * 2;
  auto key_value_projection_head_dim = key_value_projection_size / n_head;

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

  // attrs:
  auto attr_i32_n1 = IntegerAttr::get(IntegerType::get(&context, 32), -1);
  auto attr_i32_0 = IntegerAttr::get(IntegerType::get(&context, 32), 0);
  auto attr_i32_1 = IntegerAttr::get(IntegerType::get(&context, 32), 1);
  auto attr_i32_2 = IntegerAttr::get(IntegerType::get(&context, 32), 2);
  auto attr_i32_3 = IntegerAttr::get(IntegerType::get(&context, 32), 3);

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
  SmallVector<float> tmp123(seq_len);
  for (int i = 0; i < seq_len; i++) {
    tmp123[i] = i;
  }
  auto dense123 = DenseElementsAttr::get(
      RankedTensorType::get({seq_len}, type_f32), ArrayRef<float>(tmp123));
  auto constant123 = builder.create<mix::ConstantOp>(loc, dense123);

  // line 126: torch.aten.unsqueeze
  auto unsqueeze126 =
      builder.create<mix::UnsqueezeOp>(loc, constant123, attr_i32_1);

  // line 131: torch.aten.permute
  auto permute131 = builder.create<mix::PermuteOp>(
      loc, unsqueeze126, createIntArrayAttr(context, {0, 1}));

  // line 134: torch.aten.unsqueeze
  auto unsqueeze134 = builder.create<mix::UnsqueezeOp>(loc, mul116, attr_i32_1);

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
  auto unsqueeze156 =
      builder.create<mix::UnsqueezeOp>(loc, slice148, attr_i32_1);

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
  auto unsqueeze177 =
      builder.create<mix::UnsqueezeOp>(loc, slice174, attr_i32_1);

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
                 Location loc, Value hidden_states, Value residual,
                 Value attention_mask) {

  auto hidden_states_type = hidden_states.getType();

  // hidden_states dims
  auto hidden_states_shape = hidden_states_type.getShape();
  auto batch_size = hidden_states_shape[0];
  // auto seq_length = hidden_states_shape[1];
  auto hidden_size = hidden_states_shape[2];
  auto n_head = 32;
  auto head_dim = hidden_size / n_head;
  auto key_value_projection_size = hidden_size * 2;
  auto key_value_projection_head_dim = key_value_projection_size / n_head;
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

  // attrs:
  auto attr_i32_n1 = builder.getSI32IntegerAttr(-1);
  auto attr_i32_0 = IntegerAttr::get(IntegerType::get(&context, 32), 0);
  auto attr_i32_1 = IntegerAttr::get(IntegerType::get(&context, 32), 1);
  auto attr_i32_2 = IntegerAttr::get(IntegerType::get(&context, 32), 2);
  auto attr_i32_3 = IntegerAttr::get(IntegerType::get(&context, 32), 3);

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
  auto transpose14 = builder.create<mix::TransposeOp>(loc, hidden_states,
                                                      attr_i32_0, attr_i32_1);

  // line 16: torch.aten.t
  auto t16 = builder.create<mix::TransposeOp>(loc, query_weight, attr_i32_0,
                                              attr_i32_1);

  // line 21: torch.aten.view
  auto reshape21 = builder.create<mix::ReshapeOp>(
      loc, transpose14, createIntArrayAttr(context, {seq_len, hidden_size}));

  // line 23: torch.aten.mm
  auto matmul23 = builder.create<mix::MatMulOp>(loc, reshape21, t16);

  // line 29: torch.aten.view
  auto reshape29 = builder.create<mix::ReshapeOp>(
      loc, matmul23,
      createIntArrayAttr(context, {seq_len, batch_size, hidden_size}));

  // line 36: torch.aten.view
  auto reshape36 = builder.create<mix::ReshapeOp>(
      loc, reshape29,
      createIntArrayAttr(context, {seq_len, batch_size, n_head, head_dim}));

  // line 38: torch.aten.t
  auto t38 = builder.create<mix::TransposeOp>(loc, key_value_weight, attr_i32_0,
                                              attr_i32_1);

  // line 43: torch.aten.view
  auto reshape43 = builder.create<mix::ReshapeOp>(
      loc, transpose14, createIntArrayAttr(context, {seq_len, hidden_size}));

  // line 45: torch.aten.mm
  auto matmul45 = builder.create<mix::MatMulOp>(loc, reshape43, t38);

  // line 51: torch.aten.view
  auto reshape51 = builder.create<mix::ReshapeOp>(
      loc, matmul45,
      createIntArrayAttr(context,
                         {seq_len, batch_size, key_value_projection_size}));

  // line 58: torch.aten.view
  auto reshape58 = builder.create<mix::ReshapeOp>(
      loc, reshape51,
      createIntArrayAttr(context, {seq_len, batch_size, n_head,
                                   key_value_projection_head_dim}));

  // line 64: torch.aten.slice.Tensor
  auto slice64 = builder.create<mix::SliceOp>(loc, reshape58, 3, 0, 160, 1);

  // line 70: torch.aten.slice.Tensor
  auto slice70 = builder.create<mix::SliceOp>(loc, reshape58, 3, 160, 320, 1);

  // line 76: torch.aten.view
  auto reshape76 = builder.create<mix::ReshapeOp>(
      loc, reshape36, createIntArrayAttr(context, {seq_len, n_head, -1}));

  // line 82: torch.aten.view
  auto reshape82 = builder.create<mix::ReshapeOp>(
      loc, slice64, createIntArrayAttr(context, {seq_len, n_head, -1}));

  // 下面是RotaryEmbedding 的代码

  auto [cos, sin] = genRotaryEmbedding(context, builder, loc);

  /* 下面是apply_rotary_pos_emb_torch中的代码 */

  // line 201: torch.aten.slice.Tensor
  auto slice201 = builder.create<mix::SliceOp>(loc, cos, 0, 0, seq_len, 1);

  // line 207: torch.aten.slice.Tensor
  auto slice207 = builder.create<mix::SliceOp>(loc, sin, 0, 0, seq_len, 1);

  // line 209: torch.aten.mul.Tensor
  auto mul209 = builder.create<mix::MulOp>(loc, reshape76, slice201);

  // line 215: torch.aten.slice.Tensor
  auto slice215 = builder.create<mix::SliceOp>(loc, reshape76, 2, 0, 80, 1);

  // line 221: torch.aten.slice.Tensor
  auto slice221 =
      builder.create<mix::SliceOp>(loc, reshape76, 2, 80, INT32_MAX, 1);

  // line 223: torch.aten.neg
  auto neg223 = builder.create<mix::NegOp>(loc, slice221);

  // line 227: torch.aten.cat
  SmallVector<Value> tmp227{neg223, slice215};
  auto cat227 = builder.create<mix::ConcatOp>(
      loc, tmp227, IntegerAttr::get(IntegerType::get(&context, 64), 2));

  // line 229: torch.aten.mul.Tensor
  auto mul229 = builder.create<mix::MulOp>(loc, cat227, slice207);

  // line 232: torch.aten.add.Tensor
  auto add232 = builder.create<mix::AddOp>(loc, mul209, mul229);

  // line 234: torch.aten.mul.Tensor
  auto mul234 = builder.create<mix::MulOp>(loc, reshape82, slice201);

  // line 240: torch.aten.slice.Tensor
  auto slice240 = builder.create<mix::SliceOp>(loc, reshape82, 2, 0, 80, 1);

  // line 246: torch.aten.slice.Tensor
  auto slice246 =
      builder.create<mix::SliceOp>(loc, reshape82, 2, 80, INT32_MAX, 1);

  // line 248: torch.aten.neg
  auto neg248 = builder.create<mix::NegOp>(loc, slice246);

  // line 252: torch.aten.cat
  SmallVector<Value> tmp252{neg248, slice240};
  auto cat252 = builder.create<mix::ConcatOp>(
      loc, tmp252, IntegerAttr::get(IntegerType::get(&context, 64), 2));

  // line 254: torch.aten.mul.Tensor
  auto mul254 = builder.create<mix::MulOp>(loc, cat252, slice207);

  // line 257: torch.aten.add.Tensor
  auto add257 = builder.create<mix::AddOp>(loc, mul234, mul254);

  /* 上面是apply_rotary_pos_emb_torch中的代码 */

  // line 267: torch.aten.view
  auto reshape267 = builder.create<mix::ReshapeOp>(
      loc, add232, createIntArrayAttr(context, {seq_len, 1, n_head, 160}));

  // line 274: torch.aten.view
  auto reshape274 = builder.create<mix::ReshapeOp>(
      loc, add257, createIntArrayAttr(context, {seq_len, 1, n_head, 160}));

  // line 280: torch.aten.view
  auto reshape280 = builder.create<mix::ReshapeOp>(
      loc, reshape267, createIntArrayAttr(context, {seq_len, n_head, 160}));

  // line 286: torch.aten.view
  auto reshape286 = builder.create<mix::ReshapeOp>(
      loc, reshape274, createIntArrayAttr(context, {seq_len, n_head, 160}));

  // line 290: torch.aten.transpose.int
  auto transpose290 =
      builder.create<mix::TransposeOp>(loc, reshape280, attr_i32_0, attr_i32_1);

  // line 294: torch.aten.transpose.int
  auto transpose294 =
      builder.create<mix::TransposeOp>(loc, reshape286, attr_i32_0, attr_i32_1);

  // line 298: torch.aten.transpose.int
  auto transpose298 = builder.create<mix::TransposeOp>(loc, transpose294,
                                                       attr_i32_1, attr_i32_2);

  // line 301: torch.aten.unsqueeze
  auto unsqueeze301 =
      builder.create<mix::UnsqueezeOp>(loc, transpose290, attr_i32_3);

  // line 308: torch.aten.permute
  auto permute308 = builder.create<mix::PermuteOp>(
      loc, unsqueeze301, createIntArrayAttr(context, {0, 1, 3, 2}));

  // line 311: torch.aten.unsqueeze
  auto unsqueeze311 =
      builder.create<mix::UnsqueezeOp>(loc, transpose298, attr_i32_3);

  // line 318: torch.aten.permute
  auto permute318 = builder.create<mix::PermuteOp>(
      loc, unsqueeze311, createIntArrayAttr(context, {0, 3, 2, 1}));

  // line 325: torch.aten.permute
  auto permute325 = builder.create<mix::PermuteOp>(
      loc, permute308, createIntArrayAttr(context, {0, 1, 3, 2}));

  // line 331: torch.aten.view
  auto reshape331 = builder.create<mix::ReshapeOp>(
      loc, permute325, createIntArrayAttr(context, {n_head, seq_len, 160}));

  // line 338: torch.aten.permute
  auto permute338 = builder.create<mix::PermuteOp>(
      loc, permute318, createIntArrayAttr(context, {0, 3, 2, 1}));

  // line 344: torch.aten.view
  auto reshape344 = builder.create<mix::ReshapeOp>(
      loc, permute338, createIntArrayAttr(context, {n_head, 160, seq_len}));

  // line 346: torch.aten.bmm
  auto bmm346 = builder.create<mix::BatchMatMulOp>(loc, reshape331, reshape344);

  // line 353: torch.aten.view
  auto reshape353 = builder.create<mix::ReshapeOp>(
      loc, bmm346, createIntArrayAttr(context, {n_head, seq_len, 1, seq_len}));

  // line 360: torch.aten.permute
  auto permute360 = builder.create<mix::PermuteOp>(
      loc, reshape353, createIntArrayAttr(context, {0, 1, 3, 2}));

  // line 366: torch.aten.view
  auto reshape366 = builder.create<mix::ReshapeOp>(
      loc, permute360, createIntArrayAttr(context, {n_head, seq_len, seq_len}));

  // line 368: torch.constant.float
  auto scalar368 = FloatAttr::get(type_f64, 0.079056941504209485);
  auto constant368 = builder.create<mix::ConstantOp>(loc, scalar368);

  // line 370: torch.aten.mul.Scalar
  auto mul370 = builder.create<mix::MulOp>(loc, reshape366, constant368);

  // line 377: torch.aten.view
  auto reshape377 = builder.create<mix::ReshapeOp>(
      loc, mul370, createIntArrayAttr(context, {1, n_head, seq_len, seq_len}));

  // line 380: torch.aten.masked_fill.Scalar
  auto masked_fill380 = builder.create<mix::MaskedFillOp>(
      loc, reshape377, attention_mask,
      FloatAttr::get(Float64Type::get(&context), -3.4028234663852886E+38));

  // line 384: torch.aten._softmax
  auto softmax384 =
      builder.create<mix::SoftmaxOp>(loc, masked_fill380, attr_i32_n1);

  // line 393: torch.aten.view
  auto reshape393 = builder.create<mix::ReshapeOp>(
      loc, softmax384, createIntArrayAttr(context, {n_head, seq_len, seq_len}));

  // line 399: torch.aten.view
  auto reshape399 = builder.create<mix::ReshapeOp>(
      loc, slice70, createIntArrayAttr(context, {seq_len, n_head, 160}));

  // line 403: torch.aten.transpose.int
  auto transpose403 =
      builder.create<mix::TransposeOp>(loc, reshape399, attr_i32_0, attr_i32_1);

  // line 405: torch.aten.bmm
  auto bmm405 =
      builder.create<mix::BatchMatMulOp>(loc, reshape393, transpose403);

  // line 412: torch.aten.view
  auto reshape412 = builder.create<mix::ReshapeOp>(
      loc, bmm405, createIntArrayAttr(context, {1, n_head, seq_len, 160}));

  // line 419: torch.aten.permute
  auto permute419 = builder.create<mix::PermuteOp>(
      loc, reshape412, createIntArrayAttr(context, {0, 2, 1, 3}));

  // line 428: torch.aten.view
  auto reshape428 = builder.create<mix::ReshapeOp>(
      loc, permute419, createIntArrayAttr(context, {1, seq_len, hidden_size}));

  // line 433: torch.aten.view
  auto reshape433 = builder.create<mix::ReshapeOp>(
      loc, reshape428, createIntArrayAttr(context, {seq_len, hidden_size}));

  // line 435: torch.aten.transpose.int
  auto transpose435 = builder.create<mix::TransposeOp>(loc, dense_weight,
                                                       attr_i32_0, attr_i32_1);

  // line 439: torch.aten.addmm
  auto matmul439 = builder.create<mix::MatMulOp>(loc, reshape433, transpose435);
  auto add439 = builder.create<mix::AddOp>(loc, matmul439, dense_bias);

  // line 445: torch.aten.view
  auto reshape445 = builder.create<mix::ReshapeOp>(
      loc, add439, createIntArrayAttr(context, {1, seq_len, hidden_size}));

  // line 451: torch.aten.add.Tensor
  auto add451 = builder.create<mix::MulOp>(loc, residual, reshape445);

  return add451;
}

auto genRMSNorm(mlir::MLIRContext &context, mlir::OpBuilder &builder,
                Location loc, Value hidden_states,
                const std::string weight_loc) {

  auto hidden_size = 5120;
  auto eps = APFloat(1e-6);

  auto hidden_states_type =
      mlir::dyn_cast<RankedTensorType>(hidden_states.getType());
  auto hidden_states_shape = hidden_states_type.getShape();
  auto hidden_states_rank = hidden_states_shape.size();
  auto elementType = hidden_states_type.getElementType();
  llvm::ArrayRef<int64_t> weightShape{int64_t(hidden_size)};
  auto weightTensorType = RankedTensorType::get(weightShape, elementType);

  auto _weight3 =
      builder.create<mix::WeightOp>(loc, weightTensorType, weight_loc);

  auto constantTensorType = RankedTensorType::get({1}, elementType);
  auto constantTensor = DenseElementsAttr::get(constantTensorType, {2.0f});
  auto c2Tensor = builder.create<arith::ConstantOp>(loc, constantTensor);
  auto pow0 = builder.create<mix::PowOp>(loc, hidden_states, c2Tensor);
  auto mean0 = builder.create<mix::MeanOp>(
      loc, pow0, builder.getI32ArrayAttr({int32_t(hidden_states_rank - 1)}),
      builder.getBoolAttr(true));

  auto epsAttr = builder.getFloatAttr(elementType, eps);
  auto const_eps = builder.create<arith::ConstantOp>(loc, epsAttr);
  auto add0 = builder.create<mix::AddOp>(loc, mean0, const_eps);
  auto rsqrt0 = builder.create<mix::RsqrtOp>(loc, add0);
  auto mul0 = builder.create<mix::MulOp>(loc, hidden_states, rsqrt0);
  auto mul1 = builder.create<mix::MulOp>(loc, _weight3, mul0);

  return mul1;
}

auto genTransformerBlock(mlir::MLIRContext &context, mlir::OpBuilder &builder,
                         Location loc, Value hidden_states,
                         Value attention_mask) {

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

  auto elementType = builder.getF32Type();

  auto functionTy = builder.getFunctionType({}, {});
  auto graph0 = builder.create<func::FuncOp>(loc, "self_attention", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto hidden_states_type =
      RankedTensorType::get({1, seq_len, hidden_size}, elementType);
  auto attention_mask_type =
      RankedTensorType::get({1, 1, seq_len, seq_len}, builder.getI1Type());

  auto hidden_states = builder.create<mix::WeightOp>(
      graph0->getLoc(), hidden_states_type, "hidden_states");

  auto attention_mask = builder.create<mix::WeightOp>(
      graph0->getLoc(), attention_mask_type, "attention_mask");

  auto transformerBlock =
      genTransformerBlock(context, builder, loc, hidden_states, attention_mask);

  builder.create<func::ReturnOp>(loc, ValueRange{transformerBlock});

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
