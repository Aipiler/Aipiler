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
#include <tuple>

using namespace mlir;

std::unique_ptr<Pass> createLowerModulePass();

// 模型参数定义
const int seq_len = 40;
int input_len = 40;
const int hidden_size = 5120;
const int ffn_hidden_size = 12288;
const int n_head = 32;
const int head_dim = hidden_size / n_head;
const int key_value_projection_size = hidden_size * 2;
const int key_value_projection_head_dim = key_value_projection_size / n_head;
const int vocab_size = 120000;
const int batch_size = 1;
const int n_layer = 38;

std::string getOpName(std::string_view prefix, int idx, std::string_view name) {
  return std::string(prefix) + std::to_string(idx) + std::string(name);
}

ArrayAttr createIntArrayAttr(mlir::MLIRContext &context,
                             const std::vector<int64_t> &values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());

  for (auto value : values) {
    attrs.push_back(IntegerAttr::get(IntegerType::get(&context, 64), value));
  }

  return ArrayAttr::get(&context, attrs);
}

auto genMLP(mlir::OpBuilder &builder, mlir::Location loc,
            mlir::Value hidden_states, mlir::Value residual, int idx) {
  printf("genMLP\n");
  auto elementType = builder.getF16Type();
  auto gate_proj = builder.create<mix::LinearOp>(
      loc, hidden_states, getOpName("transformer.h.", idx, ".mlp.gate_proj"),
      hidden_size, ffn_hidden_size, false, elementType);

  auto silu0 = builder.create<mix::SiLUOp>(loc, gate_proj);

  auto up_proj = builder.create<mix::LinearOp>(
      loc, hidden_states, getOpName("transformer.h.", idx, ".mlp.up_proj"),
      hidden_size, ffn_hidden_size, false, elementType);
  auto mul0 = builder.create<mix::MulOp>(loc, silu0, up_proj);

  auto down_proj = builder.create<mix::LinearOp>(
      loc, mul0, getOpName("transformer.h.", idx, ".mlp.down_proj"),
      ffn_hidden_size, hidden_size, true, elementType);
  auto output = builder.create<mix::AddOp>(loc, down_proj, residual);
  return output;
}

std::pair<Value, Value> genRotaryEmbedding(mlir::MLIRContext &context,
                                           mlir::OpBuilder &builder,
                                           Location loc) {
  printf("genRotaryEmbedding\n");
  /* 定义一些可重用的信息 */

  // types:
  auto type_i16 = builder.getI32Type();
  auto type_i32 = builder.getI32Type();
  auto type_f16 = builder.getF16Type();
  auto type_f32 = builder.getF32Type();

  // attrs:
  auto attr_i32_n1 = IntegerAttr::get(IntegerType::get(&context, 32), -1);
  auto attr_i32_0 = IntegerAttr::get(IntegerType::get(&context, 32), 0);
  auto attr_i32_1 = IntegerAttr::get(IntegerType::get(&context, 32), 1);
  auto attr_i32_2 = IntegerAttr::get(IntegerType::get(&context, 32), 2);
  auto attr_i32_3 = IntegerAttr::get(IntegerType::get(&context, 32), 3);

  /* 定义算子 */

  // 下面是RotaryEmbedding 的代码，应该在genRotaryEmbedding中实现

  // line 94: torch.aten.arange.start_step
  //   SmallVector<int32_t> tmp94(80);
  //   for (int i = 0, j = 0; i < 80; i++, j += 2) {
  //     tmp94[i] = j;
  //   }

  llvm::SmallVector<mlir::Attribute> tmp94(80);
  for (int i = 0; i < 80; ++i) {
    tmp94[i] = mlir::IntegerAttr::get(type_i16, int16_t(i));
  }

  auto dense94 =
      DenseElementsAttr::get(RankedTensorType::get({80}, type_i16), tmp94);
  auto constant94 = builder.create<mix::ConstantOp>(loc, dense94);

  // line 102: torch.aten._to_copy
  auto convert102 = builder.create<mix::ConvertOp>(loc, constant94, type_f16);

  // line 104: torch.constant.int
  auto scalar101 = IntegerAttr::get(type_i16, 160);
  auto constant101 = builder.create<mix::ConstantOp>(loc, scalar101);

  // line 106: torch.aten.div.Tensor
  auto dev106 = builder.create<mix::DivOp>(loc, convert102, constant101);

  // line 108: torch.constant.float
  auto scalar108 = FloatAttr::get(type_f16, 30420.108888514722);
  auto constant108 = builder.create<mix::ConstantOp>(loc, scalar108);

  // line 110: torch.aten.pow.Scalar
  auto pow110 = builder.create<mix::PowOp>(loc, constant108, dev106);

  // line 112: torch.aten.reciprocal
  auto reciprocal112 = builder.create<mix::ReciprocalOp>(loc, pow110);

  // line 114: torch.constant.float
  auto scalar114 = FloatAttr::get(type_f16, 1.000000e+00);
  auto constant114 = builder.create<mix::ConstantOp>(loc, scalar114);

  // line 116: torch.aten.mul.Scalar
  auto mul116 = builder.create<mix::MulOp>(loc, constant114, reciprocal112);

  // line 123: torch.aten.arange
  //   std::vector<float> tmp123(seq_len);
  //   for (int i = 0; i < seq_len; i++) {
  //     tmp123[i] = i;
  //   }
  llvm::SmallVector<mlir::Attribute> tmp123;
  for (int i = 0; i < seq_len; ++i) {
    tmp123.push_back(mlir::FloatAttr::get(type_f16, 1.0f));
  }

  auto tensorType = RankedTensorType::get({seq_len}, type_f16);

  auto dense123 = DenseElementsAttr::get(tensorType, tmp123);
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
  auto scalar164 = FloatAttr::get(type_f16, 1.000000e+00);
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
  auto scalar185 = FloatAttr::get(type_f16, 1.000000e+00);
  auto constant185 = builder.create<mix::ConstantOp>(loc, scalar185);

  // line 187: torch.aten.mul.Scalar
  auto mul187 = builder.create<mix::MulOp>(loc, slice183, constant185);

  return {mul166, mul187};
}

auto genSelfAttn(mlir::MLIRContext &context, mlir::OpBuilder &builder,
                 Location loc, Value hidden_states, Value residual,
                 Value attention_mask, int idx) {
  printf("genSelfAttn\n");
  auto hidden_states_type =
      mlir::dyn_cast<RankedTensorType>(hidden_states.getType());

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
  auto type_i16 = builder.getI16Type();
  auto type_i32 = builder.getI32Type();
  auto type_i64 = builder.getI64Type();
  auto type_f16 = builder.getF16Type();
  auto type_f32 = builder.getF32Type();
  auto type_f64 = builder.getF64Type();
  auto type_query_weight =
      RankedTensorType::get({hidden_size, hidden_size}, type_f16);
  auto type_key_value_weight =
      RankedTensorType::get({hidden_size * 2, hidden_size}, type_f16);
  auto type_dense_weight =
      RankedTensorType::get({hidden_size, hidden_size}, type_f16);
  auto type_dense_bias = RankedTensorType::get({hidden_size}, type_f16);

  // attrs:
  auto attr_i32_n1 = builder.getSI32IntegerAttr(-1);
  auto attr_i32_0 = IntegerAttr::get(IntegerType::get(&context, 32), 0);
  auto attr_i32_1 = IntegerAttr::get(IntegerType::get(&context, 32), 1);
  auto attr_i32_2 = IntegerAttr::get(IntegerType::get(&context, 32), 2);
  auto attr_i32_3 = IntegerAttr::get(IntegerType::get(&context, 32), 3);

  /* 定义算子 */
  const std::string common = "transformer.h.";

  // %arg0
  auto query_weight = builder.create<mix::WeightOp>(
      loc, type_query_weight,
      getOpName(common, idx, ".self_attention.query.weight"));

  // %arg1
  auto key_value_weight = builder.create<mix::WeightOp>(
      loc, type_key_value_weight,
      getOpName(common, idx, ".self_attention.key_value.weight"));

  // %arg2
  auto dense_weight = builder.create<mix::WeightOp>(
      loc, type_dense_weight,
      getOpName(common, idx, ".self_attention.dense.weight"));

  // %arg3
  auto dense_bias = builder.create<mix::WeightOp>(
      loc, type_dense_bias,
      getOpName(common, idx, ".self_attention.dense.bias"));

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
  auto scalar368 = FloatAttr::get(type_f16, 0.079056941504209485);
  auto constant368 = builder.create<mix::ConstantOp>(loc, scalar368);

  // line 370: torch.aten.mul.Scalar
  auto mul370 = builder.create<mix::MulOp>(loc, reshape366, constant368);

  // line 377: torch.aten.view
  auto reshape377 = builder.create<mix::ReshapeOp>(
      loc, mul370, createIntArrayAttr(context, {1, n_head, seq_len, seq_len}));

  // line 380: torch.aten.masked_fill.Scalar
  auto constant380 = builder.create<mix::ConstantOp>(
      loc, builder.getFloatAttr(type_f16, -3.4028234663852886E+38));
  auto masked_fill380 = builder.create<mix::MaskedFillOp>(
      loc, reshape377, attention_mask, constant380);

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

auto genFusedRMSNorm(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value hidden_states, const std::string &name) {

  printf("genFusedRMSNorm\n");
  auto elementType = builder.getF16Type();
  float eps = 1e-6;
  auto hidden_states_type =
      llvm::dyn_cast<RankedTensorType>(hidden_states.getType());
  auto hidden_states_shape = hidden_states_type.getShape();
  auto hidden_states_rank = hidden_states_shape.size();
  llvm::ArrayRef<int64_t> weightShape{int64_t(hidden_size)};
  auto weightTensorType =
      RankedTensorType::get(weightShape, hidden_states_type.getElementType());

  auto _weight3 = builder.create<mix::WeightOp>(loc, weightTensorType, name);

  llvm::SmallVector<mlir::Attribute> tmp{mlir::FloatAttr::get(elementType, 1)};
  auto constantTensorType = RankedTensorType::get({1}, elementType);
  auto constantTensor = DenseElementsAttr::get(constantTensorType, tmp);
  auto c2Tensor = builder.create<mix::ConstantOp>(loc, constantTensor);
  auto pow0 = builder.create<mix::PowOp>(loc, hidden_states, c2Tensor);
  auto mean0 = builder.create<mix::MeanOp>(
      loc, pow0, builder.getI32ArrayAttr({int32_t(hidden_states_rank - 1)}),
      builder.getBoolAttr(true));

  auto epsAttr = builder.getFloatAttr(elementType, eps);
  auto const_eps = builder.create<mix::ConstantOp>(loc, epsAttr);
  auto add0 = builder.create<mix::AddOp>(loc, mean0, const_eps);
  auto rsqrt0 = builder.create<mix::RsqrtOp>(loc, add0);
  auto mul0 = builder.create<mix::MulOp>(loc, hidden_states, rsqrt0);
  auto mul1 = builder.create<mix::MulOp>(loc, _weight3, mul0);
  return mul1;
}

auto genTransformerBlock(mlir::MLIRContext &context, mlir::OpBuilder &builder,
                         Location loc, Value hidden_states,
                         Value attention_mask, int idx) {
  printf("genTransformerBlock\n");
  // RMSNorm
  auto input_RMSNorm = genFusedRMSNorm(
      builder, loc, hidden_states,
      getOpName("transformer.h.", idx, ".input_layernorm.weight"));

  // Self_attention
  auto self_attn =
      genSelfAttn(context, builder, input_RMSNorm->getLoc(), input_RMSNorm,
                  hidden_states, attention_mask, idx);

  // RMSNorm
  auto post_RMSNorm = genFusedRMSNorm(
      builder, self_attn->getLoc(), self_attn,
      getOpName("transformer.h.", idx, ".post_attention_layernorm.weight"));

  // MLP
  auto FFNoutput =
      genMLP(builder, post_RMSNorm->getLoc(), post_RMSNorm, self_attn, idx);
  return FFNoutput;
}

mlir::Value genEmbedding(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value indices, std::string param_loc,
                         int num_embeddings, int embedding_dim,
                         mlir::Type dtype) {
  printf("genEmbedding\n");
  auto embed0 = builder.create<mix::EmbeddingOp>(
      loc, indices, param_loc, num_embeddings, embedding_dim, dtype);
  return embed0;
}

auto genTelechatModel(mlir::MLIRContext &context, mlir::OpBuilder &builder,
                      Location loc) {
  printf("genTelechatModel\n");
  /* Types */
  auto F16Type = builder.getF16Type();
  auto I32Type = builder.getI32Type();
  auto BoolType = builder.getI1Type();
  auto input_ids_type = RankedTensorType::get({1, seq_len}, I32Type);
  auto output_type = RankedTensorType::get({1, seq_len, vocab_size}, F16Type);
  auto functionTy = builder.getFunctionType({input_ids_type}, {output_type});

  // 创建func.func
  auto graph = builder.create<func::FuncOp>(loc, "Telechat", functionTy);
  graph.setPrivate();
  auto body = graph.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto Argument0 = graph.getArgument(0);
  auto input_ids = mlir::dyn_cast<TypedValue<RankedTensorType>>(Argument0);

  /* 逻辑 */
  Value hidden_states = genEmbedding(builder, graph->getLoc(), input_ids,
                                     "transformer.word_embeddings.weight",
                                     vocab_size, hidden_size, F16Type);

  // TODO: 创建mask，并根据torch-mlir计算mask
  auto attention_mask_tensorType = RankedTensorType::get(
      {batch_size, batch_size, input_len, input_len}, BoolType);
  auto attention_mask_tensor =
      DenseElementsAttr::get(attention_mask_tensorType, {true});
  auto attention_mask =
      builder.create<mix::ConstantOp>(loc, attention_mask_tensor);

  // 循环创建N个Block
  for (int i = 0; i < n_layer; ++i) {
    // transformer block
    hidden_states = genTransformerBlock(context, builder, graph->getLoc(),
                                        hidden_states, attention_mask, i);
  }

  // RMSNorm
  hidden_states = genFusedRMSNorm(builder, graph->getLoc(), hidden_states,
                                  "transformer.ln_f.weight");

  // Linear:将hidden_states映射到vocab_size上
  auto lm_head = builder.create<mix::LinearOp>(graph->getLoc(), hidden_states,
                                               "lm_head.weight", hidden_size,
                                               vocab_size, false, F16Type);
  builder.create<func::ReturnOp>(graph->getLoc(), ValueRange{lm_head});
  // builder.create<func::ReturnOp>(graph->getLoc(), ValueRange{hidden_states});
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
  loc = theModule->getLoc();
  auto telechat_graph = genTelechatModel(context, builder, loc);

  //   theModule->dump();

  mlir::PassManager pm(&context);
  pm.addPass(createLowerModulePass());
  if (mlir::failed(pm.run(theModule))) {
    return -1;
  }
  std::cout << "==== After Lower pass GRAPH-LOWER =====" << std::endl;
  theModule->dump();

  return 0;
}
