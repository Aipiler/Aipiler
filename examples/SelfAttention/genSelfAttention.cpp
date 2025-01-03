#include "Utils/loadPytorchModel.h"
#include "Utils/logger.h"
#include "lld/../../ELF/Config.h"
#include "lld/Common/Driver.h"
#include "mix/mixDialect.h"
#include "mix/mixOps.h"
#include "mix/mixTypes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Conversion/TosaToTensor/TosaToTensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFFormValue.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include <chrono>
#include <iomanip>
#include <iostream>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CodeGen.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
// #include <llvm/Support/Host.h>
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <optional>
#include <sys/types.h>

// LLD_HAS_DRIVER(elf)

namespace lld {
namespace elf {
bool link(llvm::MemoryBuffer *buffer, ArrayRef<const char *> args,
          llvm::raw_ostream &stdoutOS, llvm::raw_ostream &stderrOS,
          bool exitEarly = false, bool disableOutput = false);
} // namespace elf
} // namespace lld

using namespace mlir;
namespace mutil = mix::utils;
using namespace mix;
std::unique_ptr<Pass> createLowerModulePass();
std::unique_ptr<Pass> createLowerCompositePass();
std::unique_ptr<Pass> createLowerPrimaryToTosa();

void registerLowerModulePass();
void registerLowerCompositePass();
void registerLowerPrimaryToTosaPass();

// 模型参数定义
const int max_seq_len = 5;
const int hidden_size = 5120;
const int n_head = 32;
const int head_dim = hidden_size / n_head;
const int batch_size = 1;
const int key_value_projection_size = hidden_size * 2;
const int key_value_projection_head_dim = key_value_projection_size / n_head;

std::string getOpName(std::string_view prefix, int idx, std::string_view name) {
  return std::string(prefix) + std::to_string(idx) + std::string(name);
}

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
  printf("genRotaryEmbedding\n");
  /* 定义一些可重用的信息 */

  // types:
  auto type_i16 = builder.getI32Type();
  auto type_i32 = builder.getI32Type();
  auto type_f16 = builder.getF16Type();
  auto type_f32 = builder.getF32Type();
  auto printMemrefType = UnrankedTensorType::get(type_f16);

  // attrs:
  auto attr_i32_n1 = IntegerAttr::get(IntegerType::get(&context, 32), -1);
  auto attr_i32_0 = IntegerAttr::get(IntegerType::get(&context, 32), 0);
  auto attr_i32_1 = IntegerAttr::get(IntegerType::get(&context, 32), 1);
  auto attr_i32_2 = IntegerAttr::get(IntegerType::get(&context, 32), 2);
  auto attr_i32_3 = IntegerAttr::get(IntegerType::get(&context, 32), 3);

  /* 定义算子 */

  // 下面是RotaryEmbedding 的代码，应该在genRotaryEmbedding中实现

  // line 94: torch.aten.arange.start_step
  llvm::SmallVector<mlir::Attribute> tmp94;
  for (int j = 0; j < head_dim; j += 2) {
    tmp94.push_back(mlir::IntegerAttr::get(type_i16, int16_t(j)));
  }
  auto dense94 = DenseElementsAttr::get(
      RankedTensorType::get({long(tmp94.size())}, type_i16), tmp94);
  auto constant94 = builder.create<mix::ConstantOp>(loc, dense94);

  // line 102: torch.aten._to_copy
  auto convert102 = builder.create<mix::ConvertOp>(loc, constant94, type_f16);

  // line 104: torch.constant.int
  auto scalar101 = FloatAttr::get(type_f16, head_dim);
  auto constant101 = builder.create<mix::ConstantOp>(loc, scalar101);

  // line 106: torch.aten.div.Tensor
  auto dev106 = builder.create<mix::DivOp>(loc, convert102, constant101);

  // line 108: torch.constant.float | base
  auto scalar108 = FloatAttr::get(type_f16, 30420.108888514722);
  auto constant108 = builder.create<mix::ConstantOp>(loc, scalar108);

  // line 110: torch.aten.pow.Scalar
  auto pow110 = builder.create<mix::PowOp>(loc, constant108, dev106);

  // line 112: torch.aten.reciprocal | self.inv_freq
  auto reciprocal112 = builder.create<mix::ReciprocalOp>(loc, pow110);

  // line 114: torch.constant.float
  auto scalar114 = FloatAttr::get(type_f16, 1.000000e+00);
  auto constant114 = builder.create<mix::ConstantOp>(loc, scalar114);

  // line 116: torch.aten.mul.Scalar
  auto mul116 = builder.create<mix::MulOp>(loc, constant114, reciprocal112);

  // line 123: torch.aten.arange
  llvm::SmallVector<mlir::Attribute> tmp123;
  for (int i = 0; i < max_seq_len; ++i) {
    tmp123.push_back(mlir::FloatAttr::get(type_f16, float(i)));
  }
  auto tensorType = RankedTensorType::get({max_seq_len}, type_f16);
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

  // auto cast_mul141 =
  //     builder.create<tensor::CastOp>(loc, printMemrefType, mul141);
  // builder.create<func::CallOp>(loc, printMemRefFunc,
  // ValueRange{cast_mul141});

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
                 Value attention_mask, int idx,
                 mlir::func::FuncOp printMemRefFunc) {
  printf("genSelfAttn\n");
  /* 定义一些可重用的信息 */

  // types:
  auto type_f16 = builder.getF16Type();
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

  // %arg0
  auto query_weight =
      builder.create<mix::WeightOp>(loc, type_query_weight, "query.weight");

  // %arg1
  auto key_value_weight = builder.create<mix::WeightOp>(
      loc, type_key_value_weight, "key_value.weight");

  // %arg2
  auto dense_weight =
      builder.create<mix::WeightOp>(loc, type_dense_weight, "dense.weight");

  // %arg3
  auto dense_bias =
      builder.create<mix::WeightOp>(loc, type_dense_bias, "dense.bias");

  // line 14: torch.aten.transpose.int
  auto transpose14 = builder.create<mix::TransposeOp>(loc, hidden_states,
                                                      attr_i32_0, attr_i32_1);

  // line 16: torch.aten.t
  auto t16 = builder.create<mix::TransposeOp>(loc, query_weight, attr_i32_0,
                                              attr_i32_1);

  // line 21: torch.aten.view
  auto reshape21 = builder.create<mix::ReshapeOp>(
      loc, transpose14,
      createIntArrayAttr(context, {max_seq_len, hidden_size}));

  // line 23: torch.aten.mm -- queary_layer
  auto matmul23 = builder.create<mix::MatMulOp>(loc, reshape21, t16);

  // line 29: torch.aten.view
  auto reshape29 = builder.create<mix::ReshapeOp>(
      loc, matmul23,
      createIntArrayAttr(context, {max_seq_len, batch_size, hidden_size}));

  // line 36: torch.aten.view
  auto reshape36 = builder.create<mix::ReshapeOp>(
      loc, reshape29,
      createIntArrayAttr(context, {max_seq_len, batch_size, n_head, head_dim}));

  // line 38: torch.aten.t
  auto t38 = builder.create<mix::TransposeOp>(loc, key_value_weight, attr_i32_0,
                                              attr_i32_1);

  // line 43: torch.aten.view
  auto reshape43 = builder.create<mix::ReshapeOp>(
      loc, transpose14,
      createIntArrayAttr(context, {max_seq_len, hidden_size}));

  // line 45: torch.aten.mm -- mixed_kv_layer
  auto matmul45 = builder.create<mix::MatMulOp>(loc, reshape43, t38);

  //   auto cast_matmul45 = builder.create<tensor::CastOp>(
  //       loc, UnrankedTensorType::get(type_f16), matmul45);
  //   builder.create<func::CallOp>(loc, printMemRefFunc,
  //   ValueRange{cast_matmul45});

  // line 51: torch.aten.view
  auto reshape51 = builder.create<mix::ReshapeOp>(
      loc, matmul45,
      createIntArrayAttr(context,
                         {max_seq_len, batch_size, key_value_projection_size}));

  // line 58: torch.aten.view
  auto reshape58 = builder.create<mix::ReshapeOp>(
      loc, reshape51,
      createIntArrayAttr(context, {max_seq_len, batch_size, n_head,
                                   key_value_projection_head_dim}));

  // line 64: torch.aten.slice.Tensor
  auto slice64 = builder.create<mix::SliceOp>(loc, reshape58, 3, 0, 160, 1);

  // line 70: torch.aten.slice.Tensor
  auto slice70 = builder.create<mix::SliceOp>(loc, reshape58, 3, 160, 320, 1);

  // line 76: torch.aten.view
  auto reshape76 = builder.create<mix::ReshapeOp>(
      loc, reshape36, createIntArrayAttr(context, {max_seq_len, n_head, -1}));

  // line 82: torch.aten.view -- kv_layer
  auto reshape82 = builder.create<mix::ReshapeOp>(
      loc, slice64, createIntArrayAttr(context, {max_seq_len, n_head, -1}));

  // 下面是RotaryEmbedding 的代码

  auto [cos, sin] = genRotaryEmbedding(context, builder, loc);

  /* 下面是apply_rotary_pos_emb_torch中的代码 */

  // line 201: torch.aten.slice.Tensor
  auto slice201 = builder.create<mix::SliceOp>(loc, cos, 0, 0, max_seq_len, 1);

  // line 207: torch.aten.slice.Tensor
  auto slice207 = builder.create<mix::SliceOp>(loc, sin, 0, 0, max_seq_len, 1);

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
      loc, add232, createIntArrayAttr(context, {max_seq_len, 1, n_head, 160}));

  // line 274: torch.aten.view
  auto reshape274 = builder.create<mix::ReshapeOp>(
      loc, add257, createIntArrayAttr(context, {max_seq_len, 1, n_head, 160}));

  // line 280: torch.aten.view
  auto reshape280 = builder.create<mix::ReshapeOp>(
      loc, reshape267, createIntArrayAttr(context, {max_seq_len, n_head, 160}));

  // line 286: torch.aten.view
  auto reshape286 = builder.create<mix::ReshapeOp>(
      loc, reshape274, createIntArrayAttr(context, {max_seq_len, n_head, 160}));

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
      loc, permute325, createIntArrayAttr(context, {n_head, max_seq_len, 160}));

  // line 338: torch.aten.permute
  auto permute338 = builder.create<mix::PermuteOp>(
      loc, permute318, createIntArrayAttr(context, {0, 3, 2, 1}));

  // line 344: torch.aten.view
  auto reshape344 = builder.create<mix::ReshapeOp>(
      loc, permute338, createIntArrayAttr(context, {n_head, 160, max_seq_len}));

  // line 346: torch.aten.bmm
  auto bmm346 = builder.create<mix::BatchMatMulOp>(loc, reshape331, reshape344);

  // line 353: torch.aten.view
  auto reshape353 = builder.create<mix::ReshapeOp>(
      loc, bmm346,
      createIntArrayAttr(context, {n_head, max_seq_len, 1, max_seq_len}));

  //   auto cast_bmm346 = builder.create<tensor::CastOp>(
  //       loc, UnrankedTensorType::get(type_f16), bmm346);
  //   builder.create<func::CallOp>(loc, printMemRefFunc,
  //   ValueRange{cast_bmm346});

  // line 360: torch.aten.permute
  auto permute360 = builder.create<mix::PermuteOp>(
      loc, reshape353, createIntArrayAttr(context, {0, 1, 3, 2}));

  // line 366: torch.aten.view
  auto reshape366 = builder.create<mix::ReshapeOp>(
      loc, permute360,
      createIntArrayAttr(context, {n_head, max_seq_len, max_seq_len}));

  // line 368: torch.constant.float
  auto scalar368 = FloatAttr::get(type_f16, 0.079056941504209485);
  auto constant368 = builder.create<mix::ConstantOp>(loc, scalar368);

  // line 370: torch.aten.mul.Scalar
  auto mul370 = builder.create<mix::MulOp>(loc, reshape366, constant368);

  // line 377: torch.aten.view
  auto reshape377 = builder.create<mix::ReshapeOp>(
      loc, mul370,
      createIntArrayAttr(context, {1, n_head, max_seq_len, max_seq_len}));

  // line 380: torch.aten.masked_fill.Scalar
  auto constant380 = builder.create<mix::ConstantOp>(
      loc,
      builder.getFloatAttr(type_f16, llvm::getAPFloatFromSize(-65504.0f, 16)));
  auto masked_fill380 = builder.create<mix::MaskedFillOp>(
      loc, reshape377, attention_mask, constant380);

  // line 384: torch.aten._softmax
  auto softmax384 =
      builder.create<mix::SoftmaxOp>(loc, masked_fill380, attr_i32_n1);

  //   auto cast_masked_fill380 = builder.create<tensor::CastOp>(
  //       loc, UnrankedTensorType::get(type_f16), masked_fill380);
  //   builder.create<func::CallOp>(loc, printMemRefFunc,
  //                                ValueRange{cast_masked_fill380});

  // line 393: torch.aten.view
  auto reshape393 = builder.create<mix::ReshapeOp>(
      loc, softmax384,
      createIntArrayAttr(context, {n_head, max_seq_len, max_seq_len}));

  //   auto cast_softmax384 = builder.create<tensor::CastOp>(
  //       loc, UnrankedTensorType::get(type_f16), softmax384);
  //   builder.create<func::CallOp>(loc, printMemRefFunc,
  //                                ValueRange{cast_softmax384});

  // line 399: torch.aten.view
  auto reshape399 = builder.create<mix::ReshapeOp>(
      loc, slice70, createIntArrayAttr(context, {max_seq_len, n_head, 160}));

  // line 403: torch.aten.transpose.int
  auto transpose403 =
      builder.create<mix::TransposeOp>(loc, reshape399, attr_i32_0, attr_i32_1);

  // line 405: torch.aten.bmm
  auto bmm405 =
      builder.create<mix::BatchMatMulOp>(loc, reshape393, transpose403);

  // line 412: torch.aten.view
  auto reshape412 = builder.create<mix::ReshapeOp>(
      loc, bmm405, createIntArrayAttr(context, {1, n_head, max_seq_len, 160}));

  //   auto cast_reshape399 = builder.create<tensor::CastOp>(
  //       loc, UnrankedTensorType::get(type_f16), reshape399);
  //   builder.create<func::CallOp>(loc, printMemRefFunc,
  //                                ValueRange{cast_reshape399});

  //   auto cast_bmm405 = builder.create<tensor::CastOp>(
  //       loc, UnrankedTensorType::get(type_f16), bmm405);
  //   builder.create<func::CallOp>(loc, printMemRefFunc,
  //   ValueRange{cast_bmm405});

  // line 419: torch.aten.permute
  auto permute419 = builder.create<mix::PermuteOp>(
      loc, reshape412, createIntArrayAttr(context, {0, 2, 1, 3}));

  // line 428: torch.aten.view
  auto reshape428 = builder.create<mix::ReshapeOp>(
      loc, permute419,
      createIntArrayAttr(context, {1, max_seq_len, hidden_size}));

  // line 433: torch.aten.view
  auto reshape433 = builder.create<mix::ReshapeOp>(
      loc, reshape428, createIntArrayAttr(context, {max_seq_len, hidden_size}));

  // line 435: torch.aten.transpose.int
  auto transpose435 = builder.create<mix::TransposeOp>(loc, dense_weight,
                                                       attr_i32_0, attr_i32_1);

  // line 439: torch.aten.addmm
  auto matmul439 = builder.create<mix::MatMulOp>(loc, reshape433, transpose435);
  auto add439 = builder.create<mix::AddOp>(loc, matmul439, dense_bias);

  // line 445: torch.aten.view
  auto reshape445 = builder.create<mix::ReshapeOp>(
      loc, add439, createIntArrayAttr(context, {1, max_seq_len, hidden_size}));

  // line 451: torch.aten.add.Tensor
  auto add451 = builder.create<mix::MulOp>(loc, residual, reshape445);

  return add451;
}

void generateCode(mlir::ModuleOp &theModule, mlir::OpBuilder &builder,
                  mlir::MLIRContext &context) {

  auto elementType = builder.getF16Type();
  auto hidden_states_type = RankedTensorType::get(
      {batch_size, max_seq_len, hidden_size}, elementType);
  auto residual_type = RankedTensorType::get(
      {batch_size, max_seq_len, hidden_size}, elementType);
  auto attention_mask_type = RankedTensorType::get(
      {batch_size, batch_size, max_seq_len, max_seq_len}, builder.getI1Type());

  // printMemrefF16
  builder.setInsertionPointToEnd(theModule.getBody());
  auto printInputType = UnrankedTensorType::get(elementType);
  auto printFunTy =
      builder.getFunctionType(TypeRange{printInputType}, TypeRange{});
  auto printMemRefFunc = builder.create<func::FuncOp>(
      theModule->getLoc(), "printMemrefF16", printFunTy);
  printMemRefFunc.setPrivate();

  // Self Attention
  auto functionTy = builder.getFunctionType(
      {hidden_states_type, residual_type, attention_mask_type},
      {hidden_states_type});
  auto graph0 = builder.create<func::FuncOp>(theModule->getLoc(),
                                             "Self_Attention", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);
  auto loc = graph0->getLoc();
  auto output = genSelfAttn(context, builder, graph0->getLoc(),
                            graph0.getArgument(0), graph0.getArgument(1),
                            graph0.getArgument(2), 0, printMemRefFunc);

  builder.create<func::ReturnOp>(graph0->getLoc(), ValueRange{output});

  // main
  builder.setInsertionPointToEnd(theModule.getBody());
  auto mainfunc = builder.create<func::FuncOp>(loc, "main",
                                               builder.getFunctionType({}, {}));
  mainfunc.setPrivate();
  auto mainbody = mainfunc.addEntryBlock();
  builder.setInsertionPointToEnd(mainbody);

  // hidden_states constant
  llvm::SmallVector<mlir::Attribute> tmp1;
  for (int i = 0; i < batch_size * max_seq_len * hidden_size; i++) {
    tmp1.push_back(mlir::FloatAttr::get(builder.getF16Type(), float(1)));
  }
  auto hidden_states_attr = DenseElementsAttr::get(hidden_states_type, tmp1);
  auto hidden_states =
      builder.create<arith::ConstantOp>(loc, hidden_states_attr);

  // residual constant
  llvm::SmallVector<mlir::Attribute> tmp2;
  for (int i = 0; i < batch_size * max_seq_len * hidden_size; i++) {
    tmp2.push_back(mlir::FloatAttr::get(builder.getF16Type(), float(1)));
  }
  auto residual_attr = DenseElementsAttr::get(residual_type, tmp2);
  auto residual = builder.create<arith::ConstantOp>(loc, residual_attr);

  // attention_mask constant
  auto attention_mask_attr =
      DenseElementsAttr::get(attention_mask_type, {true});
  auto attention_mask =
      builder.create<arith::ConstantOp>(loc, attention_mask_attr);
  auto res = builder.create<func::CallOp>(
      loc, graph0, ValueRange{hidden_states, residual, attention_mask});

  auto castCos =
      builder.create<tensor::CastOp>(loc, printInputType, res->getResult(0));
  builder.create<func::CallOp>(loc, printMemRefFunc, ValueRange{castCos});
  builder.create<func::ReturnOp>(loc);
}

void generateExecutable(llvm::Module &module, llvm::StringRef outputFilename) {
  // llvm::InitializeAllTargetInfos();
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();
  std::string targetTriple = llvm::sys::getDefaultTargetTriple();
  module.setTargetTriple(targetTriple);

  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    llvm::errs() << "Error: " << error << "\n";
    return;
  }

  llvm::TargetOptions opt;
  llvm::TargetMachine *targetMachine =
      target->createTargetMachine(targetTriple, "generic", "", opt,
                                  llvm::Reloc::PIC_, llvm::CodeModel::Large);

  module.setDataLayout(targetMachine->createDataLayout());

  llvm::SmallVector<char, 0> buffer;
  llvm::raw_svector_ostream objStream(buffer);

  mutil::log(mutil::LogLevel::INFO, "Start create object.");

  llvm::legacy::PassManager pass;
  if (targetMachine->addPassesToEmitFile(pass, objStream, nullptr,
                                         llvm::CodeGenFileType::ObjectFile)) {
    llvm::errs() << "TargetMachine can't emit a file of this type\n";
    return;
  }

  pass.run(module);
  mutil::log(mutil::LogLevel::INFO, "End create object.");

  llvm::SmallVector<const char *, 32> args = {
      "/home/gaoshihao/project/Aipiler/thirdparty/llvm/build/bin/ld.lld",
      "-z",
      "relro",
      "--hash-style=gnu",
      "--eh-frame-hdr",
      "-m",
      "elf_x86_64",
      "-pie",
      "-dynamic-linker",
      "/lib64/ld-linux-x86-64.so.2",
      "-o",
      outputFilename.data(),
      "/lib/x86_64-linux-gnu/Scrt1.o",
      "/lib/x86_64-linux-gnu/crti.o",
      "/usr/lib/gcc/x86_64-linux-gnu/11/crtbeginS.o",
      "-L/home/gaoshihao/project/Aipiler/thirdparty/llvm/build/lib",
      "-L/usr/lib/gcc/x86_64-linux-gnu/11",
      "-L/usr/lib/gcc/x86_64-linux-gnu/11/../../../../lib64",
      "-L/lib/x86_64-linux-gnu",
      "-L/lib/../lib64",
      "-L/usr/lib/x86_64-linux-gnu",
      "-L/usr/lib/../lib64",
      "-L/lib",
      "-L/usr/lib",
      "from_memory", // load from memory
      "-lgcc",
      "--as-needed",
      "-lgcc_s",
      "--no-as-needed",
      "-lc",
      "-lgcc",
      "-lm",
      "-lmlir_runner_utils",
      "-lmlir_c_runner_utils",
      "--as-needed",
      "-lgcc_s",
      "--no-as-needed",
      "/usr/lib/gcc/x86_64-linux-gnu/11/crtendS.o",
      "/lib/x86_64-linux-gnu/crtn.o"};

  auto objBuffer = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(buffer.data(), buffer.size()), "inMemoryObjectBuffer",
      false);

  mutil::log(mutil::LogLevel::INFO, "Start link object.");

  auto flag = lld::elf::link(objBuffer.get(), args, llvm::outs(), llvm::errs());
  // lld::elf::ctx.driver.addFileFromMemory(objBuffer.get(), false);
  // lld::Result s = lld::lldMain(args, llvm::outs(), llvm::errs(),
  //                              {{lld::Gnu, &lld::elf::link}});

  mutil::log(mutil::LogLevel::INFO, "End link object.");

  if (!flag) {
    llvm::errs() << "Linking failed.\n";
  }
}

int main() {

  // Register all MLIR passes.

  mlir::registerAllPasses();
  mlir::registerAllTranslations();
  mlir::DialectRegistry registry;
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllFromLLVMIRTranslations(registry);
  mlir::registerBuiltinDialectTranslation(registry);

  MLIRContext context(registry);

  registerLowerModulePass();
  registerLowerCompositePass();
  registerLowerPrimaryToTosaPass();
  context.disableMultithreading();
  context.getOrLoadDialect<mix::MIXDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();

  mlir::OpBuilder builder(&context);

  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);
  generateCode(theModule, builder, context);

  load_model("/home/gaoshihao/project/Aipiler/examples/SelfAttention/"
             "attention_model.bin",
             theModule, builder, builder.getF16Type());

  mlir::PassManager pm(&context);
  pm.addPass(createLowerModulePass());
  pm.addPass(createLowerCompositePass());
  pm.addPass(createLowerPrimaryToTosa());
  pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
  pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToLinalg());
  pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToArith());
  pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToTensor());
  pm.addPass(mlir::bufferization::createEmptyTensorToAllocTensorPass());
  mlir::bufferization::OneShotBufferizationOptions opt;
  opt.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(opt));
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  mlir::bufferization::BufferDeallocationPipelineOptions deallocopt;
  mlir::bufferization::buildBufferDeallocationPipeline(pm.nest<ModuleOp>(),
                                                       deallocopt);
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.nest<func::FuncOp>().addPass(mlir::LLVM::createRequestCWrappersPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createConvertMathToLibmPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(pm.run(theModule))) {
    return -1;
  }

  mutil::log(mutil::LogLevel::INFO, "End pass.");

  mutil::log(mutil::LogLevel::INFO, "Start gen LLVM IR.");

  //   translate to llvm ir
  llvm::LLVMContext LLVMContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(theModule, LLVMContext);
  if (!llvmModule) {
    theModule->dump();
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }

  mutil::log(mutil::LogLevel::INFO, "End gen LLVM IR.");

  mutil::log(mutil::LogLevel::INFO, "Start gen exe.");

  generateExecutable(*llvmModule.get(), "Self_attention");

  mutil::log(mutil::LogLevel::INFO, "End gen exe.");

  return 0;
}
