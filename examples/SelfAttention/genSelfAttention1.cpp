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
#include "mlir/Dialect/Affine/Passes.h"
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

  // attrs:
  auto attr_i32_n1 = builder.getSI32IntegerAttr(-1);
  auto attr_i32_0 = IntegerAttr::get(IntegerType::get(&context, 32), 0);
  auto attr_i32_1 = IntegerAttr::get(IntegerType::get(&context, 32), 1);
  auto attr_i32_2 = IntegerAttr::get(IntegerType::get(&context, 32), 2);
  auto attr_i32_3 = IntegerAttr::get(IntegerType::get(&context, 32), 3);

  /* 定义算子 */

  auto linearQ = builder.create<mix::LinearOp>(
      loc, hidden_states, "query", hidden_size, hidden_size, false, type_f16);

  auto reshapeQ = builder.create<mix::ReshapeOp>(
      loc, linearQ,
      createIntArrayAttr(context, {max_seq_len, n_head, head_dim}));

  auto linearKV = builder.create<mix::LinearOp>(
      loc, hidden_states, "key_value", hidden_size, key_value_projection_size,
      false, type_f16);

  // line 58: torch.aten.view
  auto reshapeKV = builder.create<mix::ReshapeOp>(
      loc, linearKV,
      createIntArrayAttr(context,
                         {max_seq_len, n_head, key_value_projection_head_dim}));

  // line 64: torch.aten.slice.Tensor
  auto sliceK = builder.create<mix::SliceOp>(loc, reshapeKV, 2, 0, 160, 1);

  auto reshapeK = builder.create<mix::ReshapeOp>(
      loc, sliceK, createIntArrayAttr(context, {max_seq_len, hidden_size}));

  // line 70: torch.aten.slice.Tensor
  auto sliceV = builder.create<mix::SliceOp>(loc, reshapeKV, 2, 160, 320, 1);

  // 下面是RotaryEmbedding 的代码

  auto [cos, sin] = genRotaryEmbedding(context, builder, loc);

  /* 下面是apply_rotary_pos_emb_torch中的代码 */

  // line 209: torch.aten.mul.Tensor
  auto mul209 = builder.create<mix::MulOp>(loc, reshapeQ, cos);

  // line 215: torch.aten.slice.Tensor
  auto slice215 = builder.create<mix::SliceOp>(loc, reshapeQ, 2, 0, 80, 1);

  // line 221: torch.aten.slice.Tensor
  auto slice221 = builder.create<mix::SliceOp>(loc, reshapeQ, 2, 80, 160, 1);

  // line 223: torch.aten.neg
  auto neg223 = builder.create<mix::NegOp>(loc, slice221);

  // line 227: torch.aten.cat
  SmallVector<Value> tmp227{neg223, slice215};
  auto cat227 = builder.create<mix::ConcatOp>(
      loc, tmp227, IntegerAttr::get(IntegerType::get(&context, 64), 2));

  // line 229: torch.aten.mul.Tensor
  auto mul229 = builder.create<mix::MulOp>(loc, cat227, sin);

  // line 232: torch.aten.add.Tensor
  auto add232 = builder.create<mix::AddOp>(loc, mul209, mul229);

  // 以上是：(q * cos) + (rotate_half(q) * sin)

  // 下面是：(k * cos) + (rotate_half(k) * sin)

  // line 234: torch.aten.mul.Tensor
  auto mul234 = builder.create<mix::MulOp>(loc, sliceK, cos);

  // line 240: torch.aten.slice.Tensor
  auto slice240 = builder.create<mix::SliceOp>(loc, sliceK, 2, 0, 80, 1);

  // line 246: torch.aten.slice.Tensor
  auto slice246 = builder.create<mix::SliceOp>(loc, sliceK, 2, 80, 160, 1);

  // line 248: torch.aten.neg
  auto neg248 = builder.create<mix::NegOp>(loc, slice246);

  // line 252: torch.aten.cat
  SmallVector<Value> tmp252{neg248, slice240};
  auto cat252 = builder.create<mix::ConcatOp>(
      loc, tmp252, IntegerAttr::get(IntegerType::get(&context, 64), 2));

  // line 254: torch.aten.mul.Tensor
  auto mul254 = builder.create<mix::MulOp>(loc, cat252, sin);

  // line 257: torch.aten.add.Tensor
  auto add257 = builder.create<mix::AddOp>(loc, mul234, mul254);

  /* 上面是apply_rotary_pos_emb_torch中的代码 */

  auto MNHQ = builder.create<mix::ReshapeOp>(
      loc, add232,
      createIntArrayAttr(context, {max_seq_len, n_head, head_dim}));

  auto MNHK = builder.create<mix::ReshapeOp>(
      loc, add257,
      createIntArrayAttr(context, {max_seq_len, n_head, head_dim}));

  // line 290: torch.aten.transpose.int
  auto transpose290 =
      builder.create<mix::TransposeOp>(loc, MNHQ, attr_i32_0, attr_i32_1);

  // line 294: torch.aten.transpose.int
  auto transpose294 =
      builder.create<mix::TransposeOp>(loc, MNHK, attr_i32_0, attr_i32_1);

  auto cast_MNHQ = builder.create<tensor::CastOp>(
      loc, UnrankedTensorType::get(type_f16), MNHQ);
  builder.create<func::CallOp>(loc, printMemRefFunc, ValueRange{cast_MNHQ});

  auto cast_MNHK = builder.create<tensor::CastOp>(
      loc, UnrankedTensorType::get(type_f16), MNHK);
  builder.create<func::CallOp>(loc, printMemRefFunc, ValueRange{cast_MNHK});

  // line 298: torch.aten.transpose.int
  auto transpose298 = builder.create<mix::TransposeOp>(loc, transpose294,
                                                       attr_i32_1, attr_i32_2);

  // line 346: torch.aten.bmm
  auto bmm346 =
      builder.create<mix::BatchMatMulOp>(loc, transpose290, transpose298);

  // line 368: torch.constant.float
  auto scalar368 = FloatAttr::get(type_f16, 0.079056941504209485);
  auto constant368 = builder.create<mix::ConstantOp>(loc, scalar368);

  // line 370: torch.aten.mul.Scalar
  auto mul370 = builder.create<mix::MulOp>(loc, bmm346, constant368);

  auto cast_bmm346 = builder.create<tensor::CastOp>(
      loc, UnrankedTensorType::get(type_f16), bmm346);
  builder.create<func::CallOp>(loc, printMemRefFunc, ValueRange{cast_bmm346});

  // line 380: torch.aten.masked_fill.Scalar
  auto constant380 = builder.create<mix::ConstantOp>(
      loc,
      builder.getFloatAttr(type_f16, llvm::getAPFloatFromSize(-65504.0f, 16)));
  auto masked_fill380 = builder.create<mix::MaskedFillOp>(
      loc, mul370, attention_mask, constant380);

  auto cast_mul370 = builder.create<tensor::CastOp>(
      loc, UnrankedTensorType::get(type_f16), mul370);
  builder.create<func::CallOp>(loc, printMemRefFunc, ValueRange{cast_mul370});

  // line 384: torch.aten._softmax
  auto softmax384 =
      builder.create<mix::SoftmaxOp>(loc, masked_fill380, attr_i32_n1);

  //   auto cast_masked_fill380 = builder.create<tensor::CastOp>(
  //       loc, UnrankedTensorType::get(type_f16), masked_fill380);
  //   builder.create<func::CallOp>(loc, printMemRefFunc,
  //                                ValueRange{cast_masked_fill380});

  // line 403: torch.aten.transpose.int
  auto transpose403 =
      builder.create<mix::TransposeOp>(loc, sliceV, attr_i32_0, attr_i32_1);

  // auto cast_sliceV = builder.create<tensor::CastOp>(
  //     loc, UnrankedTensorType::get(type_f16), sliceV);
  // builder.create<func::CallOp>(loc, printMemRefFunc,
  // ValueRange{cast_sliceV});

  // line 405: torch.aten.bmm
  auto bmm405 =
      builder.create<mix::BatchMatMulOp>(loc, softmax384, transpose403);

  //   auto cast_softmax384 = builder.create<tensor::CastOp>(
  //       loc, UnrankedTensorType::get(type_f16), softmax384);
  //   builder.create<func::CallOp>(loc, printMemRefFunc,
  //                                ValueRange{cast_softmax384});

  // 下面是merge_heads

  //   auto cast_reshape399 = builder.create<tensor::CastOp>(
  //       loc, UnrankedTensorType::get(type_f16), reshape399);
  //   builder.create<func::CallOp>(loc, printMemRefFunc,
  //                                ValueRange{cast_reshape399});

  // line 419: torch.aten.permute
  auto permute419 = builder.create<mix::PermuteOp>(
      loc, bmm405, createIntArrayAttr(context, {1, 0, 2}));

  // auto cast_bmm405 = builder.create<tensor::CastOp>(
  //     loc, UnrankedTensorType::get(type_f16), bmm405);
  // builder.create<func::CallOp>(loc, printMemRefFunc,
  // ValueRange{cast_bmm405});

  // line 428: torch.aten.view
  auto reshape428 = builder.create<mix::ReshapeOp>(
      loc, permute419, createIntArrayAttr(context, {max_seq_len, hidden_size}));

  auto linearD = builder.create<mix::LinearOp>(
      loc, reshape428, "dense", hidden_size, hidden_size, true, type_f16);

  // line 451: torch.aten.add.Tensor
  auto add451 = builder.create<mix::AddOp>(loc, residual, linearD);

  // auto cast_linearD = builder.create<tensor::CastOp>(
  //     loc, UnrankedTensorType::get(type_f16), linearD);
  // builder.create<func::CallOp>(loc, printMemRefFunc,
  // ValueRange{cast_linearD});

  return add451;
}

void generateCode(mlir::ModuleOp &theModule, mlir::OpBuilder &builder,
                  mlir::MLIRContext &context) {

  auto elementType = builder.getF16Type();
  auto hidden_states_type =
      RankedTensorType::get({max_seq_len, hidden_size}, elementType);
  auto residual_type =
      RankedTensorType::get({max_seq_len, hidden_size}, elementType);
  auto attention_mask_type =
      RankedTensorType::get({1, max_seq_len, max_seq_len}, builder.getI1Type());

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
  for (int i = 0; i < max_seq_len * hidden_size; i++) {
    tmp1.push_back(mlir::FloatAttr::get(builder.getF16Type(), float(1)));
  }
  auto hidden_states_attr = DenseElementsAttr::get(hidden_states_type, tmp1);
  auto hidden_states =
      builder.create<arith::ConstantOp>(loc, hidden_states_attr);

  // residual constant
  llvm::SmallVector<mlir::Attribute> tmp2;
  for (int i = 0; i < max_seq_len * hidden_size; i++) {
    tmp2.push_back(mlir::FloatAttr::get(builder.getF16Type(), float(1)));
  }
  auto residual_attr = DenseElementsAttr::get(residual_type, tmp2);
  auto residual = builder.create<arith::ConstantOp>(loc, residual_attr);

  // attention_mask constant
  auto attention_mask_attr =
      DenseElementsAttr::get(attention_mask_type, {false});
  auto attention_mask =
      builder.create<arith::ConstantOp>(loc, attention_mask_attr);
  auto res = builder.create<func::CallOp>(
      loc, graph0, ValueRange{hidden_states, residual, attention_mask});

  //   auto castResult =
  //       builder.create<tensor::CastOp>(loc, printInputType,
  //       res->getResult(0));
  //   builder.create<func::CallOp>(loc, printMemRefFunc,
  //   ValueRange{castResult});
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

  mix::utils::load_model("./attention_model.bin", theModule, builder,
                         builder.getF16Type());

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
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  //   pm.addNestedPass<func::FuncOp>(mlir::affine::createAffineLoopNormalizePass());
  mlir::bufferization::BufferDeallocationPipelineOptions deallocopt;
  mlir::bufferization::buildBufferDeallocationPipeline(pm.nest<ModuleOp>(),
                                                       deallocopt);
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addNestedPass<func::FuncOp>(mlir::LLVM::createRequestCWrappersPass());
  //   pm.addPass(mlir::createConvertSCFToOpenMPPass());
  //   pm.addPass(mlir::createConvertOpenMPToLLVMPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  //   pm.addPass(mlir::createConvertCFToLLVMPass());
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
