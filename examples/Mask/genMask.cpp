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
#include <cstdint>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <optional>
#include <stdio.h>
#include <sys/types.h>

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
const int max_seq_len = 10;
const int hidden_size = 5120;
const int n_head = 32;
const int seq_len = 3;
const int head_dim = hidden_size / n_head;
const int batch_size = 1;
const int key_value_projection_size = hidden_size * 2;
const int key_value_projection_head_dim = key_value_projection_size / n_head;

ArrayAttr createIntArrayAttr(MLIRContext &context,
                             const std::vector<int64_t> &values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());

  for (auto value : values) {
    attrs.push_back(IntegerAttr::get(IntegerType::get(&context, 64), value));
  }

  return ArrayAttr::get(&context, attrs);
}

auto genMask(MLIRContext &context, OpBuilder &builder, Location loc,
             int seq_len, mlir::func::FuncOp printMemRefFunc) {

  // types:
  auto type_i1 = builder.getI1Type();
  auto type_i16 = builder.getI16Type();
  auto type_i32 = builder.getI32Type();
  auto type_i64 = builder.getI64Type();
  auto type_f32 = builder.getF32Type();
  auto type_f16 = builder.getF16Type();
  auto type_f64 = builder.getF64Type();

  // attrs:
  auto attr_i32_n1 = IntegerAttr::get(IntegerType::get(&context, 32), -1);
  auto attr_i32_0 = IntegerAttr::get(IntegerType::get(&context, 32), 0);
  auto attr_i32_1 = IntegerAttr::get(IntegerType::get(&context, 32), 1);
  auto attr_i32_2 = IntegerAttr::get(IntegerType::get(&context, 32), 2);
  auto attr_i32_3 = IntegerAttr::get(IntegerType::get(&context, 32), 3);
  auto attr_i64_1 = IntegerAttr::get(IntegerType::get(&context, 64), 1);
  auto attr_i64_0 = IntegerAttr::get(IntegerType::get(&context, 64), 0);
  // 逻辑

  // line 19 : torch.aten.arange
  SmallVector<int16_t> tmp19(seq_len);
  for (int i = 0; i < seq_len; i++) {
    tmp19[i] = i;
  }
  auto dense19 = DenseElementsAttr::get(
      RankedTensorType::get({seq_len}, type_i16), ArrayRef<int16_t>(tmp19));
  auto constant94 = builder.create<mix::ConstantOp>(loc, dense19);

  // line 25: torch.aten.slice.Tensor
  auto slice25 =
      builder.create<mix::SliceOp>(loc, constant94, 0, 0, INT64_MAX, 1);

  // line 28: torch.aten.unsqueeze
  auto unsqueeze28 = builder.create<mix::UnsqueezeOp>(loc, slice25, attr_i32_1);

  // line 31: torch.aten.unsqueeze
  auto unsqueeze31 =
      builder.create<mix::UnsqueezeOp>(loc, constant94, attr_i32_0);

  // line 37: torch.aten.slice.Tensor
  auto slice37 =
      builder.create<mix::SliceOp>(loc, unsqueeze31, 1, 0, INT64_MAX, 1);

  // line 39: torch.aten.lt.Tensor
  auto lt39 = builder.create<mix::LtOp>(loc, unsqueeze28, slice37);

  auto dense1 = DenseElementsAttr::get(
      RankedTensorType::get({seq_len, max_seq_len - seq_len}, type_i1), {true});
  auto constant1 = builder.create<mix::ConstantOp>(loc, dense1);

  auto dense2 = DenseElementsAttr::get(
      RankedTensorType::get({max_seq_len - seq_len, seq_len}, type_i1), {true});
  auto constant2 = builder.create<mix::ConstantOp>(loc, dense2);

  auto dense3 = DenseElementsAttr::get(
      RankedTensorType::get({max_seq_len - seq_len, max_seq_len - seq_len},
                            type_i1),
      {true});
  auto constant3 = builder.create<mix::ConstantOp>(loc, dense3);

  //   SmallVector<Value> tmp1;
  auto cat1 = builder.create<mix::ConcatOp>(loc, ValueRange{lt39, constant1},
                                            attr_i64_1);

  SmallVector<Value> tmp2{constant2, constant3};
  auto cat2 = builder.create<mix::ConcatOp>(loc, tmp2, attr_i64_1);

  SmallVector<Value> tmp3{cat1, cat2};
  auto cat3 = builder.create<mix::ConcatOp>(loc, tmp3, attr_i64_0);

  auto unsqueeze81 = builder.create<mix::UnsqueezeOp>(loc, cat3, attr_i32_0);

  // auto unsqueeze84 =
  //     builder.create<mix::UnsqueezeOp>(loc, unsqueeze81, attr_i32_0);

  return unsqueeze81;
}

void generateCode(mlir::ModuleOp &theModule, mlir::OpBuilder &builder,
                  mlir::MLIRContext &context) {

  auto elementType = builder.getI1Type();
  auto attention_mask_type =
      RankedTensorType::get({1, max_seq_len, max_seq_len}, builder.getI1Type());

  // printMemrefI1
  builder.setInsertionPointToEnd(theModule.getBody());
  auto printInputType = UnrankedTensorType::get(elementType);
  auto printFunTy =
      builder.getFunctionType(TypeRange{printInputType}, TypeRange{});
  auto printMemRefFunc = builder.create<func::FuncOp>(
      theModule->getLoc(), "printMemrefI8", printFunTy);
  printMemRefFunc.setPrivate();

  // Mask
  auto functionTy = builder.getFunctionType({}, {attention_mask_type});
  auto graph0 =
      builder.create<func::FuncOp>(theModule->getLoc(), "Mask", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);
  auto loc = graph0->getLoc();
  auto output =
      genMask(context, builder, graph0->getLoc(), seq_len, printMemRefFunc);

  builder.create<func::ReturnOp>(graph0->getLoc(), ValueRange{output});

  // main
  builder.setInsertionPointToEnd(theModule.getBody());
  auto mainfunc = builder.create<func::FuncOp>(loc, "main",
                                               builder.getFunctionType({}, {}));
  mainfunc.setPrivate();
  auto mainbody = mainfunc.addEntryBlock();
  builder.setInsertionPointToEnd(mainbody);

  auto res = builder.create<func::CallOp>(loc, graph0, ValueRange{});

  auto castResult =
      builder.create<tensor::CastOp>(loc, printInputType, res->getResult(0));
  builder.create<func::CallOp>(loc, printMemRefFunc, ValueRange{castResult});
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
      "/lib/x86_64-linux-gnu/crtn.o",
      "/home/gaoshihao/project/Aipiler/examples/Mask/print.o"};

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

  // mix::utils::load_model("./attention_model.bin", theModule, builder,
  //                        builder.getF16Type());

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

  generateExecutable(*llvmModule.get(), "Mask");

  mutil::log(mutil::LogLevel::INFO, "End gen exe.");

  return 0;
}
