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
#include "llvm/Support/MemoryBuffer.h"
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
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <memory>
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

mlir::Value MLP(mlir::OpBuilder &builder, mlir::Location loc,
                mlir::Value hidden_states, mlir::Value residual,
                int hidden_size = 16) {
  auto elementType = builder.getF32Type();
  auto linear0 =
      builder.create<mix::LinearOp>(loc, hidden_states, "gate_proj",
                                    hidden_size, 10000000, false, elementType);

  auto silu0 = builder.create<mix::SiLUOp>(loc, linear0);

  auto linear1 = builder.create<mix::LinearOp>(
      loc, hidden_states, "up_proj", hidden_size, 10000000, false, elementType);
  auto mul0 = builder.create<mix::MulOp>(loc, silu0, linear1);

  auto linear2 = builder.create<mix::LinearOp>(loc, mul0, "down_proj", 10000000,
                                               hidden_size, true, elementType);
  auto output = builder.create<mix::AddOp>(loc, linear2, residual);
  return output;
}

void generateCode(mlir::ModuleOp &theModule, mlir::OpBuilder &builder) {
  auto loc = theModule->getLoc();
  builder.setInsertionPointToEnd(theModule.getBody());

  auto elementType = builder.getF32Type();
  auto printInputType = UnrankedTensorType::get(elementType);
  auto printFunTy =
      builder.getFunctionType(TypeRange{printInputType}, TypeRange{});
  auto printfunc =
      builder.create<func::FuncOp>(loc, "printMemrefF32", printFunTy);
  printfunc.setPrivate();
  auto tensorType = RankedTensorType::get({4, 16}, elementType);

  auto functionTy =
      builder.getFunctionType({tensorType, tensorType}, {tensorType});
  auto graph0 = builder.create<func::FuncOp>(loc, "graph0", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto hidden_states = graph0.getArgument(0);
  auto residual = graph0.getArgument(1);
  auto output = MLP(builder, loc, hidden_states, residual);
  graph0.setFunctionType(
      builder.getFunctionType({tensorType, tensorType}, {output.getType()}));
  builder.create<func::ReturnOp>(loc, ValueRange{output});

  builder.setInsertionPointToEnd(theModule.getBody());
  auto mainfunc = builder.create<func::FuncOp>(loc, "main",
                                               builder.getFunctionType({}, {}));
  mainfunc.setPrivate();
  auto mainbody = mainfunc.addEntryBlock();
  builder.setInsertionPointToEnd(mainbody);

  auto arg1Attr = DenseElementsAttr::get(
      tensorType,
      llvm::ArrayRef<float>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                            14, 15, 16, 16, 15, 14, 13, 12, 11, 10, 9,  8,  7,
                            6,  5,  4,  3,  2,  1,  1,  2,  3,  4,  5,  6,  7,
                            8,  9,  10, 11, 12, 13, 14, 15, 16, 16, 15, 14, 13,
                            12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1});

  auto arg2Attr = DenseElementsAttr::get(
      tensorType,
      llvm::ArrayRef<float>{16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,
                            3,  2,  1,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                            11, 12, 13, 14, 15, 16, 16, 15, 14, 13, 12, 11, 10,
                            9,  8,  7,  6,  5,  4,  3,  2,  1,  1,  2,  3,  4,
                            5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16});

  auto arg0 = builder.create<arith::ConstantOp>(loc, arg1Attr);
  auto arg1 = builder.create<arith::ConstantOp>(loc, arg2Attr);
  auto res = builder.create<func::CallOp>(loc, graph0, ValueRange{arg0, arg1});
  auto cast =
      builder.create<tensor::CastOp>(loc, printInputType, res->getResult(0));
  builder.create<func::CallOp>(loc, printfunc, ValueRange{cast});

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
  llvm::TargetMachine *targetMachine = target->createTargetMachine(
      targetTriple, "generic", "", opt, llvm::Reloc::PIC_);

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
      "/root/Aipiler/thirdparty/llvm/build/bin/ld.lld",
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
      "-L../../thirdparty/llvm/build/lib",
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
  context.enableMultithreading();

  registerLowerModulePass();
  registerLowerCompositePass();
  registerLowerPrimaryToTosaPass();

  context.getOrLoadDialect<mix::MIXDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);

  generateCode(theModule, builder);

  load_model("./mlp_model.bin", theModule, builder, builder.getF32Type());
  mutil::log(mutil::LogLevel::INFO, "Start pass.");
  mlir::PassManager pm(&context);

  // "builtin.module( \
// 	lower-mix-module, \
// 	lower-mix-composite, \
// 	lower-mix-primary-to-tosa, \
// 	func.func( \
// 		tosa-to-linalg-named, \
// 		tosa-to-linalg, \
// 		tosa-to-tensor \
// 	), \
// 	empty-tensor-to-alloc-tensor, \
// 	one-shot-bufferize{bufferize-function-boundaries}, \
// 	convert-linalg-to-loops, \
// 	buffer-deallocation-pipeline, \
// 	expand-strided-metadata, \
// 	lower-affine, \
// 	finalize-memref-to-llvm, \
// 	convert-math-to-llvm, \
// 	convert-scf-to-cf, \
// 	convert-cf-to-llvm, \
// 	convert-func-to-llvm, \
// 	reconcile-unrealized-casts)"

  pm.addPass(createLowerModulePass());
  pm.addPass(createLowerCompositePass());
  pm.addPass(createLowerPrimaryToTosa());
  pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToLinalgNamed());
  pm.addNestedPass<func::FuncOp>(mlir::tosa::createTosaToLinalg());
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
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertMathToLLVMPass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(pm.run(theModule))) {
    return 4;
  }

  mutil::log(mutil::LogLevel::INFO, "End pass.");

  // translate to llvm ir
  mutil::log(mutil::LogLevel::INFO, "Start create llvm module.");
  llvm::LLVMContext LLVMContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(theModule, LLVMContext);
  mutil::log(mutil::LogLevel::INFO, "End create llvm module.");
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return -1;
  }
  mutil::log(mutil::LogLevel::INFO, "Start create exe.");
  generateExecutable(*llvmModule.get(), "mlp");
  mutil::log(mutil::LogLevel::INFO, "End create exe.");
  return 0;
}