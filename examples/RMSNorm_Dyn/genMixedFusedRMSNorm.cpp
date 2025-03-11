#include "Utils/loadPytorchModel.h"
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

#include "Utils/compileUtils.h"
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
#include <optional>
#include <sys/types.h>

using namespace mlir;
std::unique_ptr<Pass> createLowerModulePass();
std::unique_ptr<Pass> createLowerCompositePass(bool);
std::unique_ptr<Pass> createLowerPrimaryToTosa();

void registerLowerModulePass();
void registerLowerCompositePass();
void registerLowerPrimaryToTosaPass();

// 动态加载参数
bool DynamicLoadWeight = true;

// 模型参数定义
const int max_seq_len = 5;
const int hidden_size = 5120;
const int ffn_hidden_size = 12288;
const int n_head = 32;
const int head_dim = hidden_size / n_head;
const int key_value_projection_size = hidden_size * 2;
const int key_value_projection_head_dim = key_value_projection_size / n_head;
const int vocab_size = 120000;
const int batch_size = 1;
const int n_layer = 2; //  38

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

  llvm::SmallVector<mlir::Attribute> tmp{mlir::FloatAttr::get(elementType, 2)};
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

void generateCode(mlir::ModuleOp &theModule, mlir::OpBuilder &builder) {
  auto loc = theModule->getLoc();
  builder.setInsertionPointToEnd(theModule.getBody());

  auto elementType = builder.getF16Type();
  // auto printInputType = UnrankedTensorType::get(elementType);
  // auto printFunTy =
  //     builder.getFunctionType(TypeRange{printInputType}, TypeRange{});
  // auto printfunc =
  //     builder.create<func::FuncOp>(loc, "printMemrefF32", printFunTy);
  // printfunc.setPrivate();
  auto tensorType =
      RankedTensorType::get({max_seq_len, hidden_size}, elementType);

  // graph0
  auto functionTy = builder.getFunctionType({tensorType}, {});
  auto graph0 = builder.create<func::FuncOp>(loc, "graph0", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto hidden_states = graph0.getArgument(0);
  auto res = genFusedRMSNorm(builder, loc, hidden_states, "rms.weight");
  graph0.setFunctionType(
      builder.getFunctionType({tensorType}, {res.getType()}));

  builder.create<func::ReturnOp>(loc, ValueRange{res});
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

  context.getOrLoadDialect<mix::MIXDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();

  mlir::OpBuilder builder(&context);

  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);
  generateCode(theModule, builder);

  if (!DynamicLoadWeight)
    mix::utils::load_model(std::vector<std::string>{"./rms_model.bin"},
                           theModule, builder, builder.getF16Type());
  mlir::PassManager pm(&context);
  pm.addPass(createLowerModulePass());
  pm.addPass(createLowerCompositePass(DynamicLoadWeight));
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
    return 4;
  }

  mix::utils::CompileUtils util(
      theModule, "rms_norm",
      std::set{mix::utils::CompileUtils::TARGET::MLIR,
               mix::utils::CompileUtils::TARGET::LLVMIR,
               mix::utils::CompileUtils::TARGET::OBJECT});
  util.compile();
  return 0;
}