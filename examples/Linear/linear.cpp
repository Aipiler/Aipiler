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
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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

#include <chrono>
#include <iomanip>
#include <iostream>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Passes/PassBuilder.h>
#include <optional>
#include <sys/types.h>

using namespace mlir;
std::unique_ptr<Pass> createLowerModulePass();
std::unique_ptr<Pass> createLowerCompositePass(bool dynamicLoadWeight = false);
std::unique_ptr<Pass> createLowerPrimaryToTosa();

void registerLowerModulePass();
void registerLowerCompositePass();
void registerLowerPrimaryToTosaPass();

void generateCode(mlir::ModuleOp &theModule, mlir::OpBuilder &builder) {
  auto loc = builder.getUnknownLoc();
  builder.setInsertionPointToEnd(theModule.getBody());
  // printMemrefF32
  auto elementType = builder.getF32Type();
  auto printInputType = UnrankedTensorType::get(elementType);
  auto printFunTy =
      builder.getFunctionType(TypeRange{printInputType}, TypeRange{});
  auto printfunc =
      builder.create<func::FuncOp>(loc, "printMemrefF32", printFunTy);
  printfunc.setPrivate();

  // Graph0
  auto inputType = RankedTensorType::get({2, 10}, builder.getF32Type());
  //   auto resultType
  auto functionTy = builder.getFunctionType({inputType}, {});
  auto graph0 = builder.create<func::FuncOp>(loc, "graph0", functionTy);
  graph0.setPrivate();
  auto body = graph0.addEntryBlock();
  builder.setInsertionPointToEnd(body);

  auto input = graph0.getArgument(0);
  auto linear0 = builder.create<mix::LinearOp>(loc, input, "linear", 10, 5,
                                               true, builder.getF32Type());

  auto returnType = linear0.getType();
  graph0.setFunctionType(builder.getFunctionType(inputType, returnType));
  builder.create<func::ReturnOp>(loc, ValueRange{linear0});

  // Main
  builder.setInsertionPointToEnd(theModule.getBody());
  auto mainfunc = builder.create<func::FuncOp>(loc, "call_graph0",
                                               builder.getFunctionType({}, {}));
  mainfunc.setPrivate();
  auto mainbody = mainfunc.addEntryBlock();
  builder.setInsertionPointToEnd(mainbody);

  SmallVector<float> inputdata;
  for (int i = 0; i < 20; i++) {
    inputdata.push_back(float(i));
  }

  auto argAttr =
      DenseElementsAttr::get(inputType, llvm::ArrayRef<float>(inputdata));

  auto arg0 = builder.create<arith::ConstantOp>(loc, argAttr);
  auto call0 = builder.create<func::CallOp>(loc, graph0, ValueRange{arg0});
  auto cast =
      builder.create<tensor::CastOp>(loc, printInputType, call0->getResult(0));
  builder.create<func::CallOp>(loc, printfunc, ValueRange{cast});
  // print bias
  // auto biasType = MemRefType::get(ArrayRef<int64_t>{5},
  // builder.getF32Type()); auto bias = builder.create<memref::GetGlobalOp>(loc,
  // biasType, "linear_bias"); auto biasTensor =
  //     builder.create<bufferization::ToTensorOp>(loc, biasType, bias, true);
  // auto cast1 = builder.create<tensor::CastOp>(loc, printInputType,
  //                                             biasTensor->getResult(0));
  // builder.create<func::CallOp>(loc, printfunc, ValueRange{cast1});
  builder.create<func::ReturnOp>(loc);
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
  context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  mlir::OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  auto theModule = mlir::ModuleOp::create(loc);

  generateCode(theModule, builder);

  // load_model("./linear_model.bin", theModule, builder, builder.getF32Type());

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
  pm.addPass(createLowerCompositePass(true));
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

  mix::utils::CompileUtils util(theModule, "linear",
                                {mix::utils::CompileUtils::TARGET::MLIR,
                                 mix::utils::CompileUtils::TARGET::LLVMIR,
                                 mix::utils::CompileUtils::TARGET::OBJECT});
  util.compile();
  return 0;
}