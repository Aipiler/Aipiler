#include "mix/mixDialect.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

void registerLowerModulePass();
void registerLowerCompositePass();
void registerLowerPrimaryToTosaPass();
void registerBatchMatMulOptimizePass();
void registerConv2dNhwcFhwcOptimizePass();
void registerPoolingNhwcMaxOptimizePass();

int main(int argc, char **argv) {
  // Register all MLIR passes.
  mlir::registerAllPasses();

  registerLowerModulePass();
  registerLowerCompositePass();
  registerLowerPrimaryToTosaPass();
  registerBatchMatMulOptimizePass();
  registerConv2dNhwcFhwcOptimizePass();
  registerPoolingNhwcMaxOptimizePass();

  mlir::DialectRegistry registry;
  // Register all MLIR core dialects.
  registerAllDialects(registry);
  mlir::registerAllExtensions(registry);

  registry.insert<mix::MIXDialect>();

  // clang-format on

  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "mix-mlir optimizer driver", registry));
}
