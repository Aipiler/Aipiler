#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"

#include "mix/mixDialect.h"
#include "mix/mixOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Rewrite Pattern
//===----------------------------------------------------------------------===//

// Todo: Add more from mix Dialect to Tosa Dialect patterns here.

// populate lower mix conversion pass.
void populateLowerMixToTosaPatterns(RewritePatternSet &patterns) {
  // patterns.add<>(patterns.getContext());
}

namespace {
class LowerMixToTosaPass
    : public PassWrapper<LowerMixToTosaPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerMixToTosaPass)
  StringRef getArgument() const final { return "convert-mix-to-tosa"; }
  StringRef getDescription() const final {
    return "convert mix low level ops to tosa.";
  }
  LowerMixToTosaPass() = default;
  LowerMixToTosaPass(const LowerMixToTosaPass &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, LLVM::LLVMDialect, tosa::TosaDialect>();
  }

  void runOnOperation() override;
};
} // namespace

void LowerMixToTosaPass::runOnOperation() {
  MLIRContext &context = this->getContext();
  ModuleOp module = this->getOperation();
  ConversionTarget target(context);
  target.addLegalDialect<arith::ArithDialect, tosa::TosaDialect>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&context);
  populateLowerMixToTosaPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

void registerLowerMixToTosaPass() { PassRegistration<LowerMixToTosaPass>(); }

std::unique_ptr<Pass> createLowerMixToTosaPass() {
  return std::make_unique<LowerMixToTosaPass>();
}