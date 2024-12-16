#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <iostream>
#include <memory>

#include "mix/mixDialect.h"
#include "mix/mixOps.h"

using namespace mlir;

namespace {

class AddLoweringPattern : public OpRewritePattern<mix::AddOp> {
public:
  using OpRewritePattern<mix::AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::AddOp op,
                                PatternRewriter &rewriter) const override {
    return success();
  }
};

} // namespace

void populateLowerPrimaryToTosaPatterns(RewritePatternSet &patterns) {
  //   patterns.add<AddLoweringPattern>(patterns.getContext());
}

namespace {
class LowerPrimaryToTosa
    : public PassWrapper<LowerPrimaryToTosa, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPrimaryToTosa)
  LowerPrimaryToTosa() = default;
  LowerPrimaryToTosa(const LowerPrimaryToTosa &) {}
  StringRef getArgument() const final { return "lower-mix-primary"; }
  StringRef getDescription() const final {
    return "Convert mix.prim ops to tosa.";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ml_program::MLProgramDialect, func::FuncDialect,
                    arith::ArithDialect, index::IndexDialect>();
  }
};
} // namespace

void LowerPrimaryToTosa::runOnOperation() {
  MLIRContext &context = this->getContext();
  ModuleOp module = this->getOperation();
  ConversionTarget target(context);
  target.addLegalDialect<arith::ArithDialect, ml_program::MLProgramDialect,
                         mix::MIXDialect>();
  target.addIllegalOp<mix::SiLUOp, mix::SoftmaxOp>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&context);
  populateLowerPrimaryToTosaPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

void registerLowerPrimaryToTosa() { PassRegistration<LowerPrimaryToTosa>(); }

std::unique_ptr<Pass> createLowerPrimaryToTosa() {
  return std::make_unique<LowerPrimaryToTosa>();
}