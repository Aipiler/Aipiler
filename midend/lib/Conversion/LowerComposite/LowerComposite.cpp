#include "math.h"
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
#include "mlir/IR/Attributes.h"
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

class SiLULoweringPattern : public OpRewritePattern<mix::SiLUOp> {
public:
  using OpRewritePattern<mix::SiLUOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::SiLUOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto inputTy = input.getType();
    auto elementTy = inputTy.getElementType();
    auto loc = op->getLoc();
    auto neg0 = rewriter.create<mix::NegOp>(loc, inputTy, input);
    auto exp0 = rewriter.create<mix::ExpOp>(loc, inputTy, neg0);
    Value c1;
    if (elementTy.isa<IntegerType>()) {
      auto c1_attr = rewriter.getIntegerAttr(elementTy, 1);
      c1 = rewriter.create<arith::ConstantOp>(loc, c1_attr);
    } else if (elementTy.isa<FloatType>()) {
      auto c1_attr = rewriter.getFloatAttr(elementTy, 1);
      c1 = rewriter.create<arith::ConstantOp>(loc, c1_attr);
    }
    auto add0 = rewriter.create<mix::AddOp>(loc, inputTy, c1, exp0);
    auto div0 = rewriter.create<mix::DivOp>(loc, inputTy, input, add0);
    rewriter.replaceOp(op, div0);
    return success();
  }
};

} // namespace

void populateLowerCompositeOpPatterns(RewritePatternSet &patterns) {
  patterns.add<SiLULoweringPattern>(patterns.getContext());
}

namespace {
class LowerCompositePass
    : public PassWrapper<LowerCompositePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerCompositePass)
  LowerCompositePass() = default;
  LowerCompositePass(const LowerCompositePass &) {}
  StringRef getArgument() const final { return "lower-mix-composite"; }
  StringRef getDescription() const final {
    return "Convert mix.comp ops to mix.prim ops.";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ml_program::MLProgramDialect, func::FuncDialect,
                    arith::ArithDialect, index::IndexDialect>();
  }
};
} // namespace

void LowerCompositePass::runOnOperation() {
  MLIRContext &context = this->getContext();
  ModuleOp module = this->getOperation();
  ConversionTarget target(context);
  target.addLegalDialect<arith::ArithDialect, ml_program::MLProgramDialect,
                         mix::MIXDialect>();
  target.addIllegalOp<mix::SiLUOp>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&context);
  populateLowerCompositeOpPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

void registerLowerCompositePass() { PassRegistration<LowerCompositePass>(); }

std::unique_ptr<Pass> createLowerCompositePass() {
  return std::make_unique<LowerCompositePass>();
}