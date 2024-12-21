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
#include "mlir/IR/BuiltinAttributes.h"
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
#include <cstdint>
#include <iostream>
#include <memory>

#include "mix/mixDialect.h"
#include "mix/mixOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

class SiLULoweringPattern : public OpRewritePattern<mix::SiLUOp> {
public:
  using OpRewritePattern<mix::SiLUOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::SiLUOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto loc = op->getLoc();

    auto sigmoid0 = rewriter.create<mix::SigmoidOp>(loc, input);
    auto mul0 = rewriter.create<mix::MulOp>(loc, input, sigmoid0);
    rewriter.replaceOp(op, mul0);
    return success();
  }
};

class SigmoidLoweringPattern : public OpRewritePattern<mix::SigmoidOp> {
public:
  using OpRewritePattern<mix::SigmoidOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::SigmoidOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto inputTy = input.getType();
    auto elementTy = inputTy.getElementType();
    auto loc = op->getLoc();
    auto neg0 = rewriter.create<mix::NegOp>(loc, inputTy, input);
    auto exp0 = rewriter.create<mix::ExpOp>(loc, inputTy, neg0);

    auto tensorTy = RankedTensorType::get({1}, elementTy);
    auto c1_attr = DenseElementsAttr::get(tensorTy, {1.0f});
    auto c1 = rewriter.create<arith::ConstantOp>(loc, c1_attr);
    auto add0 = rewriter.create<mix::AddOp>(loc, c1, exp0);
    auto div0 = rewriter.create<mix::DivOp>(loc, c1, add0);
    rewriter.replaceOp(op, div0);
    return success();
  }
};

class MeanLoweringPattern : public OpRewritePattern<mix::MeanOp> {
public:
  using OpRewritePattern<mix::MeanOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::MeanOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto inputTy = input.getType();
    auto inputShape = inputTy.getShape();
    auto elementTy = inputTy.getElementType();
    auto dimsArrAttr = op.getDims();
    auto keepDim = op.getKeepDim();
    auto loc = op->getLoc();
    Value res = input;
    llvm::SmallVector<int64_t> resShape(inputShape);
    float scale = 1.0f;
    // TODO: how about dynamic shape?
    for (auto reduceDimAttr : dimsArrAttr) {
      auto axis = dyn_cast<IntegerAttr>(reduceDimAttr);
      auto axisNum = axis.getInt();
      res = rewriter.create<mix::ReduceSumOp>(loc, res, axis);
      scale *= resShape[axisNum];
      if (keepDim) {
        resShape[axisNum] = 1;
      } else {
        resShape.erase(resShape.begin() + axisNum);
      }
    }
    if (!keepDim) {
      auto resType = RankedTensorType::get(resShape, elementTy);
      res = rewriter.create<mix::ReshapeOp>(loc, resType, res,
                                            rewriter.getI64ArrayAttr(resShape));
    }
    auto scaleType = RankedTensorType::get({1}, elementTy);
    auto scaleAttr = DenseElementsAttr::get(scaleType, float(1.0 / scale));
    auto scaleValue = rewriter.create<arith::ConstantOp>(loc, scaleAttr);
    res = rewriter.create<mix::MulOp>(loc, res, scaleValue);
    rewriter.replaceOp(op, res);
    return success();
  }
};

} // namespace

void populateLowerCompositeOpPatterns(RewritePatternSet &patterns) {
  patterns
      .add<SiLULoweringPattern, SigmoidLoweringPattern, MeanLoweringPattern>(
          patterns.getContext());
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
  target.addIllegalOp<mix::SiLUOp, mix::SigmoidOp, mix::MeanOp>();
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