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

class MLPLoweringPattern : public OpRewritePattern<mix::MLPOp> {
public:
  using OpRewritePattern<mix::MLPOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::MLPOp op,
                                PatternRewriter &rewriter) const override {
    auto hidden_states = op.getHiddenStates();
    auto residual = op.getResidual();

    auto tensorType = hidden_states.getType();

    auto loc = op->getLoc();
    auto context = op->getContext();

    auto module = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<ml_program::GlobalOp>(loc, "weight0", tensorType, true,
                                          nullptr,
                                          rewriter.getStringAttr("public"));
    rewriter.create<ml_program::GlobalOp>(loc, "weight1", tensorType, true,
                                          nullptr,
                                          rewriter.getStringAttr("public"));
    rewriter.create<ml_program::GlobalOp>(loc, "weight2", tensorType, true,
                                          nullptr,
                                          rewriter.getStringAttr("public"));
    rewriter.create<ml_program::GlobalOp>(loc, "bias2", tensorType, true,
                                          nullptr,
                                          rewriter.getStringAttr("public"));
    rewriter.setInsertionPoint(op);
    auto _weight0 = rewriter.create<ml_program::GlobalLoadOp>(
        loc, tensorType, SymbolRefAttr::get(context, "weight0"));
    auto linear0 = rewriter.create<mix::LinearOp>(
        loc, tensorType, hidden_states, _weight0, nullptr);
    auto silu0 = rewriter.create<mix::SiLUOp>(loc, tensorType, linear0);
    auto _weight1 = rewriter.create<ml_program::GlobalLoadOp>(
        loc, tensorType, SymbolRefAttr::get(context, "weight1"));
    auto linear1 = rewriter.create<mix::LinearOp>(
        loc, tensorType, hidden_states, _weight1, nullptr);
    auto mul0 = rewriter.create<mix::MulOp>(loc, tensorType, silu0, linear1);
    auto _weight2 = rewriter.create<ml_program::GlobalLoadOp>(
        loc, tensorType, SymbolRefAttr::get(context, "weight2"));
    auto _bias2 = rewriter.create<ml_program::GlobalLoadOp>(
        loc, tensorType, SymbolRefAttr::get(context, "bias2"));
    auto linear2 =
        rewriter.create<mix::LinearOp>(loc, tensorType, mul0, _weight2, _bias2);
    auto output =
        rewriter.create<mix::AddOp>(loc, tensorType, linear2, residual);
    rewriter.replaceOp(op, output);
    return success();
  }
};

} // namespace

void populateConvertGraphPatterns(RewritePatternSet &patterns) {
  patterns.add<MLPLoweringPattern>(patterns.getContext());
}

namespace {
class LowerGraphPass
    : public PassWrapper<LowerGraphPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerGraphPass)
  LowerGraphPass() = default;
  LowerGraphPass(const LowerGraphPass &) {}
  StringRef getArgument() const final { return "lower-graph"; }
  StringRef getDescription() const final { return "Lower Graph Ops."; }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ml_program::MLProgramDialect, func::FuncDialect,
                    arith::ArithDialect, index::IndexDialect>();
  }
};
} // namespace

void LowerGraphPass::runOnOperation() {
  MLIRContext &context = this->getContext();
  ModuleOp module = this->getOperation();
  ConversionTarget target(context);
  target.addLegalDialect<arith::ArithDialect, ml_program::MLProgramDialect>();
  target.addLegalOp<ModuleOp, mix::LinearOp, mix::SiLUOp, mix::AddOp,
                    mix::SubOp, mix::MulOp, mix::DivOp>();
  RewritePatternSet patterns(&context);
  populateConvertGraphPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

void registerLowerGraphPass() { PassRegistration<LowerGraphPass>(); }

std::unique_ptr<Pass> createLowerGraphPass() {
  return std::make_unique<LowerGraphPass>();
}