#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
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
#include <iostream>
#include <memory>

#include "mix/mixDialect.h"
#include "mix/mixOps.h"

// #define USE_MLP

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
#ifdef USE_MLP
    rewriter.create<ml_program::GlobalOp>(loc, "weight0", tensorType, true,
                                          nullptr,
                                          rewriter.getStringAttr("private"));
    rewriter.create<ml_program::GlobalOp>(loc, "weight1", tensorType, true,
                                          nullptr,
                                          rewriter.getStringAttr("private"));
    rewriter.create<ml_program::GlobalOp>(loc, "weight2", tensorType, true,
                                          nullptr,
                                          rewriter.getStringAttr("private"));
    rewriter.create<ml_program::GlobalOp>(loc, "bias2", tensorType, true,
                                          nullptr,
                                          rewriter.getStringAttr("private"));
#else
    auto tensorShape = tensorType.getShape();
    auto elementType = tensorType.getElementType();
    auto memrefType = MemRefType::get(tensorShape, elementType);
    rewriter.create<memref::GlobalOp>(loc, rewriter.getStringAttr("weight0"),
                                      rewriter.getStringAttr("private"),
                                      TypeAttr::get(memrefType), Attribute{},
                                      UnitAttr{}, IntegerAttr{});
    rewriter.create<memref::GlobalOp>(loc, rewriter.getStringAttr("weight1"),
                                      rewriter.getStringAttr("private"),
                                      TypeAttr::get(memrefType), Attribute{},
                                      UnitAttr{}, IntegerAttr{});
    rewriter.create<memref::GlobalOp>(loc, rewriter.getStringAttr("weight2"),
                                      rewriter.getStringAttr("private"),
                                      TypeAttr::get(memrefType), Attribute{},
                                      UnitAttr{}, IntegerAttr{});
    rewriter.create<memref::GlobalOp>(
        loc, rewriter.getStringAttr("bias2"), rewriter.getStringAttr("private"),
        TypeAttr::get(memrefType), Attribute{}, UnitAttr{}, IntegerAttr{});
#endif
    rewriter.setInsertionPoint(op);
#ifdef USE_MLP
    auto _weight0 = rewriter.create<ml_program::GlobalLoadOp>(
        loc, tensorType, SymbolRefAttr::get(context, "weight0"));
    auto _weight1 = rewriter.create<ml_program::GlobalLoadOp>(
        loc, tensorType, SymbolRefAttr::get(context, "weight1"));
    auto _weight2 = rewriter.create<ml_program::GlobalLoadOp>(
        loc, tensorType, SymbolRefAttr::get(context, "weight2"));
    auto _bias2 = rewriter.create<ml_program::GlobalLoadOp>(
        loc, tensorType, SymbolRefAttr::get(context, "bias2"));
#else
    auto _weight0Memref =
        rewriter.create<memref::GetGlobalOp>(loc, memrefType, "weight0");
    auto _weight0 = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, _weight0Memref, UnitAttr{});
    auto _weight1Memref =
        rewriter.create<memref::GetGlobalOp>(loc, memrefType, "weight1");
    auto _weight1 = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, _weight1Memref, UnitAttr{});
    auto _weight2Memref =
        rewriter.create<memref::GetGlobalOp>(loc, memrefType, "weight2");
    auto _weight2 = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, _weight2Memref, UnitAttr{});
    auto _bias2Memref =
        rewriter.create<memref::GetGlobalOp>(loc, memrefType, "bias2");
    auto _bias2 = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, _bias2Memref, UnitAttr{});
#endif
    auto linear0 = rewriter.create<mix::LinearOp>(
        loc, tensorType, hidden_states, _weight0, nullptr);
    auto silu0 = rewriter.create<mix::SiLUOp>(loc, linear0);

    auto linear1 = rewriter.create<mix::LinearOp>(
        loc, tensorType, hidden_states, _weight1, nullptr);
    auto mul0 = rewriter.create<mix::MulOp>(loc, silu0, linear1);

    auto linear2 =
        rewriter.create<mix::LinearOp>(loc, tensorType, mul0, _weight2, _bias2);
    auto output = rewriter.create<mix::AddOp>(loc, linear2, residual);
    rewriter.replaceOp(op, output);
    return success();
  }
};

class LinearLoweringPattern : public OpRewritePattern<mix::LinearOp> {
public:
  using OpRewritePattern<mix::LinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::LinearOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto weight = op.getWeight();
    auto bias = op.getBias();
    auto loc = op->getLoc();
    auto matmul0 = rewriter.create<mix::MatMulOp>(loc, input, weight);
    Value output = matmul0;
    if (bias) {
      output = rewriter.create<mix::AddOp>(loc, output, bias);
    }
    rewriter.replaceOp(op, output);
    return success();
  }
};

} // namespace

void populateLowerModulePatterns(RewritePatternSet &patterns) {
  patterns.add<MLPLoweringPattern, LinearLoweringPattern>(
      patterns.getContext());
}

namespace {
class LowerModulePass
    : public PassWrapper<LowerModulePass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerModulePass)
  LowerModulePass() = default;
  LowerModulePass(const LowerModulePass &) {}
  StringRef getArgument() const final { return "lower-mix-module"; }
  StringRef getDescription() const final {
    return "Convert mix.module ops to mix.comp op and mix.prim ops.";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<ml_program::MLProgramDialect, func::FuncDialect,
                arith::ArithDialect, index::IndexDialect, memref::MemRefDialect,
                bufferization::BufferizationDialect>();
  }
};
} // namespace

void LowerModulePass::runOnOperation() {
  MLIRContext &context = this->getContext();
  ModuleOp module = this->getOperation();
  ConversionTarget target(context);
  target.addLegalDialect<arith::ArithDialect, ml_program::MLProgramDialect,
                         mix::MIXDialect, memref::MemRefDialect,
                         bufferization::BufferizationDialect>();
  target.addIllegalOp<mix::MLPOp, mix::LinearOp>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&context);
  populateLowerModulePatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

void registerLowerModulePass() { PassRegistration<LowerModulePass>(); }

std::unique_ptr<Pass> createLowerModulePass() {
  return std::make_unique<LowerModulePass>();
}