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
#include <cstdint>
#include <iostream>
#include <memory>

#include "mix/mixDialect.h"
#include "mix/mixOps.h"
#include "llvm/ADT/ArrayRef.h"

#define USE_MLP

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
#else
    auto tensorShape = tensorType.getShape();
    auto elementType = tensorType.getElementType();
    auto memrefType = MemRefType::get(tensorShape, elementType);
    rewriter.create<memref::GlobalOp>(loc, rewriter.getStringAttr("weight0"),
                                      rewriter.getStringAttr("public"),
                                      TypeAttr::get(memrefType), Attribute{},
                                      UnitAttr{}, IntegerAttr{});
    rewriter.create<memref::GlobalOp>(loc, rewriter.getStringAttr("weight1"),
                                      rewriter.getStringAttr("public"),
                                      TypeAttr::get(memrefType), Attribute{},
                                      UnitAttr{}, IntegerAttr{});
    rewriter.create<memref::GlobalOp>(loc, rewriter.getStringAttr("weight2"),
                                      rewriter.getStringAttr("public"),
                                      TypeAttr::get(memrefType), Attribute{},
                                      UnitAttr{}, IntegerAttr{});
    rewriter.create<memref::GlobalOp>(
        loc, rewriter.getStringAttr("bias2"), rewriter.getStringAttr("public"),
        TypeAttr::get(memrefType), Attribute{}, UnitAttr{}, IntegerAttr{});
#endif
    rewriter.setInsertionPoint(op);
#ifdef USE_MLP
    auto _weight0 = rewriter.create<ml_program::GlobalLoadConstOp>(
        loc, tensorType, SymbolRefAttr::get(context, "weight0"));
    auto _weight1 = rewriter.create<ml_program::GlobalLoadConstOp>(
        loc, tensorType, SymbolRefAttr::get(context, "weight1"));
    auto _weight2 = rewriter.create<ml_program::GlobalLoadConstOp>(
        loc, tensorType, SymbolRefAttr::get(context, "weight2"));
    auto _bias2 = rewriter.create<ml_program::GlobalLoadConstOp>(
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

class RMSNormLoweringPattern : public OpRewritePattern<mix::RMSNormOp> {
public:
  using OpRewritePattern<mix::RMSNormOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::RMSNormOp op,
                                PatternRewriter &rewriter) const override {
    auto hidden_states = op.getHiddenStates();
    auto hidden_size = op.getHiddenSize();
    auto eps = op.getEps();
    auto hidden_states_type = hidden_states.getType();
    auto hidden_states_shape = hidden_states_type.getShape();
    auto hidden_states_rank = hidden_states_shape.size();
    auto elementType = hidden_states_type.getElementType();
    llvm::ArrayRef<int64_t> weightShape{int64_t(hidden_size)};
    auto weightTensorType = RankedTensorType::get(weightShape, elementType);

    auto loc = op.getLoc();
    auto context = op->getContext();

    auto module = op->getParentOfType<ModuleOp>();
    rewriter.setInsertionPointToStart(module.getBody());
#ifdef USE_MLP
    rewriter.create<ml_program::GlobalOp>(loc, "weight3", weightTensorType,
                                          true, nullptr,
                                          rewriter.getStringAttr("public"));
#else
    auto memrefType = MemRefType::get(weightShape, elementType);
    rewriter.create<memref::GlobalOp>(loc, rewriter.getStringAttr("weight3"),
                                      rewriter.getStringAttr("public"),
                                      TypeAttr::get(memrefType), Attribute{},
                                      UnitAttr{}, IntegerAttr{});
#endif
    rewriter.setInsertionPoint(op);

#ifdef USE_MLP
    auto _weight3 = rewriter.create<ml_program::GlobalLoadOp>(
        loc, weightTensorType, SymbolRefAttr::get(context, "weight3"));
#else
    auto _weight3Memref =
        rewriter.create<memref::GetGlobalOp>(loc, memrefType, "weight3");
    auto _weight3 = rewriter.create<bufferization::ToTensorOp>(
        loc, weightTensorType, _weight3Memref, UnitAttr{});
#endif
    auto constantTensorType = RankedTensorType::get({1}, elementType);
    auto constantTensor = DenseElementsAttr::get(constantTensorType, {2.0f});
    auto c2Tensor = rewriter.create<arith::ConstantOp>(loc, constantTensor);
    auto pow0 = rewriter.create<mix::PowOp>(loc, hidden_states, c2Tensor);
    auto mean0 = rewriter.create<mix::MeanOp>(
        loc, pow0, rewriter.getI32ArrayAttr({int32_t(hidden_states_rank - 1)}),
        rewriter.getBoolAttr(true));
    // TODO: eps is F64Attr, but element may not be.
    auto epsAttr = rewriter.getFloatAttr(elementType, eps.convertToDouble());
    auto const_eps = rewriter.create<arith::ConstantOp>(loc, epsAttr);
    auto add0 = rewriter.create<mix::AddOp>(loc, mean0, const_eps);
    auto rsqrt0 = rewriter.create<mix::RsqrtOp>(loc, add0);
    auto mul0 = rewriter.create<mix::MulOp>(loc, hidden_states, rsqrt0);
    auto mul1 = rewriter.create<mix::MulOp>(loc, _weight3, mul0);
    rewriter.replaceOp(op, mul1);
    return success();
  }
};

class SelfAttentionLoweringPattern
    : public OpRewritePattern<mix::SelfAttentionOp> {
public:
  using OpRewritePattern<mix::SelfAttentionOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::SelfAttentionOp op,
                                PatternRewriter &rewriter) const override {
    auto hiddenStates = op.getHiddenStates();
    auto residual = op.getResidual();
    auto attentionMask = op.getAttentionMask();
    auto returnType = hiddenStates.getType();
    auto loc = op->getLoc();
    MLIRContext *context = op->getContext(); // 获取 MLIRContext

    // transpose
    auto attr1 =
        IntegerAttr::get(IntegerType::get(context, 32), llvm::APInt(32, 1));
    auto attr0 =
        IntegerAttr::get(IntegerType::get(context, 32), llvm::APInt(32, 0));
    llvm::SmallVector<Attribute> attrs;
    attrs.push_back(attr1);
    attrs.push_back(attr0);
    ArrayAttr arrayAttr = ArrayAttr::get(context, attrs);
    // TODO: 缺少返回值shape信息。
    // auto transpose =
    //     rewriter.create<mix::PermuteOp>(loc, hiddenStates, arrayAttr);
    // rewriter.replaceOp(op, transpose);
    return success();
  }
};
} // namespace

void populateLowerModulePatterns(RewritePatternSet &patterns) {
  patterns
      .add<MLPLoweringPattern, LinearLoweringPattern, RMSNormLoweringPattern>(
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
  target.addIllegalOp<mix::MLPOp, mix::LinearOp, mix::RMSNormOp>(); //
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