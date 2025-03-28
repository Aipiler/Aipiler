#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MLProgram/IR/MLProgramAttributes.h"
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
class LinearLoweringPattern : public OpRewritePattern<mix::LinearOp> {
public:
  using OpRewritePattern<mix::LinearOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::LinearOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto inputType = input.getType();
    auto inputShape = inputType.cast<RankedTensorType>().getShape();
    auto loc = op.getLoc();
    auto in_feature = op.getInFeature();
    auto out_feature = op.getOutFeature();
    auto dtypeAttr = op.getDtype();
    auto has_bias = op.getHasBias();
    Type dtype =
        dtypeAttr.has_value() ? dtypeAttr.value() : rewriter.getF32Type();
    std::string params_loc(op.getParamsLoc());
    auto weight_loc = params_loc + ".weight";

    SmallVector<int64_t> weightShape{out_feature, in_feature};
    auto weightType = RankedTensorType::get(weightShape, dtype);
    auto weight = rewriter.create<mix::WeightOp>(loc, weightType, weight_loc);
    // TODO: load bin needs transpose
    auto transpose0 = rewriter.create<mix::TransposeOp>(loc, weight, 0, 1);
    Value matmul0;
    if (inputShape.size() == 2) {
      matmul0 = rewriter.create<mix::MatMulOp>(loc, input, transpose0);
    } else if (inputShape.size() == 3) {
      auto weight_reshape = rewriter.create<mix::ReshapeOp>(
          loc, transpose0,
          rewriter.getI64ArrayAttr({1, in_feature, out_feature}));
      matmul0 = rewriter.create<mix::BatchMatMulOp>(loc, input, weight_reshape);
    } else {
      return op.emitOpError("Unexpect input shape");
    }

    Value output = matmul0;
    if (has_bias) {
      auto bias_loc = params_loc + ".bias";
      SmallVector<int64_t> biasShape{out_feature};
      auto biasType = RankedTensorType::get(biasShape, dtype);
      auto bias = rewriter.create<mix::WeightOp>(loc, biasType, bias_loc);
      output = rewriter.create<mix::AddOp>(loc, output, bias);
    }
    rewriter.replaceOp(op, output);
    return success();
  }
};

class EmbeddingLoweringPattern : public OpRewritePattern<mix::EmbeddingOp> {
public:
  using OpRewritePattern<mix::EmbeddingOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::EmbeddingOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto inputType = input.getType();
    auto loc = op.getLoc();
    auto num_embeddings = op.getNumEmbeddings();
    auto embedding_dim = op.getEmbeddingDim();
    auto opt_pad_idx = op.getPaddingIdx();
    auto opt_max_norm = op.getMaxNorm();
    auto dtypeAttr = op.getDtype();
    Type dtype =
        dtypeAttr.has_value() ? dtypeAttr.value() : rewriter.getF32Type();
    std::string params_loc(op.getParamsLoc());
    auto weight_loc = params_loc + ".weight";
    // load weight
    auto weightType = RankedTensorType::get(
        ArrayRef<int64_t>{num_embeddings, embedding_dim}, dtype);
    Value _weight = rewriter.create<mix::WeightOp>(loc, weightType, weight_loc);
    Value weight = rewriter.create<mix::ReshapeOp>(
        loc, _weight,
        rewriter.getI64ArrayAttr(
            ArrayRef<int64_t>{1, num_embeddings, embedding_dim}));
    if (opt_pad_idx.has_value()) {
      auto pad_idx = opt_pad_idx.value();
      std::vector<float> mask_data(num_embeddings, 1.0f);
      mask_data[pad_idx] = 0.0f;
      SmallVector<int64_t> mask_shape{num_embeddings, 1};
      auto mask_type = RankedTensorType::get(mask_shape, rewriter.getF32Type());
      auto mask = rewriter.create<mix::ConstantOp>(
          loc, DenseElementsAttr::get(mask_type, ArrayRef<float>(mask_data)));
      weight = rewriter.create<mix::MulOp>(loc, weight, mask);
    }
    // TODO: process max_norm
    if (opt_max_norm.has_value()) {
      auto max_norm = opt_max_norm.value().convertToFloat();
      return op.emitOpError("max_norm is unsupport now");
    }

    auto inputShape = input.getType().getShape();
    int64_t inputCount = 1;
    for (auto dimSize : inputShape) {
      inputCount *= dimSize;
    }
    auto reshape0 = rewriter.create<mix::ReshapeOp>(
        loc, input, rewriter.getI64ArrayAttr({1, inputCount}));
    auto inputElementType = inputType.getElementType();
    auto integerElementTy = dyn_cast<IntegerType>(inputElementType);
    Value gatherIndices = reshape0;
    if (!integerElementTy || inputElementType.getIntOrFloatBitWidth() != 32) {
      auto newType = RankedTensorType::get(inputShape, rewriter.getI32Type());
      gatherIndices = rewriter.create<mix::ConvertOp>(loc, reshape0, newType);
    }
    auto gather0 = rewriter.create<mix::GatherOp>(loc, weight, gatherIndices);
    auto output = rewriter.create<mix::ReshapeOp>(
        loc, gather0, rewriter.getI64ArrayAttr(op.getType().getShape()));
    rewriter.replaceOp(op, output);
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
  patterns.add<LinearLoweringPattern, EmbeddingLoweringPattern>(
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
  target.addIllegalOp<mix::LinearOp, mix::EmbeddingOp>(); //
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