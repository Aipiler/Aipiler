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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"
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

#include "Utils/readjson.h"
#include "mix/mixDialect.h"
#include "mix/mixOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace {

class AddLoweringPattern : public OpRewritePattern<mix::AddOp> {
public:
  using OpRewritePattern<mix::AddOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto loc = op->getLoc();
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();
    auto resultType = op.getType();
    auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
    Value newop;
    if (!resultTensorType) {
      if (auto resIntType = dyn_cast<IntegerType>(resultType)) {
        newop = rewriter.create<arith::AddIOp>(loc, lhs, rhs);
      } else if (auto resFloatType = dyn_cast<FloatType>(resultType)) {
        newop = rewriter.create<arith::AddFOp>(loc, lhs, rhs);
      } else {
        return op.emitOpError() << "Unexpected types.";
      }
    } else {
      auto lhsTensorType = dyn_cast<RankedTensorType>(lhsType);
      auto rhsTensorType = dyn_cast<RankedTensorType>(rhsType);
      auto elemTy = resultTensorType.getElementType();
      auto tensorTy = RankedTensorType::get({1}, elemTy);
      if (!lhsTensorType) {
        lhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, lhs);
      }
      if (!rhsTensorType) {
        rhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, rhs);
      }
      newop = rewriter.create<tosa::AddOp>(loc, resultTensorType, lhs, rhs);
    }
    rewriter.replaceOp(op, newop);
    return success();
  }
};

class SubLoweringPattern : public OpRewritePattern<mix::SubOp> {
public:
  using OpRewritePattern<mix::SubOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::SubOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto loc = op->getLoc();
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();
    auto resultType = op.getType();
    auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
    Value newop;
    if (!resultTensorType) {
      if (auto resIntType = dyn_cast<IntegerType>(resultType)) {
        newop = rewriter.create<arith::SubIOp>(loc, lhs, rhs);
      } else if (auto resFloatType = dyn_cast<FloatType>(resultType)) {
        newop = rewriter.create<arith::SubFOp>(loc, lhs, rhs);
      } else {
        return op.emitOpError() << "Unexpected types.";
      }
    } else {
      auto lhsTensorType = dyn_cast<RankedTensorType>(lhsType);
      auto rhsTensorType = dyn_cast<RankedTensorType>(rhsType);
      auto elemTy = resultTensorType.getElementType();
      auto tensorTy = RankedTensorType::get({1}, elemTy);
      if (!lhsTensorType) {
        lhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, lhs);
      } else if (!rhsTensorType) {
        rhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, rhs);
      }
      newop = rewriter.create<tosa::SubOp>(loc, resultTensorType, lhs, rhs);
    }
    rewriter.replaceOp(op, newop);
    return success();
  }
};

class MulLoweringPattern : public OpRewritePattern<mix::MulOp> {
public:
  using OpRewritePattern<mix::MulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::MulOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto loc = op->getLoc();
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();
    auto resultType = op.getType();
    auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
    Value newop;
    if (!resultTensorType) {
      if (auto resIntType = dyn_cast<IntegerType>(resultType)) {
        newop = rewriter.create<arith::MulIOp>(loc, lhs, rhs);
      } else if (auto resFloatType = dyn_cast<FloatType>(resultType)) {
        newop = rewriter.create<arith::MulFOp>(loc, lhs, rhs);
      } else {
        return op.emitOpError() << "Unexpected types.";
      }
    } else {
      auto lhsTensorType = dyn_cast<RankedTensorType>(lhsType);
      auto rhsTensorType = dyn_cast<RankedTensorType>(rhsType);
      auto elemTy = resultTensorType.getElementType();
      auto tensorTy = RankedTensorType::get({1}, elemTy);
      if (!lhsTensorType) {
        lhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, lhs);
      } else if (!rhsTensorType) {
        rhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, rhs);
      }
      newop = rewriter.create<tosa::MulOp>(loc, resultTensorType, lhs, rhs, 0);
    }
    rewriter.replaceOp(op, newop);
    return success();
  }
};

class DivLoweringPattern : public OpRewritePattern<mix::DivOp> {
public:
  using OpRewritePattern<mix::DivOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::DivOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto loc = op->getLoc();
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();
    auto resultType = op.getType();
    auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
    Value newop;
    if (!resultTensorType) {
      if (auto resIntType = dyn_cast<IntegerType>(resultType)) {
        newop = rewriter.create<arith::DivSIOp>(loc, lhs, rhs);
      } else if (auto resFloatType = dyn_cast<FloatType>(resultType)) {
        newop = rewriter.create<arith::DivFOp>(loc, lhs, rhs);
      } else {
        return op.emitOpError() << "Unexpected types.";
      }
    } else {
      auto lhsTensorType = dyn_cast<RankedTensorType>(lhsType);
      auto rhsTensorType = dyn_cast<RankedTensorType>(rhsType);
      auto elemTy = resultTensorType.getElementType();
      auto tensorTy = RankedTensorType::get({1}, elemTy);
      if (!lhsTensorType) {
        lhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, lhs);
      } else if (!rhsTensorType) {
        rhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, rhs);
      }
      auto recip0 =
          rewriter.create<tosa::ReciprocalOp>(loc, rhs.getType(), rhs);
      newop =
          rewriter.create<tosa::MulOp>(loc, resultTensorType, lhs, recip0, 0);
    }
    rewriter.replaceOp(op, newop);
    return success();
  }
};

class MatmulLoweringPattern : public OpRewritePattern<mix::MatMulOp> {
public:
  using OpRewritePattern<mix::MatMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::MatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto resType = op.getType();
    auto loc = op.getLoc();
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
    auto resShape = resType.getShape();
    SmallVector<int64_t> newLhsShape{1, lhsShape[0], lhsShape[1]};
    SmallVector<int64_t> newRhsShape{1, rhsShape[0], rhsShape[1]};
    SmallVector<int64_t> newResShape{1, resShape[0], resShape[1]};
    auto newResType =
        RankedTensorType::get(newResShape, resType.getElementType());
    auto newLhs = rewriter.create<tosa::ReshapeOp>(
        loc, lhs, rewriter.getDenseI64ArrayAttr(newLhsShape));
    auto newRhs = rewriter.create<tosa::ReshapeOp>(
        loc, rhs, rewriter.getDenseI64ArrayAttr(newRhsShape));

    auto matmul0 =
        rewriter.create<tosa::MatMulOp>(loc, newResType, newLhs, newRhs);
    auto res = rewriter.create<tosa::ReshapeOp>(
        loc, matmul0, rewriter.getDenseI64ArrayAttr(resShape));
    rewriter.replaceOp(op, res);
    return success();
  }
};

class NegLoweringPattern : public OpRewritePattern<mix::NegOp> {
public:
  using OpRewritePattern<mix::NegOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::NegOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto resultType = op.getType();
    auto loc = op->getLoc();
    auto neg0 = rewriter.create<tosa::NegateOp>(loc, resultType, input);
    rewriter.replaceOp(op, neg0);
    return success();
  }
};

class ExpLoweringPattern : public OpRewritePattern<mix::ExpOp> {
public:
  using OpRewritePattern<mix::ExpOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::ExpOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto resultType = op.getType();
    auto loc = op->getLoc();
    auto exp0 = rewriter.create<tosa::ExpOp>(loc, resultType, input);
    rewriter.replaceOp(op, exp0);
    return success();
  }
};

class PowLoweringPattern : public OpRewritePattern<mix::PowOp> {
public:
  using OpRewritePattern<mix::PowOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::PowOp op,
                                PatternRewriter &rewriter) const override {

    auto lhs = op.getInput();
    auto rhs = op.getExponent();
    auto loc = op->getLoc();
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();
    auto resultType = op.getType();
    auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
    Value newop;
    if (!resultTensorType) {
      // TODO
      return op.emitOpError() << "Not support scale pow now.";
    } else {
      auto lhsTensorType = dyn_cast<RankedTensorType>(lhsType);
      auto rhsTensorType = dyn_cast<RankedTensorType>(rhsType);
      auto elemTy = resultTensorType.getElementType();
      auto tensorTy = RankedTensorType::get({1}, elemTy);
      if (!lhsTensorType) {
        lhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, lhs);
      } else if (!rhsTensorType) {
        rhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, rhs);
      }
      newop = rewriter.create<tosa::PowOp>(loc, resultTensorType, lhs, rhs);
    }
    rewriter.replaceOp(op, newop);
    return success();
  }
};

class ReduceSumLoweringPattern : public OpRewritePattern<mix::ReduceSumOp> {
public:
  using OpRewritePattern<mix::ReduceSumOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::ReduceSumOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto axis = op.getAxis();
    auto loc = op.getLoc();
    auto reducesum0 = rewriter.create<tosa::ReduceSumOp>(loc, input, axis);
    rewriter.replaceOp(op, reducesum0);
    return success();
  }
};

class ReshapeLoweringPattern : public OpRewritePattern<mix::ReshapeOp> {
public:
  using OpRewritePattern<mix::ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::ReshapeOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto shape = op.getShape().getValue();
    llvm::SmallVector<int64_t> shapeNum;
    for (auto attr : shape) {
      auto numAttr = dyn_cast<IntegerAttr>(attr);
      auto num = numAttr.getInt();
      shapeNum.push_back(num);
    }
    auto loc = op->getLoc();
    auto reshape0 = rewriter.create<tosa::ReshapeOp>(
        loc, input, rewriter.getDenseI64ArrayAttr(shapeNum));
    rewriter.replaceOp(op, reshape0);
    return success();
  }
};

class RsqrtSumLoweringPattern : public OpRewritePattern<mix::RsqrtOp> {
public:
  using OpRewritePattern<mix::RsqrtOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::RsqrtOp op,
                                PatternRewriter &rewriter) const override {

    auto input = op.getInput();
    auto loc = op->getLoc();
    auto resultType = op.getType();
    auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
    Value newop;
    if (!resultTensorType) {
      // TODO
      return op.emitOpError() << "Not support scale pow now.";
    } else {
      newop = rewriter.create<tosa::RsqrtOp>(loc, resultTensorType, input);
    }
    rewriter.replaceOp(op, newop);
    return success();
  }
};

#define GET_DENSE(T)                                                           \
  std::vector<T> castedResult;                                                 \
  for (auto e : result) {                                                      \
    castedResult.push_back(T(e));                                              \
  }                                                                            \
  dataAttr = DenseElementsAttr::get(returnType, ArrayRef<T>(castedResult));

class WeightOpLoweringPattern : public OpRewritePattern<mix::WeightOp> {
public:
  using OpRewritePattern<mix::WeightOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::WeightOp op,
                                PatternRewriter &rewriter) const override {

    auto returnType = op.getType();
    auto elementType = returnType.getElementType();
    auto data_loc = op.getParamLoc();
    auto loc = op->getLoc();
    std::vector<double> result;
    // read weight from json file.
    if (!getElementFromJson(data_loc.str(), result)) {
      return op->emitOpError()
             << "error happended when reading data from: " << data_loc;
    }
    // generate dense attr
    DenseElementsAttr dataAttr;
    if (auto floatElemType = llvm::dyn_cast<FloatType>(elementType)) {
      auto width = floatElemType.getWidth();
      if (width == 32) {
        GET_DENSE(float)
      } else if (width == 64) {
        GET_DENSE(double)
      } else {
        op.emitOpError() << "unsupported element type.";
      }
    } else if (auto intElemType = llvm::dyn_cast<IntegerType>(elementType)) {
      auto width = intElemType.getWidth();
      if (width == 32) {
        GET_DENSE(int)
      } else if (width == 64) {
        GET_DENSE(long)
      } else {
        op.emitOpError() << "unsupported element type.";
      }
    } else {
      return op.emitOpError() << "Unexpected type.";
    }
    // create consatant op
    auto constantOp = rewriter.create<mix::ConstantOp>(loc, dataAttr);
    rewriter.replaceOp(op, constantOp);
    return success();
  }
};

#undef GET_DENSE

class TanhOpLoweringPattern : public OpRewritePattern<mix::TanhOp> {
public:
  using OpRewritePattern<mix::TanhOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::TanhOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto input = op.getInput();
    auto resultType = op.getType();
    auto resultTensorType = dyn_cast<RankedTensorType>(resultType);
    Value newop;
    if (!resultTensorType) {
      // TODO
      return op.emitOpError() << "Not support scale pow now.";
    } else {
      newop = rewriter.create<tosa::TanhOp>(loc, resultTensorType, input);
    }
    rewriter.replaceOp(op, newop);
    return success();
  }
};

class ConstantLoweringPattern : public OpRewritePattern<mix::ConstantOp> {
public:
  using OpRewritePattern<mix::ConstantOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::ConstantOp op,
                                PatternRewriter &rewriter) const override {
    auto value = op.getValue();
    auto loc = op.getLoc();
    auto constOp = rewriter.create<mlir::arith::ConstantOp>(loc, value);
    rewriter.replaceOp(op, constOp);
    return success();
  }
};
} // namespace

void populateLowerPrimaryToTosaPatterns(RewritePatternSet &patterns) {
  patterns.add<AddLoweringPattern, SubLoweringPattern, MulLoweringPattern,
               DivLoweringPattern, MatmulLoweringPattern, NegLoweringPattern,
               ExpLoweringPattern, PowLoweringPattern, ReduceSumLoweringPattern,
               ReshapeLoweringPattern, RsqrtSumLoweringPattern,
               WeightOpLoweringPattern, TanhOpLoweringPattern,
               ConstantLoweringPattern>(patterns.getContext());
}

namespace {
class LowerPrimaryToTosaPass
    : public PassWrapper<LowerPrimaryToTosaPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerPrimaryToTosaPass)
  LowerPrimaryToTosaPass() = default;
  LowerPrimaryToTosaPass(const LowerPrimaryToTosaPass &) {}
  StringRef getArgument() const final { return "lower-mix-primary-to-tosa"; }
  StringRef getDescription() const final {
    return "Convert mix.prim ops to tosa.";
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tosa::TosaDialect,
                    tensor::TensorDialect>();
  }
};
} // namespace

void LowerPrimaryToTosaPass::runOnOperation() {
  MLIRContext &context = this->getContext();
  ModuleOp module = this->getOperation();
  ConversionTarget target(context);
  target.addLegalDialect<arith::ArithDialect, ml_program::MLProgramDialect,
                         mix::MIXDialect, tosa::TosaDialect,
                         tensor::TensorDialect,
                         bufferization::BufferizationDialect>();
  target.addIllegalOp<mix::AddOp, mix::SubOp, mix::MulOp, mix::DivOp,
                      mix::MatMulOp, mix::NegOp, mix::ExpOp, mix::PowOp,
                      mix::ReduceSumOp, mix::ReshapeOp, mix::RsqrtOp,
                      mix::WeightOp, mix::TanhOp, mix::ConstantOp>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&context);
  populateLowerPrimaryToTosaPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

void registerLowerPrimaryToTosaPass() {
  PassRegistration<LowerPrimaryToTosaPass>();
}

std::unique_ptr<Pass> createLowerPrimaryToTosa() {
  return std::make_unique<LowerPrimaryToTosaPass>();
}