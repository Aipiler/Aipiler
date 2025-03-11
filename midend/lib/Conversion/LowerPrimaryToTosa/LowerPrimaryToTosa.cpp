#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

#include "mix/mixDialect.h"
#include "mix/mixOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

using namespace mlir;

namespace {

template <typename SourceOp, typename TargetOp>
class UnaryLoweringPattern : public OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto input = op.getInput();
    Type resultType = op.getType();
    // auto resultTensorType = resultType.dyn_cast<RankedTensorType>();
    Value newop;
    if (auto resultTensorType = resultType.dyn_cast<RankedTensorType>()) {
      newop = rewriter.create<TargetOp>(loc, resultTensorType, input);
    } else {
      newop = rewriter.create<TargetOp>(loc, resultType, input);
    }
    rewriter.replaceOp(op, newop);
    return success();
  }
};
// using NegLoweringPattern = UnaryLoweringPattern<mix::NegOp, tosa::NegateOp>;
// using ExpLoweringPattern = UnaryLoweringPattern<mix::ExpOp, tosa::ExpOp>;
// using RsqrtLoweringPattern = UnaryLoweringPattern<mix::RsqrtOp,
// tosa::RsqrtOp>; using TanhLoweringPattern = UnaryLoweringPattern<mix::TanhOp,
// tosa::TanhOp>; using ReciprocalLoweringPattern =
//     UnaryLoweringPattern<mix::ReciprocalOp, tosa::ReciprocalOp>;
// using CosLoweringPattern = UnaryLoweringPattern<mix::CosOp, tosa::CosOp>;
// using SinLoweringPattern = UnaryLoweringPattern<mix::SinOp, tosa::SinOp>;

template <typename SourceOp, typename Target1Op, typename Target2Op>
class Unary2LoweringPattern : public OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto input = op.getInput();
    Type resultType = op.getType();
    // auto resultTensorType = resultType.dyn_cast<RankedTensorType>();
    Value newop;
    if (auto resultTensorType = resultType.dyn_cast<RankedTensorType>()) {
      newop = rewriter.create<Target1Op>(loc, resultTensorType, input);
    } else {
      newop = rewriter.create<Target2Op>(loc, resultType, input);
    }
    rewriter.replaceOp(op, newop);
    return success();
  }
};

using NegLoweringPattern = UnaryLoweringPattern<mix::NegOp, tosa::NegateOp>;
using ExpLoweringPattern = UnaryLoweringPattern<mix::ExpOp, tosa::ExpOp>;
using RsqrtLoweringPattern = UnaryLoweringPattern<mix::RsqrtOp, tosa::RsqrtOp>;
using TanhLoweringPattern = UnaryLoweringPattern<mix::TanhOp, tosa::TanhOp>;
using ReciprocalLoweringPattern =
    UnaryLoweringPattern<mix::ReciprocalOp, tosa::ReciprocalOp>;
using CosLoweringPattern =
    Unary2LoweringPattern<mix::CosOp, tosa::CosOp, math::CosOp>;
using SinLoweringPattern =
    Unary2LoweringPattern<mix::SinOp, tosa::SinOp, math::SinOp>;

template <typename SourceOp, typename Target0Op, typename Target1OpI,
          typename Target1OpF>
class BinaryLoweringPattern : public OpRewritePattern<SourceOp> {
public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto lhs = op.getOperand(0);
    auto rhs = op.getOperand(1);
    Type lhsType = lhs.getType();
    Type rhsType = rhs.getType();
    Type resultType = op.getType();
    Value newop;
    if (auto resultTensorType = dyn_cast<RankedTensorType>(resultType)) {
      auto lhsTensorType = lhsType.dyn_cast<RankedTensorType>();
      auto rhsTensorType = rhsType.dyn_cast<RankedTensorType>();
      auto elemTy = resultTensorType.getElementType();
      auto tensorTy = RankedTensorType::get({1}, elemTy);
      if (!lhsTensorType) {
        lhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, lhs);
      }
      if (!rhsTensorType) {
        rhs = rewriter.create<tensor::FromElementsOp>(loc, tensorTy, rhs);
      }
      newop = rewriter.create<Target0Op>(loc, resultTensorType, lhs, rhs);
    } else {
      if (auto resIntType = dyn_cast<IntegerType>(resultType)) {
        newop = rewriter.create<Target1OpI>(loc, lhs, rhs);
      } else if (auto resFloatType = dyn_cast<FloatType>(resultType)) {
        newop = rewriter.create<Target1OpF>(loc, lhs, rhs);
      } else {
        return op.emitOpError() << "Unexpected types.";
      }
    }
    rewriter.replaceOp(op, newop);
    return success();
  }
};

using AddLoweringPattern = BinaryLoweringPattern<mix::AddOp, tosa::AddOp,
                                                 arith::AddIOp, arith::AddFOp>;
using SubLoweringPattern = BinaryLoweringPattern<mix::SubOp, tosa::SubOp,
                                                 arith::SubIOp, arith::SubFOp>;
// using MulLoweringPattern = BinaryLoweringPattern<mix::MulOp, tosa::MulOp,
// arith::MulIOp, arith::MulFOp>;
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

class LtLoweringPattern : public OpRewritePattern<mix::LtOp> {
public:
  using OpRewritePattern<mix::LtOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::LtOp op,
                                PatternRewriter &rewriter) const override {
    auto context = rewriter.getContext();
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

        newop = rewriter.create<arith::CmpIOp>(
            loc,
            ::mlir::arith::CmpIPredicateAttr::get(
                context, mlir::arith::CmpIPredicate::slt),
            lhs, rhs); // slt
      } else if (auto resFloatType = dyn_cast<FloatType>(resultType)) {
        newop = rewriter.create<arith::CmpFOp>(
            loc,
            ::mlir::arith::CmpFPredicateAttr::get(
                context, mlir::arith::CmpFPredicate::OLT),
            lhs, rhs);
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

      newop = rewriter.create<tosa::GreaterOp>(loc, op.getType(), rhs, lhs);
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

class BatchMatmulLoweringPattern : public OpRewritePattern<mix::BatchMatMulOp> {
public:
  using OpRewritePattern<mix::BatchMatMulOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::BatchMatMulOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto resType = op.getType();
    auto loc = op.getLoc();
    auto lhsType = lhs.getType();
    auto rhsType = rhs.getType();
    auto lhsShape = lhsType.getShape();

    Value newop;
    if (lhsShape.size() != 3) {
      auto rhsShape = rhsType.getShape();
      auto resShape = resType.getShape();
      SmallVector<int64_t> newLhsShape;
      SmallVector<int64_t> newRhsShape;
      SmallVector<int64_t> newResShape;
      switch (lhsShape.size()) {
      case 1:
        newLhsShape = {1, 1, lhsShape[1]};
        newRhsShape = {1, 1, rhsShape[1]};
        newResShape = {1, 1, resShape[1]};
        break;
      case 2:
        newLhsShape = {1, lhsShape[0], lhsShape[1]};
        newRhsShape = {1, rhsShape[0], rhsShape[1]};
        newResShape = {1, resShape[0], resShape[1]};
        break;
      default:
        llvm_unreachable("unsupported shape");
        auto newResType =
            RankedTensorType::get(newResShape, resType.getElementType());
        auto newLhs = rewriter.create<tosa::ReshapeOp>(
            loc, lhs, rewriter.getDenseI64ArrayAttr(newLhsShape));
        auto newRhs = rewriter.create<tosa::ReshapeOp>(
            loc, rhs, rewriter.getDenseI64ArrayAttr(newRhsShape));
        newop =
            rewriter.create<tosa::MatMulOp>(loc, newResType, newLhs, newRhs);
      }
    }
    newop = rewriter.create<tosa::MatMulOp>(loc, resType, lhs, rhs);
    rewriter.replaceOp(op, newop);
    return success();
  }
};

class PowLoweringPattern : public OpRewritePattern<mix::PowOp> {
public:
  using OpRewritePattern<mix::PowOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::PowOp op,
                                PatternRewriter &rewriter) const override {

    auto lhs = op.getOperand(0);
    auto rhs = op.getOperand(1);
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
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto inputType = input.getType();
    auto inputShape = inputType.getShape();
    auto elementType = inputType.getElementType();
    auto axis = int(op.getAxis());

    SmallVector<int64_t> reduceShape;
    SmallVector<Value> dynDims;
    for (unsigned i = 0; i < inputType.getRank(); i++) {
      if (axis != i) {
        reduceShape.push_back(inputType.getDimSize(i));
      }
    }

    // 创建全 0 的 init 张量
    auto zeroAttr = rewriter.getZeroAttr(elementType);
    auto initType = RankedTensorType::get(reduceShape, elementType);
    auto initAttr = DenseElementsAttr::get(initType, zeroAttr);
    auto init = rewriter.create<arith::ConstantOp>(loc, initType, initAttr);

    auto newop = rewriter.create<linalg::ReduceOp>(
        loc,                     // 操作位置
        ValueRange{input},       // 输入张量
        ValueRange{init},        // 初始值
        ArrayRef<int64_t>{axis}, // 归约维度
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          // args[0] 是累加器 (accumulator)，args[1] 是当前输入元素
          Value sum = nestedBuilder.create<mlir::arith::AddFOp>(
              nestedLoc, args[0], args[1]); // 创建浮点加法操作
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, sum); // 返回求和结果
        });

    auto reshape0 = rewriter.create<tosa::ReshapeOp>(
        loc, newop.getResult(0),
        rewriter.getDenseI64ArrayAttr(
            op.getType().cast<RankedTensorType>().getShape()));
    rewriter.replaceOp(op, reshape0);
    return success();
  }
};

class ConcatLoweringPattern : public OpConversionPattern<mix::ConcatOp> {
public:
  using OpConversionPattern<mix::ConcatOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(mix::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto inputs = op.getODSOperands(0);
    auto axis = op.getAxis();
    mlir::ValueRange valueRange(inputs);
    auto newop = rewriter.create<tosa::ConcatOp>(loc, valueRange, axis);
    rewriter.replaceOp(op, newop);
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

class SliceLoweringPattern : public OpRewritePattern<mix::SliceOp> {
public:
  using OpRewritePattern<mix::SliceOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::SliceOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInputs();
    auto dim = op.getDim();
    auto begin = op.getStart();
    auto end = op.getEnd();
    auto step = op.getStep();
    auto shape = input.getType().getShape();
    auto rank = shape.size();
    auto resultType = dyn_cast<ShapedType>(op.getType());
    if (end > shape[dim])
      end = shape[dim];
    if (llvm::isa<UnrankedTensorType>(resultType))
      return failure();
    SmallVector<int64_t> offset(rank, 0);
    SmallVector<int64_t> size = llvm::to_vector(shape);
    SmallVector<int64_t> stride(rank, 1);
    offset[dim] = begin;
    size[dim] = end - begin;
    stride[dim] = step;

    auto newop = rewriter.create<tensor::ExtractSliceOp>(
        loc, op.getType(), input, ValueRange({}), ValueRange({}),
        ValueRange({}), rewriter.getDenseI64ArrayAttr(offset),
        rewriter.getDenseI64ArrayAttr(size),
        rewriter.getDenseI64ArrayAttr(stride));

    rewriter.replaceOp(op, newop);
    return success();
  }
};

class TransposeLoweringPattern : public OpRewritePattern<mix::TransposeOp> {
public:
  using OpRewritePattern<mix::TransposeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto dim1 = op.getDim1();
    auto dim2 = op.getDim2();
    auto resultType = dyn_cast<ShapedType>(op.getType());
    if (llvm::isa<UnrankedTensorType>(resultType))
      return failure();

    SmallVector<int32_t> dimsVector;
    auto rank = resultType.getRank();
    for (int32_t i = 0; i < rank; i++) {
      dimsVector.push_back(i);
    }
    dimsVector[dim1] = dim2;
    dimsVector[dim2] = dim1;
    auto dimAttr = rewriter.getI32TensorAttr(dimsVector);

    auto perm = rewriter.create<arith::ConstantOp>(loc, dimAttr);

    auto newop =
        rewriter.create<tosa::TransposeOp>(loc, op.getType(), input, perm);

    rewriter.replaceOp(op, newop);
    return success();
  }
};

class PermuteLoweringPattern : public OpRewritePattern<mix::PermuteOp> {
public:
  using OpRewritePattern<mix::PermuteOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::PermuteOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto dimsAttr = op.getDims();
    auto resultType = dyn_cast<ShapedType>(op.getType());
    if (llvm::isa<UnrankedTensorType>(resultType))
      return failure();

    SmallVector<int32_t> dimsVector;
    for (auto dim : dimsAttr) {
      dimsVector.push_back(mlir::cast<IntegerAttr>(dim).getInt());
    }

    auto dimAttr = rewriter.getI32TensorAttr(dimsVector);
    auto perm = rewriter.create<arith::ConstantOp>(loc, dimAttr);
    auto newop =
        rewriter.create<tosa::TransposeOp>(loc, op.getType(), input, perm);

    rewriter.replaceOp(op, newop);
    return success();
  }
};

class ConvertLoweringPattern : public OpRewritePattern<mix::ConvertOp> {
public:
  using OpRewritePattern<mix::ConvertOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::ConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getValue();
    auto resultType = op.getType();

    Value newop;
    if (auto resTensorType = mlir::dyn_cast<RankedTensorType>(resultType)) {
      newop = rewriter.create<tosa::CastOp>(loc, resultType, input);
    } else // TODO: airth::BitcastOp must accept the same bit width.
      newop = rewriter.create<arith::BitcastOp>(loc, resultType, input);

    rewriter.replaceOp(op, newop);
    return success();
  }
};

class UnsqueezeLoweringPattern : public OpRewritePattern<mix::UnsqueezeOp> {
public:
  using OpRewritePattern<mix::UnsqueezeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::UnsqueezeOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto axis = op.getAxis();
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType)
      return failure();
    auto inputShape = inputType.getShape();
    auto rank = inputType.getRank();
    auto elemTy = inputType.getElementType();
    if (!elemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(
          op, "Only floating-point or integer datatype legalization supported");
    }

    if (axis > rank)
      return rewriter.notifyMatchFailure(op, "axis is invalid");

    llvm::SmallVector<int64_t> shapeNum;
    for (auto idx = 0; idx < rank; idx++) {
      if (idx == axis) {
        shapeNum.push_back(1);
        shapeNum.push_back(inputShape[idx]);
      } else {
        shapeNum.push_back(inputShape[idx]);
      }
    }
    if (axis == rank)
      shapeNum.push_back(1);

    rewriter.replaceOpWithNewOp<tosa::ReshapeOp>(
        op, mlir::dyn_cast<RankedTensorType>(op.getType()), input,
        rewriter.getDenseI64ArrayAttr(shapeNum));
    return success();
  }
};

class GatherLoweringPattern : public OpRewritePattern<mix::GatherOp> {
public:
  using OpRewritePattern<mix::GatherOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::GatherOp op,
                                PatternRewriter &rewriter) const override {
    auto gather0 = rewriter.create<tosa::GatherOp>(
        op->getLoc(), op.getType(), op.getValues(), op.getIndices());
    rewriter.replaceOp(op, gather0);
    return success();
  }
};

class MaskFillLoweringPattern : public OpRewritePattern<mix::MaskedFillOp> {
public:
  using OpRewritePattern<mix::MaskedFillOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mix::MaskedFillOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto mask = op.getMask();
    auto onTrue = op.getNumber();
    auto onTrueTensorType =
        RankedTensorType::get(ArrayRef<int64_t>{1}, onTrue.getType());
    auto onTureTensor =
        rewriter.create<tensor::FromElementsOp>(loc, onTrueTensorType, onTrue);
    auto select0 = rewriter.create<tosa::SelectOp>(loc, input.getType(), mask,
                                                   onTureTensor, input);
    rewriter.replaceOp(op, select0);
    return success();
  }
};

class SoftmaxLoweringPattern : public OpRewritePattern<mix::SoftmaxOp> {
public:
  using OpRewritePattern<mix::SoftmaxOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mix::SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto signed_axis = op.getAxis();

    // 处理负轴索引，确保为正轴索引
    uint64_t axis =
        signed_axis < 0
            ? input.getType().cast<RankedTensorType>().getRank() + signed_axis
            : signed_axis;
    auto outputType = op.getType();
    auto elementType =
        input.getType().cast<RankedTensorType>().getElementType();
    auto outputShape = outputType.cast<RankedTensorType>().getShape();
    auto rank = outputShape.size();

    // 1. 计算沿指定 axis 的最大值 (reduceMax)
    auto reduceMax = rewriter.create<tosa::ReduceMaxOp>(loc, input, axis);

    // 广播 reduceMax 的结果以匹配输入形状
    // auto broadcastReduceMax = rewriter.create<tosa::BroadcastOp>(
    //     loc, input.getType(), reduceMax.getResult(),
    //     rewriter.getI64ArrayAttr(outputShape));

    // 2. 输入减去广播后的 max
    auto inputMinusMax = rewriter.create<mix::SubOp>(loc, input, reduceMax);

    // 3. 创建 Sum Tensor 的初始值（为 0 的张量）
    llvm::SmallVector<int64_t, 4> sumTensorShape(outputShape.begin(),
                                                 outputShape.end());
    sumTensorShape[axis] = 1;
    auto sumTensorType =
        mlir::RankedTensorType::get(sumTensorShape, elementType);
    auto zeroAttr = rewriter.getZeroAttr(elementType);
    auto sumTensorInit = rewriter.create<mlir::arith::ConstantOp>(
        loc, sumTensorType,
        mlir::DenseElementsAttr::get(sumTensorType, zeroAttr));

    // 4. 创建 AffineMap 用于描述输入和输出的关系
    llvm::SmallVector<mlir::AffineExpr, 4> inputExprs, sumExprs, resultExprs;
    for (int i = 0; i < rank; ++i) {
      inputExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
      resultExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
      if (i == axis)
        sumExprs.push_back(rewriter.getAffineConstantExpr(0));
      else
        sumExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
    }
    auto inputMap =
        mlir::AffineMap::get(rank, 0, inputExprs, rewriter.getContext());
    auto sumMap =
        mlir::AffineMap::get(rank, 0, sumExprs, rewriter.getContext());
    auto resultMap =
        mlir::AffineMap::get(rank, 0, resultExprs, rewriter.getContext());

    llvm::SmallVector<mlir::AffineMap, 4> indexingMaps = {inputMap, sumMap};

    // 定义迭代器类型：parallel 或 reduction
    llvm::SmallVector<mlir::utils::IteratorType, 4> iteratorTypes(
        rank, mlir::utils::IteratorType::parallel);
    iteratorTypes[axis] = mlir::utils::IteratorType::reduction;

    // 5. 创建 linalg::GenericOp 用于计算 sum(exp(input - max))
    auto sumTensorOp = rewriter.create<mlir::linalg::GenericOp>(
        loc, sumTensorType, mlir::ValueRange{inputMinusMax},
        mlir::ValueRange{sumTensorInit}, indexingMaps, iteratorTypes);

    // 填充 reduction 区域
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&sumTensorOp->getRegion(0), {},
                           {elementType, elementType}, {loc, loc});
      auto blockArgs = sumTensorOp.getRegion().front().getArguments();
      auto expOp = rewriter.create<mlir::math::ExpOp>(loc, blockArgs[0]);
      auto addOp =
          rewriter.create<mlir::arith::AddFOp>(loc, expOp, blockArgs[1]);
      rewriter.create<mlir::linalg::YieldOp>(loc, addOp.getResult());
    }

    // 6. 创建 Softmax 的最终输出
    auto finalResultTensor =
        rewriter.create<mlir::tensor::EmptyOp>(loc, outputShape, elementType);
    indexingMaps.push_back(resultMap); // 添加 resultMap
    auto softmaxOp = rewriter.create<mlir::linalg::GenericOp>(
        loc, outputType,
        mlir::ValueRange{inputMinusMax, sumTensorOp.getResult(0)},
        mlir::ValueRange{finalResultTensor}, indexingMaps, iteratorTypes);

    // 填充 softmax 的计算逻辑
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&softmaxOp->getRegion(0), {},
                           {elementType, elementType, elementType},
                           {loc, loc, loc});
      auto blockArgs = softmaxOp.getRegion().front().getArguments();
      auto expOp = rewriter.create<mlir::math::ExpOp>(loc, blockArgs[0]);
      auto divOp =
          rewriter.create<mlir::arith::DivFOp>(loc, expOp, blockArgs[1]);
      rewriter.create<mlir::linalg::YieldOp>(loc, divOp.getResult());
    }

    // 用 softmax 的结果替换原始操作
    rewriter.replaceOp(op, softmaxOp.getResult(0));
    return success();
  }
};

} // namespace
void populateLowerPrimaryToTosaPatterns(RewritePatternSet &patterns) {
  patterns.add<
      AddLoweringPattern, SubLoweringPattern, MulLoweringPattern,
      DivLoweringPattern, MatmulLoweringPattern, NegLoweringPattern,
      ExpLoweringPattern, PowLoweringPattern, ReduceSumLoweringPattern,
      ReshapeLoweringPattern, RsqrtLoweringPattern, TanhLoweringPattern,
      ConcatLoweringPattern, ReciprocalLoweringPattern, GatherLoweringPattern,
      CosLoweringPattern, SinLoweringPattern, BatchMatmulLoweringPattern,
      ConvertLoweringPattern, PermuteLoweringPattern, SliceLoweringPattern,
      UnsqueezeLoweringPattern, TransposeLoweringPattern,
      ConstantLoweringPattern, LtLoweringPattern, MaskFillLoweringPattern,
      SoftmaxLoweringPattern>(patterns.getContext());
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
    registry.insert<arith::ArithDialect, tosa::TosaDialect, math::MathDialect,
                    tensor::TensorDialect, linalg::LinalgDialect>();
  }
};
} // namespace

void LowerPrimaryToTosaPass::runOnOperation() {
  MLIRContext &context = this->getContext();
  ModuleOp module = this->getOperation();
  ConversionTarget target(context);
  target.addLegalDialect<
      arith::ArithDialect, ml_program::MLProgramDialect, mix::MIXDialect,
      tosa::TosaDialect, tensor::TensorDialect, math::MathDialect,
      bufferization::BufferizationDialect, linalg::LinalgDialect>();
  target.addIllegalOp<
      mix::AddOp, mix::SubOp, mix::MulOp, mix::DivOp, mix::MatMulOp,
      mix::BatchMatMulOp, mix::NegOp, mix::ExpOp, mix::PowOp, mix::ConcatOp,
      mix::ReduceSumOp, mix::ReshapeOp, mix::RsqrtOp, mix::TanhOp,
      mix::ReciprocalOp, mix::CosOp, mix::SinOp, mix::ConstantOp, mix::GatherOp,
      mix::ConvertOp, mix::PermuteOp, mix::SliceOp, mix::UnsqueezeOp,
      mix::TransposeOp, mix::LtOp, mix::MaskedFillOp, mix::SoftmaxOp>();
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