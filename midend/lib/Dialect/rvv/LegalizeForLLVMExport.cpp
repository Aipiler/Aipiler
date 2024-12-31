#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

#include "rvv/rvvDialect.h"
#include "rvv/Transforms.h"

using namespace mlir;
using namespace rvv;

// RVVTargetIndexBitwidth can be 64(by default) or 32(when rv32 option is set)
// to configure the size of operands' type in lowering.

static int64_t RVVTargetIndexBitwidth;

template <typename SourceOp, typename TargetOp>
class ConvertPassthruOperandOpToLLVMPattern
    : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  /// This pattern creates an `undef` operation, inserts the `undef`
  /// operation to the beginning of the operand list, and creates the intrinsic
  /// operation.
  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    unsigned numResults = op->getNumResults();
    auto resultType = op->getResultTypes();
    Type packedType;

    Value src1 = op.getOperand(0);
    Value src2 = op.getOperand(1);
    Value vl = op.getOperand(2);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           op.getLoc(),
                           rewriter.getIntegerType(RVVTargetIndexBitwidth), vl)
                       .getResult(0);
    SmallVector<Value, 6> operandsVector({src1, src2, vlCast});

    Value passthru = rewriter.create<LLVM::UndefOp>(loc, resultType[0]);
    operandsVector.insert(operandsVector.begin(), passthru);

    const LLVMTypeConverter *typeConverter = this->getTypeConverter();
    if (numResults != 0) {
      packedType = typeConverter->packFunctionResults(op->getResultTypes());
      if (!packedType)
        return failure();
    }

    // Create the intrinsic operation.
    OperationState state(loc, TargetOp::getOperationName());
    state.addTypes(packedType);
    state.addOperands(operandsVector);
    Operation *newOp = rewriter.create(state);
    return rewriter.replaceOp(op, newOp->getResult(0)), success();
  }
};

template <typename SourceOp, typename TargetOp>
class ConvertPassthruOperandRoundingOpToLLVMPattern
    : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  /// This pattern creates an `undef` operation, inserts the `undef`
  /// operation to the beginning of the operand list, and creates the intrinsic
  /// operation.
  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    unsigned numResults = op->getNumResults();
    auto resultType = op->getResultTypes();
    Type packedType;

    Value src1 = op.getOperand(0);
    Value src2 = op.getOperand(1);
    Value frm = op.getOperand(2);
    Value vl = op.getOperand(3);
    Value frmCast = rewriter
                        .create<UnrealizedConversionCastOp>(
                            op.getLoc(),
                            rewriter.getIntegerType(RVVTargetIndexBitwidth), frm)
                        .getResult(0);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           op.getLoc(),
                           rewriter.getIntegerType(RVVTargetIndexBitwidth), vl)
                       .getResult(0);
    SmallVector<Value, 6> operandsVector({src1, src2, frmCast, vlCast});

    Value passthru = rewriter.create<LLVM::UndefOp>(loc, resultType[0]);
    operandsVector.insert(operandsVector.begin(), passthru);

    const LLVMTypeConverter *typeConverter = this->getTypeConverter();
    if (numResults != 0) {
      packedType = typeConverter->packFunctionResults(op->getResultTypes());
      if (!packedType)
        return failure();
    }

    // Create the intrinsic operation.
    OperationState state(loc, TargetOp::getOperationName());
    state.addTypes(packedType);
    state.addOperands(operandsVector);
    Operation *newOp = rewriter.create(state);
    return rewriter.replaceOp(op, newOp->getResult(0)), success();
  }
};

// TODO: Check if the changes will cause errors
template <typename OpTy>
class ForwardOperands : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (adaptor.getOperands().getTypes() == op->getOperands().getTypes())
      return rewriter.notifyMatchFailure(op, "operand types already match");

    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

// TODO: Check if the changes will cause errors
class ReturnOpTypeConversion : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct RVVSetVlOpLowering : public ConvertOpToLLVMPattern<RVVSetVlOp> {
  using ConvertOpToLLVMPattern<RVVSetVlOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RVVSetVlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = rewriter.getIntegerType(RVVTargetIndexBitwidth);
    Value avl = op.getOperand(0);
    Value avlCast =
        rewriter
            .create<UnrealizedConversionCastOp>(
                op.getLoc(), rewriter.getIntegerType(RVVTargetIndexBitwidth),
                avl)
            .getResult(0);
    Value sew = op.getOperand(1);
    Value sewCast =
        rewriter
            .create<UnrealizedConversionCastOp>(
                op.getLoc(), rewriter.getIntegerType(RVVTargetIndexBitwidth),
                sew)
            .getResult(0);
    Value lmul = op.getOperand(2);
    Value lmulCast =
        rewriter
            .create<UnrealizedConversionCastOp>(
                op.getLoc(), rewriter.getIntegerType(RVVTargetIndexBitwidth),
                lmul)
            .getResult(0);
    rewriter.replaceOpWithNewOp<RVVIntrSetVlIOp>(op, resultType, avlCast,
                                                 sewCast, lmulCast);
    return success();
  }
};

// TODO: Check if the changes will cause errors
struct RVVLoadOpLowering : public ConvertOpToLLVMPattern<RVVLoadOp> {
  using ConvertOpToLLVMPattern<RVVLoadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RVVLoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = loadOp.getMemRefType();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    LLVMTypeConverter converter(loadOp.getContext());

    auto resultType = loadOp.getResult().getType();
    Value passthru =
        rewriter.create<LLVM::UndefOp>(loadOp.getLoc(), resultType);
    LLVM::LLVMPointerType llvmDataTypePtr =
        LLVM::LLVMPointerType::get(loadOp.getContext());
    Value dataPtr = getStridedElementPtr(
        loadOp.getLoc(), type, adaptor.getBase(), adaptor.getIndex(), rewriter);
    Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
        loadOp.getLoc(), llvmDataTypePtr, dataPtr);
    Value vl = loadOp.getOperand(2);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           loadOp.getLoc(),
                           rewriter.getIntegerType(RVVTargetIndexBitwidth), vl)
                       .getResult(0);
    rewriter.replaceOpWithNewOp<RVVIntrLoadEleOp>(loadOp, resultType, passthru,
                                                  bitCastedPtr, vlCast);
    return success();
  }
};

// TODO: Check if the changes will cause errors
struct RVVStoreOpLowering : public ConvertOpToLLVMPattern<RVVStoreOp> {
  using ConvertOpToLLVMPattern<RVVStoreOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RVVStoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = storeOp.getMemRefType();
    if (!isConvertibleAndHasIdentityMaps(type))
      return failure();

    LLVMTypeConverter converter(storeOp.getContext());

    auto resultType = storeOp.getValue().getType();
    LLVM::LLVMPointerType llvmDataTypePtr =
        LLVM::LLVMPointerType::get(storeOp.getContext());
    Value dataPtr =
        getStridedElementPtr(storeOp.getLoc(), type, adaptor.getBase(),
                             adaptor.getIndex(), rewriter);
    Value bitCastedPtr = rewriter.create<LLVM::BitcastOp>(
        storeOp.getLoc(), llvmDataTypePtr, dataPtr);
    Value vl = storeOp.getOperand(3);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           storeOp.getLoc(),
                           rewriter.getIntegerType(RVVTargetIndexBitwidth), vl)
                       .getResult(0);
    rewriter.replaceOpWithNewOp<RVVIntrStoreEleOp>(storeOp, adaptor.getValue(),
                                                   bitCastedPtr, vlCast);
    return success();
  }
};

using RVVAddOpLowering =
    ConvertPassthruOperandOpToLLVMPattern<RVVAddOp, RVVIntrAddOp>;
using RVVMulOpLowering =
    ConvertPassthruOperandOpToLLVMPattern<RVVMulOp, RVVIntrMulOp>;
using RVVFMaxOpLowering =
    ConvertPassthruOperandOpToLLVMPattern<RVVFMaxOp, RVVIntrFMaxOp>;
using RVVFMinOpLowering =
    ConvertPassthruOperandOpToLLVMPattern<RVVFMinOp, RVVIntrFMinOp>;
using RVVFAddOpLowering =
    ConvertPassthruOperandRoundingOpToLLVMPattern<RVVFAddOp, RVVIntrFAddOp>;
using RVVFMulOpLowering =
    ConvertPassthruOperandRoundingOpToLLVMPattern<RVVFMulOp, RVVIntrFMulOp>;

struct RsqrtOpLowering : public ConvertOpToLLVMPattern<RsqrtOp> {
  using ConvertOpToLLVMPattern<RsqrtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RsqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultType = op.getResult().getType();
    Value passthru = rewriter.create<LLVM::UndefOp>(op.getLoc(), resultType);
    Value src = op.getOperand(0);
    Value vl = op.getOperand(1);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           op.getLoc(),
                           rewriter.getIntegerType(RVVTargetIndexBitwidth), vl)
                       .getResult(0);
    rewriter.replaceOpWithNewOp<IntrFrsqrt7Op>(op, resultType, passthru, src,
                                               vlCast);
    return success();
  }
};

template <typename SourceOp, typename TargetOp>
struct UnaryAAUnMaskedRoundingModeOpLowering : public ConvertOpToLLVMPattern<SourceOp> {
  using ConvertOpToLLVMPattern<SourceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultType = op.getResult().getType();
    Value passthru = rewriter.create<LLVM::UndefOp>(op.getLoc(), resultType);
    Value src = op.getOperand(0);
    Value frm = op.getOperand(1);
    Value vl = op.getOperand(2);
    Value frmCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           op.getLoc(),
                           rewriter.getIntegerType(RVVTargetIndexBitwidth), frm)
                       .getResult(0);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(
                           op.getLoc(),
                           rewriter.getIntegerType(RVVTargetIndexBitwidth), vl)
                       .getResult(0);
    rewriter.replaceOpWithNewOp<TargetOp>(op, resultType, passthru, src,
                                               frmCast, vlCast);
    return success();
  }
};

using RVVFrec7OpLowering =
    UnaryAAUnMaskedRoundingModeOpLowering<RVVFrec7Op, RVVIntrFrec7Op>;
using RVVFsqrtOpLowering =
    UnaryAAUnMaskedRoundingModeOpLowering<RVVFsqrtOp, RVVIntrFsqrtOp>;

struct RVVMAccOpLowering : public ConvertOpToLLVMPattern<RVVMAccOp> {
  using ConvertOpToLLVMPattern<RVVMAccOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RVVMAccOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto resultType = op.getResult().getType();
    Value src1 = op.getOperand(0);
    Value src2 = op.getOperand(1);
    Value src3 = op.getOperand(2);
    Value vl = op.getOperand(3);
    Value vlCast = rewriter
                       .create<UnrealizedConversionCastOp>(loc,
                           rewriter.getIntegerType(RVVTargetIndexBitwidth),
                           vl)
                       .getResult(0);
    
    ValueRange operands = adaptor.getOperands();
    // Get the type of the `vl` value.
    Type vlType = operands.back().getType();
    auto attrs = op->getAttrs();
    Value vtaValue;
    if (attrs.empty()) {
      // Default attribute for the vta setting (vta = 1).
      // Add the vta = 1 to the operand list.
      Attribute vtaDefaultAttr = rewriter.getIntegerAttr(
          vlType, APInt(vlType.cast<IntegerType>().getWidth(), 0));
     vtaValue =
          rewriter.create<LLVM::ConstantOp>(loc, vlType, vtaDefaultAttr);
    } else if (attrs.size() == 1) {
      // Add the vta to the operand list according to the attribute value.
      Attribute attr = attrs[0].getValue();
      IntegerAttr vtaAttr = attr.cast<IntegerAttr>();
      vtaValue = rewriter.create<LLVM::ConstantOp>(loc, vlType, vtaAttr);
    } else {
      return failure();
    }

    rewriter.replaceOpWithNewOp<RVVIntrMAccOp>(op, resultType, src1, src2,
                                               src3, vlCast, vtaValue);
    return success();
  }
};

/// Populate the given list with patterns that convert from RVV to LLVM.
void mlir::populateRVVLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns,
    int64_t RVVIndexBitwidth) {

  RVVTargetIndexBitwidth = RVVIndexBitwidth;
  // clang-format off
  patterns.add<ForwardOperands<func::CallOp>,
               ForwardOperands<func::CallIndirectOp>,
               ForwardOperands<func::ReturnOp>
               >(converter, &converter.getContext());
  patterns.add<RVVSetVlOpLowering>(converter);
  patterns.add<RVVLoadOpLowering,
               RVVStoreOpLowering>(converter);
  patterns.add<RsqrtOpLowering, 
               RVVFrec7OpLowering,
               RVVFsqrtOpLowering>(converter);
  patterns.add<RVVMAccOpLowering>(converter);
  patterns.add<RVVAddOpLowering,
               RVVMulOpLowering,
               RVVFMaxOpLowering,
               RVVFMinOpLowering,
               RVVFAddOpLowering,
               RVVFMulOpLowering>(converter);
  // clang-format on
}

void mlir::configureRVVLegalizeForExportTarget(LLVMConversionTarget &target) {
  // clang-format off
  target.addLegalOp<RVVIntrSetVlIOp,
                    RVVIntrLoadEleOp,
                    RVVIntrStoreEleOp,
                    IntrFrsqrt7Op,
                    RVVIntrFsqrtOp,
                    RVVIntrFrec7Op,
                    RVVIntrMAccOp,
                    RVVIntrAddOp,
                    RVVIntrMulOp,
                    RVVIntrFMaxOp,
                    RVVIntrFMinOp,
                    RVVIntrFAddOp,
                    RVVIntrFMulOp>();
  target.addIllegalOp<RVVSetVlOp,
                      RVVLoadOp,
                      RVVStoreOp,
                      RsqrtOp,
                      RVVFsqrtOp,
                      RVVFrec7Op,
                      RVVMAccOp,
                      RVVAddOp,
                      RVVMulOp,
                      RVVFMaxOp,
                      RVVFMinOp,
                      RVVFAddOp,
                      RVVFMulOp>();
  // clang-format on
}
