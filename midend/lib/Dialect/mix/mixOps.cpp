#include "mix/mixOps.h"
#include "mix/mixDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <optional>

using namespace mlir;

LogicalResult mix::SiLUOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = operands[0].getType();
  inferredReturnTypes.push_back(inputType);
  return success();
}

LogicalResult mix::SigmoidOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = operands[0].getType();
  inferredReturnTypes.push_back(inputType);
  return success();
}

bool verifyBroadcastCompatibility(TensorType lhsTensor, TensorType rhsTensor) {

  ArrayRef<int64_t> lhsShape = lhsTensor.getShape();
  ArrayRef<int64_t> rhsShape = rhsTensor.getShape();

  int lhsRank = lhsShape.size();
  int rhsRank = rhsShape.size();
  int maxRank = std::max(lhsRank, rhsRank);

  for (int i = 0; i < maxRank; ++i) {
    int64_t lhsDim = (i < lhsRank) ? lhsShape[lhsRank - 1 - i] : 1;
    int64_t rhsDim = (i < rhsRank) ? rhsShape[rhsRank - 1 - i] : 1;

    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1) {
      return false;
    }
  }
  return true;
}

LogicalResult mix::AddOp::verify() {
  /*
  verify:
    - element types are the same: refuse i32 + f32
      - both are Tensor: boardcastable and element
      - one is Tensor, another is Scale: success
      - both are Scale: success
  */
  auto lhsTy = this->getLhs().getType();
  auto rhsTy = this->getRhs().getType();
  auto lhsTensorTy = lhsTy.dyn_cast<RankedTensorType>();
  auto rhsTensorTy = rhsTy.dyn_cast<RankedTensorType>();
  // check element types.
  Type lhsElemTy = lhsTensorTy ? lhsTensorTy.getElementType() : lhsTy;
  Type rhsElemTy = rhsTensorTy ? rhsTensorTy.getElementType() : rhsTy;
  if (lhsElemTy != rhsElemTy) {
    return emitOpError() << "Expect the same element types for AddOp";
  }

  // check types are boradcastable
  if (lhsTensorTy && rhsTensorTy) {
    if (verifyBroadcastCompatibility(lhsTensorTy, rhsTensorTy)) {
      return success();
    } else {
      return this->emitOpError() << "Failed broadcast shapes.";
    }
  } else {
    return success();
  }
}

LogicalResult mix::SubOp::verify() {
  /*
verify:
  - element types are the same: refuse i32 - f32
    - both are Tensor: boardcastable and element
    - one is Tensor, another is Scale: success
    - both are Scale: success
*/
  auto lhsTy = this->getLhs().getType();
  auto rhsTy = this->getRhs().getType();
  auto lhsTensorTy = lhsTy.dyn_cast<RankedTensorType>();
  auto rhsTensorTy = rhsTy.dyn_cast<RankedTensorType>();
  // check element types.
  Type lhsElemTy = lhsTensorTy ? lhsTensorTy.getElementType() : lhsTy;
  Type rhsElemTy = rhsTensorTy ? rhsTensorTy.getElementType() : rhsTy;
  if (lhsElemTy != rhsElemTy) {
    return emitOpError() << "Expect the same element types for SubOp";
  }

  // check types are boradcastable
  if (lhsTensorTy && rhsTensorTy) {
    if (verifyBroadcastCompatibility(lhsTensorTy, rhsTensorTy)) {
      return success();
    } else {
      return this->emitOpError() << "Failed broadcast shapes.";
    }
  } else {
    return success();
  }
}

LogicalResult mix::MulOp::verify() {
  /*
verify:
  - element types are the same: refuse i32 * f32
    - both are Tensor: boardcastable and element
    - one is Tensor, another is Scale: success
    - both are Scale: success
*/
  auto lhsTy = this->getLhs().getType();
  auto rhsTy = this->getRhs().getType();
  auto lhsTensorTy = lhsTy.dyn_cast<RankedTensorType>();
  auto rhsTensorTy = rhsTy.dyn_cast<RankedTensorType>();
  // check element types.
  Type lhsElemTy = lhsTensorTy ? lhsTensorTy.getElementType() : lhsTy;
  Type rhsElemTy = rhsTensorTy ? rhsTensorTy.getElementType() : rhsTy;
  if (lhsElemTy != rhsElemTy) {
    return emitOpError() << "Expect the same element types for MulOp";
  }

  // check types are boradcastable
  if (lhsTensorTy && rhsTensorTy) {
    if (verifyBroadcastCompatibility(lhsTensorTy, rhsTensorTy)) {
      return success();
    } else {
      return this->emitOpError() << "Failed broadcast shapes.";
    }
  } else {
    return success();
  }
}

LogicalResult mix::DivOp::verify() {
  /*
verify:
  - element types are the same: refuse i32 / f32
    - both are Tensor: boardcastable and element
    - one is Tensor, another is Scale: success
    - both are Scale: success
*/
  auto lhsTy = this->getLhs().getType();
  auto rhsTy = this->getRhs().getType();
  auto lhsTensorTy = lhsTy.dyn_cast<RankedTensorType>();
  auto rhsTensorTy = rhsTy.dyn_cast<RankedTensorType>();
  // check element types.
  Type lhsElemTy = lhsTensorTy ? lhsTensorTy.getElementType() : lhsTy;
  Type rhsElemTy = rhsTensorTy ? rhsTensorTy.getElementType() : rhsTy;
  if (lhsElemTy != rhsElemTy) {
    return emitOpError() << "Expect the same element types for DivOp";
  }

  // check types are boradcastable
  if (lhsTensorTy && rhsTensorTy) {
    if (verifyBroadcastCompatibility(lhsTensorTy, rhsTensorTy)) {
      return success();
    } else {
      return this->emitOpError() << "Failed broadcast shapes.";
    }
  } else {
    return success();
  }
}

SmallVector<int64_t> inferBroadcastShape(ArrayRef<int64_t> lhsShape,
                                         ArrayRef<int64_t> rhsShape) {
  SmallVector<int64_t> resultShape;
  int lhsRank = lhsShape.size(), rhsRank = rhsShape.size();
  int maxRank = std::max(lhsRank, rhsRank);

  for (int i = 0; i < maxRank; ++i) {
    int64_t lhsDim = i < lhsRank ? lhsShape[lhsRank - 1 - i] : 1;
    int64_t rhsDim = i < rhsRank ? rhsShape[rhsRank - 1 - i] : 1;

    if (lhsDim == rhsDim || lhsDim == 1 || rhsDim == 1) {
      if (lhsDim == ShapedType::kDynamic || rhsDim == ShapedType::kDynamic) {
        resultShape.push_back(ShapedType::kDynamic);
      } else {
        resultShape.push_back(std::max(lhsDim, rhsDim));
      }
    } else {
      llvm_unreachable("Verify passed but got error when broadcasting");
    }
  }

  // Reverse to match the correct order
  std::reverse(resultShape.begin(), resultShape.end());
  return resultShape;
}

LogicalResult inferBinElementwiseOpReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {

  /*
  infer:
    - either is tensor: boardcast
    - both are scale : lhsTy
  */
  auto lhsTy = operands[0].getType();
  auto rhsTy = operands[1].getType();
  auto lhsTensorTy = lhsTy.dyn_cast<RankedTensorType>();
  auto rhsTensorTy = rhsTy.dyn_cast<RankedTensorType>();

  if (!lhsTensorTy && !rhsTensorTy) {
    inferredReturnTypes.push_back(lhsTy);
    return success();
  } else if (lhsTensorTy && !rhsTensorTy) {
    inferredReturnTypes.push_back(lhsTensorTy);
    return success();
  } else if (!lhsTensorTy && rhsTensorTy) {
    inferredReturnTypes.push_back(rhsTensorTy);
    return success();
  } else { // both are tensor
    auto resultShape =
        inferBroadcastShape(lhsTensorTy.getShape(), rhsTensorTy.getShape());
    auto resultType =
        RankedTensorType::get(resultShape, lhsTensorTy.getElementType());
    inferredReturnTypes.push_back(resultType);
    return success();
  }
}

LogicalResult mix::AddOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferBinElementwiseOpReturnTypes(context, location, operands,
                                          attributes, properties, regions,
                                          inferredReturnTypes);
}

LogicalResult mix::SubOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferBinElementwiseOpReturnTypes(context, location, operands,
                                          attributes, properties, regions,
                                          inferredReturnTypes);
}

LogicalResult mix::MulOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferBinElementwiseOpReturnTypes(context, location, operands,
                                          attributes, properties, regions,
                                          inferredReturnTypes);
}

LogicalResult mix::DivOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  return inferBinElementwiseOpReturnTypes(context, location, operands,
                                          attributes, properties, regions,
                                          inferredReturnTypes);
}

LogicalResult mix::MatMulOp::verify() {
  auto lhs = this->getLhs();
  auto rhs = this->getRhs();
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto lhsRank = lhsShape.size();
  auto rhsRank = rhsShape.size();
  auto lhsElemTy = lhsType.getElementType();
  auto rhsElemTy = rhsType.getElementType();
  if (lhsElemTy != rhsElemTy) {
    return this->emitOpError() << "Types mismatch.";
  }
  if (lhsRank != rhsRank || lhsRank != 2 || rhsRank != 2) {
    return this->emitOpError() << "Unexpected ranks.";
  }
  if (lhsShape[1] != rhsShape[0]) {
    return this->emitError() << "Unexpect shapes.";
  }
  return success();
}

LogicalResult mix::MatMulOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto lhsTy = operands[0].getType();
  auto rhsTy = operands[1].getType();
  auto lhsTensorTy = lhsTy.dyn_cast<RankedTensorType>();
  auto rhsTensorTy = rhsTy.dyn_cast<RankedTensorType>();
  auto lhsShape = lhsTensorTy.getShape();
  auto rhsShape = rhsTensorTy.getShape();
  auto elementTy = lhsTensorTy.getElementType();
  ArrayRef<int64_t> resultShape{lhsShape[0], rhsShape[1]};
  auto resultType = RankedTensorType::get(resultShape, elementTy);
  inferredReturnTypes.push_back(resultType);
  return success();
}

LogicalResult mix::NegOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = operands[0].getType();
  inferredReturnTypes.push_back(inputType);
  return success();
}

LogicalResult mix::ExpOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = operands[0].getType();
  inferredReturnTypes.push_back(inputType);
  return success();
}

#define GET_OP_CLASSES
#include "mix/mixOps.cpp.inc"
