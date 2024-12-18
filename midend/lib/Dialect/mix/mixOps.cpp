#include "mix/mixOps.h"
#include "mix/mixDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
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

    if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1 &&
        lhsDim != ShapedType::kDynamic && rhsDim != ShapedType::kDynamic) {
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

LogicalResult mix::PowOp::verify() {

  /*
verify:
  - element types are the same: refuse i32 / f32
    - both are Tensor: boardcastable and element
    - one is Tensor, another is Scale: success
    - both are Scale: success
*/
  auto lhsTy = this->getInput().getType();
  auto rhsTy = this->getExponent().getType();
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

    if (lhsDim == rhsDim || lhsDim == 1 || rhsDim == 1 ||
        lhsDim == ShapedType::kDynamic || rhsDim == ShapedType::kDynamic) {
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

LogicalResult mix::PowOp::inferReturnTypes(
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

LogicalResult mix::MeanOp::verify() {
  auto inputShape = this->getInput().getType().getShape();
  auto inputRank = inputShape.size();
  auto dimsAttr = this->getDims();
  auto dimArray = dimsAttr.getValue();
  SmallVector<int64_t> dims;
  for (auto attr : dimArray) {
    if (auto dimAttr = attr.dyn_cast<IntegerAttr>()) {
      auto dim = dimAttr.getInt();
      if (size_t(dim) >= inputRank) {
        return this->emitError() << "Unexpected dim value: " << dim << ".";
      }
      dims.push_back(dim);
    } else {
      return this->emitError() << "Unexpected dim attribute type.";
    }
  }
  for (size_t i = 0; i < dims.size(); i++) {
    auto dim = dims[i];
    for (size_t j = i + 1; j < dims.size(); j++) {
      auto another = dims[j];
      if (dim == another) {
        return this->emitError() << "Repetitive dimensions in meanOp.";
      }
    }
  }
  return success();
}

LogicalResult mix::MeanOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    MeanOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = adaptor.getInput().getType().dyn_cast<RankedTensorType>();
  auto inputShape = inputType.getShape();
  auto inputElementType = inputType.getElementType();
  auto dimsAttr = adaptor.getDimsAttr();
  auto dimArr = dimsAttr.getValue();

  auto keepDimAttr = adaptor.getKeepDimAttr();
  auto keepDim = keepDimAttr.getValue();

  SmallVector<int64_t> outputShape(inputShape);

  for (auto attr : dimArr) {
    auto dimAttr = attr.dyn_cast<IntegerAttr>();
    auto dim = dimAttr.getInt();
    if (keepDim) {
      outputShape[dim] = 1;
    } else {
      outputShape.erase(outputShape.begin() + dim);
    }
  }
  auto outputType = RankedTensorType::get(outputShape, inputElementType);
  inferredReturnTypes.push_back(outputType);
  return success();
}

LogicalResult mix::RsqrtOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

LogicalResult mix::ReduceSumOp::verify() {
  auto inputType = this->getInput().getType();
  auto inputShape = inputType.getShape();
  auto inputRank = inputShape.size();
  auto axis = this->getAxis();
  if (size_t(axis) >= inputRank) {
    return this->emitOpError("Unexpected axis.");
  }
  return success();
}

LogicalResult mix::ReduceSumOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ReduceSumOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = adaptor.getInput().getType().dyn_cast<RankedTensorType>();
  auto elementTy = inputType.getElementType();
  auto inputShape = inputType.getShape();
  auto axis = adaptor.getAxis();
  SmallVector<int64_t> outputShape(inputShape);
  outputShape[axis] = 1;
  auto outputType = RankedTensorType::get(outputShape, elementTy);
  inferredReturnTypes.push_back(outputType);
  return success();
}

#define GET_OP_CLASSES
#include "mix/mixOps.cpp.inc"
