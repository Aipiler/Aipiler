#include "mix/mixOps.h"
#include "mix/mixDialect.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <optional>

using namespace mlir;

// 单目运算符，不改变shape

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

LogicalResult mix::RsqrtOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(operands[0].getType());
  return success();
}

LogicalResult mix::CosOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    CosOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(adaptor.getInput().getType());
  return success();
}

LogicalResult mix::SinOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    SinOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(adaptor.getInput().getType());
  return success();
}

LogicalResult mix::ReciprocalOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ReciprocalOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(adaptor.getInput().getType());
  return success();
}

LogicalResult mix::ConvertOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ConvertOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  // 获取输入类型
  auto inputType = adaptor.getValue().getType();

  // 获取目标元素类型
  auto elementType = adaptor.getElementTy();
  if (!mlir::isa<Type>(elementType)) {
    return emitOptionalError(location,
                             "element_ty attribute must be a valid type");
  }

  // 处理 RankedTensorType 输入
  if (auto rankedTensorType = dyn_cast<RankedTensorType>(inputType)) {
    // 使用输入张量的形状，创建新的类型
    auto resultType =
        RankedTensorType::get(rankedTensorType.getShape(), elementType);
    inferredReturnTypes.push_back(resultType);
    return success();
  }

  // 处理 UnrankedTensorType 输入
  if (auto unrankedTensorType = dyn_cast<UnrankedTensorType>(inputType)) {
    // 仅替换元素类型，不关心形状
    auto resultType = UnrankedTensorType::get(elementType);
    inferredReturnTypes.push_back(resultType);
    return success();
  }

  // 处理其他类型（例如标量类型）
  if (mlir::isa<ShapedType>(inputType)) {
    auto shapedType = mlir::cast<ShapedType>(inputType);
    if (shapedType.hasRank()) {
      // 如果有 rank，但未显式定义为 RankedTensorType
      auto resultType =
          RankedTensorType::get(shapedType.getShape(), elementType);
      inferredReturnTypes.push_back(resultType);
      return success();
    } else {
      // 如果没有 rank
      auto resultType = UnrankedTensorType::get(elementType);
      inferredReturnTypes.push_back(resultType);
      return success();
    }
  }

  // 如果输入是标量类型
  if (inputType.isa<TensorType>() || inputType.isa<Type>()) {
    inferredReturnTypes.push_back(inputType);
    return success();
  }

  // 如果输入是无法识别的类型
  return emitOptionalError(location,
                           "unsupported input type for ConvertOp: ", inputType);
}

LogicalResult mix::SoftmaxOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    SoftmaxOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  // Extract the input type
  auto inputType = adaptor.getInput().getType().dyn_cast<RankedTensorType>();

  // Ensure the input is a ranked tensor
  if (!inputType) {
    return emitOptionalError(location, "Input must be a ranked tensor.");
  }

  // Extract shape and element type of the input
  auto inputShape = inputType.getShape();
  auto elementType = inputType.getElementType();

  // Extract and validate the axis
  auto axis = adaptor.getAxis();

  // Handle negative axis values (convert to positive index)
  if (axis < 0) {
    axis += inputShape.size();
  }

  // Ensure the axis is valid after adjustment
  if (axis < 0 || axis >= static_cast<int64_t>(inputShape.size())) {
    char errorMessage[256];
    snprintf(errorMessage, sizeof(errorMessage),
             "Axis value %d is out of bounds for input tensor with rank %ld.",
             axis, static_cast<long>(inputShape.size()));
    return emitOptionalError(location, errorMessage);
  }

  // The output type is the same as the input type
  inferredReturnTypes.push_back(inputType);
  return success();
}

// element wise Op verify

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

template <typename T> LogicalResult verifyElementWiseOp(T op) {
  /*
verify:
  - accept different element types
    - both are Tensor: boardcastable and element
    - one is Tensor, another is Scale: success
    - both are Scale: success
*/
  auto lhsTy = op.getLhs().getType();
  auto rhsTy = op.getRhs().getType();
  auto lhsTensorTy = mlir::dyn_cast<RankedTensorType>(lhsTy);
  auto rhsTensorTy = mlir::dyn_cast<RankedTensorType>(rhsTy);

  // check types are boradcastable
  if (lhsTensorTy && rhsTensorTy) {
    if (verifyBroadcastCompatibility(lhsTensorTy, rhsTensorTy)) {
      return success();
    } else {
      op->emitOpError() << "Failed broadcast shapes.";
      return failure();
    }
  } else {
    return success();
  }
}

LogicalResult mix::AddOp::verify() { return verifyElementWiseOp(*this); }

LogicalResult mix::SubOp::verify() { return verifyElementWiseOp(*this); }

LogicalResult mix::MulOp::verify() { return verifyElementWiseOp(*this); }

LogicalResult mix::DivOp::verify() { return verifyElementWiseOp(*this); }

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
  auto lhsTensorTy = dyn_cast<RankedTensorType>(lhsTy);
  auto rhsTensorTy = dyn_cast<RankedTensorType>(rhsTy);

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

LogicalResult
inferBinElementwiseOpReturnTypes(ValueRange operands,
                                 SmallVectorImpl<Type> &inferredReturnTypes) {

  /*
  infer:
    - either is tensor: boardcast
    - both are scale : lhsTy
  */
  auto lhsTy = operands[0].getType();
  auto rhsTy = operands[1].getType();
  auto lhsTensorTy = dyn_cast<RankedTensorType>(lhsTy);
  auto rhsTensorTy = dyn_cast<RankedTensorType>(rhsTy);

  // TODO: infer type for different element types.
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

// element wise shape inference
#define ELEMENTWISE_SHAPE_INFER(OP)                                            \
  LogicalResult OP::inferReturnTypes(                                          \
      MLIRContext *context, std::optional<::mlir::Location> location,          \
      ValueRange operands, DictionaryAttr attributes,                          \
      OpaqueProperties properties, RegionRange regions,                        \
      SmallVectorImpl<Type> &inferredReturnTypes) {                            \
    return inferBinElementwiseOpReturnTypes(operands, inferredReturnTypes);    \
  }

ELEMENTWISE_SHAPE_INFER(mix::AddOp)
ELEMENTWISE_SHAPE_INFER(mix::SubOp)
ELEMENTWISE_SHAPE_INFER(mix::MulOp)
ELEMENTWISE_SHAPE_INFER(mix::DivOp)
ELEMENTWISE_SHAPE_INFER(mix::PowOp)

// 特殊的shape inference 算子，需要改变shape

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
    this->emitOpError() << "Types mismatch.";
    return failure();
  }
  if (lhsRank != rhsRank || lhsRank != 2 || rhsRank != 2) {
    this->emitOpError() << "Unexpected ranks.";
    return failure();
  }
  if (lhsShape[1] != rhsShape[0]) {
    this->emitError() << "Unexpect shapes.";
    return failure();
  }
  return success();
}

LogicalResult mix::MatMulOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto lhsTy = operands[0].getType();
  auto rhsTy = operands[1].getType();
  auto lhsTensorTy = mlir::dyn_cast<RankedTensorType>(lhsTy);
  auto rhsTensorTy = mlir::dyn_cast<RankedTensorType>(rhsTy);
  auto lhsShape = lhsTensorTy.getShape();
  auto rhsShape = rhsTensorTy.getShape();
  auto elementTy = lhsTensorTy.getElementType();
  SmallVector<int64_t> resultShape{lhsShape[0], rhsShape[1]};
  auto resultType = RankedTensorType::get(resultShape, elementTy);
  inferredReturnTypes.push_back(resultType);
  return success();
}

LogicalResult mix::BatchMatMulOp::verify() {
  auto lhs = this->getLhs();
  auto rhs = this->getRhs();
  auto lhsType = lhs.getType().cast<RankedTensorType>();
  auto rhsType = rhs.getType().cast<RankedTensorType>();

  // Ensure both tensors are ranked
  if (!lhsType || !rhsType) {
    this->emitOpError() << "Both inputs must be ranked tensors.";
    return failure();
  }

  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto lhsRank = lhsShape.size();
  auto rhsRank = rhsShape.size();

  // Ensure both tensors have the same rank
  if (lhsRank != rhsRank) {
    this->emitOpError()
        << "Input tensors must have the same rank, got lhs rank " << lhsRank
        << " and rhs rank " << rhsRank << ".";
    return failure();
  }

  // Ensure the rank is at least 3 (batch dimension + 2D matrices)
  if (lhsRank < 3) {
    this->emitOpError()
        << "Inputs must have rank >= 3 for batch matrix multiplication.";
    return failure();
  }

  // Ensure element types match
  auto lhsElemTy = lhsType.getElementType();
  auto rhsElemTy = rhsType.getElementType();
  if (lhsElemTy != rhsElemTy) {
    this->emitOpError()
        << "Element types of inputs must match, got lhs element type "
        << lhsElemTy << " and rhs element type " << rhsElemTy << ".";
    return failure();
  }

  // Check matrix multiplication dimensions
  for (int64_t i = 0; i < lhsRank - 2; ++i) {
    if (lhsShape[i] != rhsShape[i] && lhsShape[i] != ShapedType::kDynamic &&
        rhsShape[i] != ShapedType::kDynamic) {
      this->emitOpError()
          << "Batch dimensions must match for inputs, mismatch at dimension "
          << i << ".";
      return failure();
    }
  }

  // Check the inner matrix dimensions
  if (lhsShape[lhsRank - 1] != rhsShape[rhsRank - 2] &&
      lhsShape[lhsRank - 1] != ShapedType::kDynamic &&
      rhsShape[rhsRank - 2] != ShapedType::kDynamic) {
    this->emitOpError()
        << "Inner dimensions of inputs do not match for matrix multiplication, "
        << "got lhs inner dimension " << lhsShape[lhsRank - 1]
        << " and rhs inner dimension " << rhsShape[rhsRank - 2] << ".";
    return failure();
  }

  return success();
}

LogicalResult mix::BatchMatMulOp::inferReturnTypes(
    MLIRContext *context, std::optional<::mlir::Location> location,
    BatchMatMulOp::Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Extract the input tensor types
  auto lhsTy = adaptor.getLhs().getType().dyn_cast<RankedTensorType>();
  auto rhsTy = adaptor.getRhs().getType().dyn_cast<RankedTensorType>();

  // Ensure both inputs are ranked tensors
  if (!lhsTy || !rhsTy) {
    return emitOptionalError(location, "Both inputs must be ranked tensors.");
  }

  // Extract shapes and element types
  auto lhsShape = lhsTy.getShape();
  auto rhsShape = rhsTy.getShape();
  auto elementTy = lhsTy.getElementType();

  // Ensure ranks match
  if (lhsShape.size() != rhsShape.size()) {

    return emitOptionalError(location,
                             "Input tensors must have the same rank.");
  }

  // Compute result shape
  SmallVector<int64_t> resultShape;
  auto rank = lhsShape.size();

  // Batch dimensions
  for (int64_t i = 0; i < rank - 2; ++i) {
    if (lhsShape[i] == rhsShape[i] || lhsShape[i] == ShapedType::kDynamic ||
        rhsShape[i] == ShapedType::kDynamic) {
      resultShape.push_back(lhsShape[i] == rhsShape[i] ? lhsShape[i]
                                                       : ShapedType::kDynamic);
    } else {

      return emitOptionalError(location,
                               "Mismatch in batch dimensions at index %ld : "
                               "lhs has %ld, rhs has %ld.",
                               i, lhsShape[i], rhsShape[i]);
    }
  }

  // Matrix multiplication dimensions
  resultShape.push_back(lhsShape[rank - 2]);
  resultShape.push_back(rhsShape[rank - 1]);

  // Create result type
  auto resultType = RankedTensorType::get(resultShape, elementTy);
  inferredReturnTypes.push_back(resultType);

  return success();
}

LogicalResult mix::MeanOp::verify() {
  auto inputShape = this->getInput().getType().getShape();
  auto inputRank = inputShape.size();
  auto dimsAttr = this->getDims();
  auto dimArray = dimsAttr.getValue();
  SmallVector<int64_t> dims;
  for (auto attr : dimArray) {
    if (auto dimAttr = dyn_cast<IntegerAttr>(attr)) {
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
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto inputShape = inputType.getShape();
  auto inputElementType = inputType.getElementType();
  auto dimsAttr = adaptor.getDimsAttr();
  auto dimArr = dimsAttr.getValue();

  auto keepDimAttr = adaptor.getKeepDimAttr();
  auto keepDim = keepDimAttr.getValue();

  SmallVector<int64_t> outputShape(inputShape);

  for (auto attr : dimArr) {
    auto dimAttr = mlir::dyn_cast<IntegerAttr>(attr);
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

LogicalResult mix::ReshapeOp::verify() {
  // Get the input tensor
  auto input = getInput();
  if (!input) {
    return emitOpError("requires an input tensor");
  }

  // Check if input is a ranked tensor
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return emitOpError("input must be a ranked tensor");
  }

  // Get the target shape from attributes
  auto targetShape = getShape();

  // Calculate total elements in input
  int64_t inputElements = 1;
  for (auto dim : inputType.getShape()) {
    if (dim == ShapedType::kDynamic) {
      // If input has dynamic dimensions, skip element count verification
      return success();
    }
    inputElements *= dim;
  }

  // Calculate total elements in target shape
  int64_t outputElements = 1;
  int64_t dynamicIndex = -1; // Track the index of the -1 dimension (if any)
  for (int64_t i = 0; i < targetShape.size(); ++i) {
    auto dimAttr = targetShape[i].dyn_cast<IntegerAttr>();
    if (!dimAttr) {
      return emitOpError("target shape must contain only integer attributes");
    }

    auto dim = dimAttr.getInt();
    if (dim == -1) {
      if (dynamicIndex != -1) {
        // More than one -1 is invalid
        return emitOpError(
            "target shape can have at most one dynamic dimension (-1)");
      }
      dynamicIndex = i; // Record the index of the -1
    } else if (dim < 0) {
      return emitOpError("target shape dimensions must be non-negative or -1");
    } else {
      outputElements *= dim;
    }
  }

  // If there is a -1, calculate its value
  if (dynamicIndex != -1) {
    if (inputElements % outputElements != 0) {
      return emitOpError("input elements (")
             << inputElements
             << ") cannot be evenly divided by the fixed dimensions ("
             << outputElements << ")";
    }

    // Replace -1 with the inferred dimension size
    int64_t inferredDim = inputElements / outputElements;
    if (inferredDim <= 0) {
      return emitOpError("inferred dimension size must be positive, got ")
             << inferredDim;
    }
  } else if (inputElements != outputElements) {
    // If no -1 exists, verify that the element counts match
    return emitOpError("number of input elements (")
           << inputElements << ") does not match output elements ("
           << outputElements << ")";
  }

  return success();
}

LogicalResult mix::ReshapeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ReshapeOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the input tensor type
  auto inputType =
      mlir::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!inputType) {
    return emitOptionalError(location, "input must be a ranked tensor");
  }

  // Get the target shape from attributes
  auto targetShape = adaptor.getShape();

  SmallVector<int64_t> outputShape;
  int64_t inputElements = 1;
  for (auto dim : inputType.getShape()) {
    if (dim != ShapedType::kDynamic) {
      inputElements *= dim;
    }
  }

  int64_t outputElements = 1;
  int64_t dynamicIndex = -1;

  // Build the output shape and calculate inferred dimension
  for (size_t i = 0; i < targetShape.size(); ++i) {
    auto dimAttr = targetShape[i].dyn_cast<IntegerAttr>();
    if (!dimAttr) {
      return emitOptionalError(
          location, "target shape must contain only integer attributes");
    }

    int64_t dim = dimAttr.getInt();
    if (dim == -1) {
      if (dynamicIndex != -1) {
        return emitOptionalError(
            location,
            "target shape can have at most one dynamic dimension (-1)");
      }
      dynamicIndex = i;
      outputShape.push_back(ShapedType::kDynamic);
    } else {
      outputShape.push_back(dim);
      if (dim > 0) {
        outputElements *= dim;
      }
    }
  }

  // Infer the dynamic dimension size if -1 exists
  if (dynamicIndex != -1) {
    if (outputElements == 0 || inputElements % outputElements != 0) {
      char errorMessage[256];
      snprintf(errorMessage, sizeof(errorMessage),
               "input elements (%ld) cannot be evenly divided by the fixed "
               "dimensions (%ld)",
               inputElements, outputElements);
      return emitOptionalError(location, errorMessage);
    }

    int64_t inferredDim = inputElements / outputElements;
    if (inferredDim <= 0) {
      char errorMessage[256];
      snprintf(errorMessage, sizeof(errorMessage),
               "inferred dimension size must be positive, got %ld",
               inferredDim);
      return emitOptionalError(location, errorMessage);
    }
    outputShape[dynamicIndex] = inferredDim;
  }

  // Create output tensor type with inferred shape
  auto resultType =
      RankedTensorType::get(outputShape, inputType.getElementType());
  inferredReturnTypes.push_back(resultType);

  return success();
}

LogicalResult mix::ConcatOp::verify() {
  // Get the inputs (tensors) and the axis attribute.
  auto inputs = getInputs();
  auto axis = getAxis();

  // Ensure there are inputs.
  if (inputs.empty()) {
    return emitOpError("requires at least one input tensor");
  }

  // Ensure the axis is within the valid range.
  // int64_t axis = axisAttr.getInt();
  auto firstTensorType = mlir::dyn_cast<RankedTensorType>(inputs[0].getType());
  if (!firstTensorType) {
    return emitOpError("all inputs must be ranked tensors");
  }
  auto rank = firstTensorType.getRank();
  if (axis >= rank) {
    return emitOpError("axis must be in range [0, ") << rank - 1 << "]";
  }

  // Ensure all inputs have the same element type and compatible shapes.
  auto elementType = firstTensorType.getElementType();
  auto firstShape = firstTensorType.getShape();
  for (auto input : inputs) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(input.getType());
    if (!tensorType) {
      return emitOpError("all inputs must be ranked tensors");
    }

    // Check element type consistency.
    if (tensorType.getElementType() != elementType) {
      return emitOpError("all inputs must have the same element type");
    }

    // Check shape consistency (except along the concatenation axis).
    auto shape = tensorType.getShape();
    for (int64_t i = 0; i < rank; ++i) {
      if (i == axis)
        continue; // Skip the concatenation axis.
      if (shape[i] != firstShape[i]) {
        return emitOpError("all inputs must have the same shape except along "
                           "the concatenation axis");
      }
    }
  }
  return success();
}

LogicalResult mix::ConcatOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ConcatOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {

  // Get the axis attribute.
  auto axis = adaptor.getAxis();

  // Get the first tensor type.
  auto firstTensorType =
      mlir::dyn_cast<RankedTensorType>(adaptor.getInputs()[0].getType());
  if (!firstTensorType) {
    return emitOptionalError(location, "all inputs must be ranked tensors");
  }

  // Ensure the axis is within the valid range.
  int64_t rank = firstTensorType.getRank();
  if (axis < 0 || axis >= rank) {
    return emitOptionalError(location, "axis must be in range [0, %d]",
                             rank - 1);
  }

  // Compute the output shape.
  SmallVector<int64_t, 4> outputShape(firstTensorType.getShape().begin(),
                                      firstTensorType.getShape().end());
  outputShape[axis] = 0;
  for (auto operand : adaptor.getInputs()) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorType) {
      return emitOptionalError(location, "all inputs must be ranked tensors");
    }

    // Add up the dimension along the concatenation axis.
    outputShape[axis] += tensorType.getShape()[axis];
  }

  // Create the output tensor type.
  auto elementType = firstTensorType.getElementType();
  auto resultType = RankedTensorType::get(outputShape, elementType);
  inferredReturnTypes.push_back(resultType);

  return success();
}

LogicalResult mix::SliceOp::verify() {
  // Get the input tensor.
  auto input = getInputs();
  if (!input) {
    return emitOpError("requires an input tensor");
  }

  // Check if input is a ranked tensor.
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return emitOpError("input must be a ranked tensor");
  }

  // Get the attributes.
  int64_t dim = getDim();
  int64_t start = getStart();
  int64_t end = getEnd();
  int64_t step = getStep();

  // Ensure dim is within range.
  int64_t rank = inputType.getRank();
  if (dim < 0 || dim >= rank) {
    return emitOpError("dimension (dim) must be in range [0, ")
           << rank - 1 << "]";
  }

  // Get the size of the dimension being sliced.
  int64_t dimSize = inputType.getShape()[dim];
  if (dimSize == ShapedType::kDynamic) {
    return emitOpError("dimension size must be static for slicing");
  }

  // Check if start and end are within bounds.
  if (start < 0 || start >= dimSize) {
    return emitOpError("start index must be in range [0, ")
           << dimSize - 1 << "]";
  }

  if (end > dimSize) {
    end = dimSize;
  }

  if (end < 0 || end > dimSize) {
    return emitOpError("end index must be in range [0, ") << dimSize << "]";
  }

  // Ensure step is not zero.
  if (step == 0) {
    return emitOpError("step size must not be zero");
  }

  return success();
}

LogicalResult mix::SliceOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    SliceOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the input tensor
  auto inputType =
      mlir::dyn_cast<RankedTensorType>(adaptor.getInputs().getType());
  if (!inputType) {
    return emitOptionalError(location, "input must be a ranked tensor");
  }

  // Get slicing parameters
  int64_t dim = adaptor.getDim();
  int64_t start = adaptor.getStart();
  int64_t end = adaptor.getEnd();
  int64_t step = adaptor.getStep();

  // Ensure dim is within range
  int64_t rank = inputType.getRank();
  if (dim < 0 || dim >= rank) {
    return emitOptionalError(
        location, "dimension (dim) must be in range [0, %d]", rank - 1);
  }

  // Get the shape of the input tensor
  SmallVector<int64_t, 4> outputShape(inputType.getShape().begin(),
                                      inputType.getShape().end());
  int64_t dimSize = inputType.getShape()[dim];

  // Adjust the end value if it exceeds the dimension size
  if (dimSize != ShapedType::kDynamic && end > dimSize) {
    end = dimSize;
  }

  // Compute the sliced size for the dimension
  if (dimSize != ShapedType::kDynamic) {
    int64_t slicedSize = 0;
    if (step > 0) {
      slicedSize = std::max(int64_t(0), (end - start + step - 1) / step);
    } else {
      slicedSize = std::max(int64_t(0), (start - end - step - 1) / (-step));
    }
    outputShape[dim] = slicedSize;
  } else {
    // If dynamic, set as dynamic
    outputShape[dim] = ShapedType::kDynamic;
  }

  // Create the output tensor type
  auto resultType =
      RankedTensorType::get(outputShape, inputType.getElementType());
  inferredReturnTypes.push_back(resultType);

  return success();
}

LogicalResult mix::MaskedFillOp::verify() {
  // 获取输入tensor
  auto input = getInput();
  if (!input) {
    return emitOpError("requires an input tensor");
  }
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return emitOpError("input must be a ranked tensor");
  }

  // 获取掩码tensor
  auto mask = getMask();
  if (!mask) {
    return emitOpError("requires a mask tensor");
  }
  auto maskType = dyn_cast<RankedTensorType>(mask.getType());
  if (!maskType) {
    return emitOpError("mask must be a ranked tensor");
  }

  // 验证掩码的元素类型是否为bool
  if (!maskType.getElementType().isInteger(1)) {
    return emitOpError("mask must have boolean (i1) element type");
  }

  // 验证输入和掩码的shape是否匹配
  if (inputType.getRank() != maskType.getRank()) {
    return emitOpError("input and mask must have the same rank, but got: ")
           << inputType.getRank() << " vs " << maskType.getRank();
  }

  // 验证每个维度的大小是否匹配
  if (verifyBroadcastCompatibility(inputType, maskType)) {
    return success();
  } else {
    return this->emitOpError("Failed broadcast shapes.");
  }
}

LogicalResult mix::MaskedFillOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    MaskedFillOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  // 获取输入tensor类型
  auto inputType =
      mlir::dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!inputType) {
    return emitOptionalError(location, "input must be a ranked tensor");
  }

  // 获取掩码tensor类型
  auto maskType = mlir::dyn_cast<RankedTensorType>(adaptor.getMask().getType());
  if (!maskType) {
    return emitOptionalError(location, "mask must be a ranked tensor");
  }

  // 检查rank是否匹配
  if (inputType.getRank() != maskType.getRank()) {
    return emitOptionalError(
        location, "input and mask must have the same rank, but got: %d vs %d",
        inputType.getRank(), maskType.getRank());
  }

  // 输出shape与输入相同
  auto outputShape = llvm::to_vector<4>(inputType.getShape());

  // 创建输出tensor类型（保持与输入相同的shape和element type）
  auto resultType =
      RankedTensorType::get(outputShape, inputType.getElementType());
  inferredReturnTypes.push_back(resultType);

  return success();
}

LogicalResult mix::PermuteOp::verify() {
  // Get the input tensor
  auto input = getInput();
  if (!input) {
    return emitOpError("requires an input tensor");
  }

  // Check if input is a ranked tensor
  auto inputType = dyn_cast<ShapedType>(input.getType());
  if (!inputType) {
    return emitOpError("input must be a ranked tensor");
  }

  // Get the permutation dimensions
  auto permDims = getDims();
  SmallVector<int64_t> dims;
  for (auto dimAttr : permDims.getValue()) {
    auto dimIAttr = mlir::dyn_cast<IntegerAttr>(dimAttr);
    dims.push_back(dimIAttr.getInt());
  }
  auto rank = inputType.getRank();
  // Check if permutation dimensions match input rank
  if (dims.size() != rank) {
    return emitOpError("permutation dimensions must match input tensor rank");
  }

  // Check if permutation dimensions are unique
  SmallVector<bool, 8> seen(inputType.getRank(), false);
  for (auto dim : dims) {
    if (dim < 0 || dim >= inputType.getRank()) {
      return emitOpError("permutation dimension out of range");
    }
    if (seen[dim]) {
      return emitOpError("permutation dimensions must be unique");
    }
    seen[dim] = true;
  }

  return success();
}

LogicalResult mix::PermuteOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    PermuteOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the input tensor type
  auto inputType = mlir::dyn_cast<ShapedType>(adaptor.getInput().getType());
  if (!inputType) {
    return emitOptionalError(location, "input must be a ranked tensor");
  }

  // Get the permutation dimensions
  auto permDims = adaptor.getDims();
  SmallVector<int64_t> dims;
  for (auto dimAttr : permDims.getValue()) {
    auto dimIAttr = mlir::dyn_cast<IntegerAttr>(dimAttr);
    dims.push_back(dimIAttr.getInt());
  }

  // Create output shape by reordering input shape
  SmallVector<int64_t> outputShape;
  for (auto dim : dims) {
    outputShape.push_back(inputType.getDimSize(dim));
  }

  // Create output tensor type with permuted shape and same element type
  auto resultType =
      RankedTensorType::get(outputShape, inputType.getElementType());
  inferredReturnTypes.push_back(resultType);

  return success();
}

LogicalResult mix::TransposeOp::verify() {
  // Get the input tensor
  auto input = getInput();
  if (!input) {
    return emitOpError("requires an input tensor");
  }

  // Check if input is a ranked tensor
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return emitOpError("input must be a ranked tensor");
  }

  // Get dimensions to transpose
  int64_t dim1 = getDim1();
  int64_t dim2 = getDim2();

  // Verify that dimensions are within bounds
  int64_t rank = inputType.getRank();
  if (dim1 < 0 || dim1 >= rank || dim2 < 0 || dim2 >= rank) {
    return emitOpError("dimensions to transpose (")
           << dim1 << ", " << dim2 << ") must be within the rank (" << rank
           << ")";
  }

  return success();
}

LogicalResult mix::TransposeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    TransposeOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the input tensor type
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!inputType) {
    return emitOptionalError(location, "input must be a ranked tensor");
  }

  // Get dimensions to transpose
  int64_t dim1 = adaptor.getDim1();
  int64_t dim2 = adaptor.getDim2();

  // Verify that dimensions are within bounds
  int64_t rank = inputType.getRank();
  if (dim1 < 0 || dim1 >= rank || dim2 < 0 || dim2 >= rank) {
    return emitOptionalError(location, "dimensions to transpose (%dim1, %dim2) "
                                       "must be within the rank (%rank)");
  }

  // Generate the output shape
  SmallVector<int64_t> outputShape(inputType.getShape().begin(),
                                   inputType.getShape().end());
  std::swap(outputShape[dim1], outputShape[dim2]);

  // Create output tensor type
  auto resultType =
      RankedTensorType::get(outputShape, inputType.getElementType());
  inferredReturnTypes.push_back(resultType);

  return success();
}

LogicalResult mix::UnsqueezeOp::verify() {
  // Get the input tensor
  auto input = getInput();
  if (!input) {
    return emitOpError("requires an input tensor");
  }

  // Check if input is a ranked tensor
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    return emitOpError("input must be a ranked tensor");
  }

  // Get the axis
  int64_t axis = getAxis();

  // Verify that the axis is within the bounds of the new rank
  int64_t rank = inputType.getRank();
  if (axis < 0 || axis > rank) {
    return emitOpError("axis (")
           << axis << ") must be within [0, " << rank << "]";
  }

  return success();
}

LogicalResult mix::UnsqueezeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    UnsqueezeOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  // Get the input tensor type
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  if (!inputType) {
    return emitOptionalError(location, "input must be a ranked tensor");
  }

  // Get the axis
  int64_t axis = adaptor.getAxis();

  // Verify that the axis is within the bounds of the new rank
  int64_t rank = inputType.getRank();
  if (axis < 0 || axis > rank) {
    return emitOptionalError(location, "axis (%d) must be within [0, rank]",
                             axis);
  }

  // Generate the output shape
  SmallVector<int64_t> outputShape;
  outputShape.reserve(rank + 1);

  for (int64_t i = 0; i < rank + 1; ++i) {
    if (i == axis) {
      outputShape.push_back(1); // Insert the unsqueezed dimension
    } else {
      outputShape.push_back(inputType.getShape()[i - (i > axis ? 1 : 0)]);
    }
  }

  // Create output tensor type
  auto resultType =
      RankedTensorType::get(outputShape, inputType.getElementType());
  inferredReturnTypes.push_back(resultType);

  return success();
}

// Reduce 类型算子，只有一个input

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
  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto elementTy = inputType.getElementType();
  auto inputShape = inputType.getShape();
  auto axis = adaptor.getAxis();
  SmallVector<int64_t> outputShape(inputShape);
  outputShape[axis] = 1;
  auto outputType = RankedTensorType::get(outputShape, elementTy);
  inferredReturnTypes.push_back(outputType);
  return success();
}

LogicalResult mix::TanhOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    TanhOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = adaptor.getInput().getType();
  inferredReturnTypes.push_back(inputType);
  return success();
}

LogicalResult mix::LinearOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    LinearOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto input = adaptor.getInput();
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto shape = inputType.getShape();
  auto output_feature = adaptor.getOutFeature();
  SmallVector<int64_t> outputShape(shape);
  outputShape.back() = output_feature;
  auto returnType =
      RankedTensorType::get(outputShape, inputType.getElementType());
  inferredReturnTypes.push_back(returnType);
  return success();
}

LogicalResult mix::LinearOp::verify() {
  auto input = this->getInput();
  auto inputType = input.getType();
  auto shape = inputType.getShape();
  auto input_feature = this->getInFeature();
  if (shape.back() != input_feature) {
    return this->emitError() << "Unexpect input shape";
  }
  return success();
}

LogicalResult mix::EmbeddingOp::verify() {
  auto paddingIdx = this->getPaddingIdx();
  auto embeddingNum = this->getEmbeddingDim();
  if (paddingIdx.has_value()) {
    auto padIdxNum = paddingIdx.value();
    if (padIdxNum >= embeddingNum) {
      return this->emitOpError("Padding_idx must be within num_embeddings");
    }
  }
  return success();
}

LogicalResult mix::EmbeddingOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    EmbeddingOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {

  auto input = adaptor.getInput();
  auto opt_dtype = adaptor.getDtype();
  auto dtype = opt_dtype.has_value() ? opt_dtype.value()
                                     : mlir::FloatType::getF32(context);
  auto inputType = llvm::dyn_cast<RankedTensorType>(input.getType());
  auto shape = inputType.getShape();
  auto embeddingdim = adaptor.getEmbeddingDim();
  SmallVector<int64_t> outputShape(shape);
  outputShape.push_back(embeddingdim);
  auto returnType = RankedTensorType::get(outputShape, dtype);
  inferredReturnTypes.push_back(returnType);
  return success();
}

LogicalResult mix::GetItemOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    GetItemOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto value = adaptor.getValue();
  auto valueType = mlir::dyn_cast<RankedTensorType>(value.getType());
  auto valueElementType = valueType.getElementType();
  auto valueShape = valueType.getShape();
  auto embeddingDim = valueShape.back();
  auto indices = adaptor.getIndice();
  auto indiceType = indices.getType();
  Type returnType;
  if (auto integerType = dyn_cast<IntegerType>(indiceType)) {
    SmallVector<int64_t> outputShape;
    outputShape.push_back(embeddingDim);
    returnType = RankedTensorType::get(outputShape, valueElementType);
  } else if (auto tensorType = dyn_cast<RankedTensorType>(indiceType)) {
    SmallVector<int64_t> outputShape(valueShape);
    returnType = RankedTensorType::get(outputShape, valueElementType);
  }
  inferredReturnTypes.push_back(returnType);
  return success();
}

LogicalResult mix::GatherOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    GatherOp::Adaptor adaptor, SmallVectorImpl<Type> &inferredReturnTypes) {
  auto indices = adaptor.getIndices();
  auto indicesType = mlir::dyn_cast<RankedTensorType>(indices.getType());
  SmallVector<int64_t> indicesShape(indicesType.getShape());
  auto value = adaptor.getValues();
  auto valueType = mlir::dyn_cast<RankedTensorType>(value.getType());
  auto embeddingDim = valueType.getShape().back();
  indicesShape.push_back(embeddingDim);
  auto retType =
      RankedTensorType::get(indicesShape, valueType.getElementType());
  inferredReturnTypes.push_back(retType);
  return success();
}

#define GET_OP_CLASSES
#include "mix/mixOps.cpp.inc"
