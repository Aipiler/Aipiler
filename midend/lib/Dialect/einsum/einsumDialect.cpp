#include "einsum/einsumDialect.h"
#include "einsum/einsumOps.h"
#include "einsum/einsumTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace einsum;

#include "einsum/einsumDialect.cpp.inc"

void EINSUMDialect::initialize() {
  // registerTypes();

  addOperations<
#define GET_OP_LIST
#include "einsum/einsumOps.cpp.inc"
      >();
}