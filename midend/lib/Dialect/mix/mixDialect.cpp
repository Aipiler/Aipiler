#include "mix/mixDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mix/mixOps.h"
#include "mix/mixTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mix;

#include "mix/mixDialect.cpp.inc"

void MIXDialect::initialize() {
  // registerTypes();

  addOperations<
#define GET_OP_LIST
#include "mix/mixOps.cpp.inc"
      >();
}