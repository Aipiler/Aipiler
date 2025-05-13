#include "einsum/einsumTypes.h"
#include "einsum/einsumDialect.h"
#include "einsum/einsumOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/TypeSize.h"
#include <optional>

using namespace mlir;
using namespace einsum;

#define GET_TYPEDEF_CLASSES
#include "einsum/einsumTypes.cpp.inc"

// void MIXDialect::registerTypes() {
//     addTypes<
// #define GET_TYPEDEF_LIST
// #include "mix/mixTypes.cpp.inc"
//       >();
// }