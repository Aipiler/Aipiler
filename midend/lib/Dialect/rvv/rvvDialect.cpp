//===- RVVDialect.cpp - MLIR RVV dialect implementation -------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// This file implements the RVV dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/TypeSwitch.h"

#include "rvv/rvvDialect.h"

using namespace mlir;

#include "rvv/rvvDialect.cpp.inc"

#define GET_OP_CLASSES
#include "rvv/rvvOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "rvv/rvvTypes.cpp.inc"

void rvv::RVVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "rvv/rvvOps.cpp.inc"
      >();
}
