#ifndef MIDEND_UTILS_LOAD_PYTORCH_MODEL_H
#define MIDEND_UTILS_LOAD_PYTORCH_MODEL_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>

namespace mix {
namespace utils {

// load model weight and append to theModule attribute
void load_model(const std::string model_path, mlir::ModuleOp &theModule,
                mlir::OpBuilder &builder, mlir::Type dtype);

// load model weight from files, and append to theModule attribute
void load_model(const std::vector<std::string> model_paths,
                mlir::ModuleOp &theModule, mlir::OpBuilder &builder,
                mlir::Type dtype);

// load model weight from weight and initialize pointer
// used for load dynamicly
/*
pseudo:
for loc, memory_block in param_and_loc:
  data = model_path.load(loc)
  memory_block.write(data)
*/
template <typename T>
void load_model(const std::vector<std::string> model_paths,
                std::map<std::string, T *> param_and_loc);

void load_model_f16(const std::vector<std::string> *model_paths,
                    std::map<std::string, int16_t *> *param_and_loc);
} // namespace utils
} // namespace mix

#endif