#ifndef RVV_DIALECT_TD
#define RVV_DIALECT_TD

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/PatternBase.td"
include "mlir/Dialect/Vector/IR/VectorAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMAttrDefs.td"

//===----------------------------------------------------------------------===//
// Scalable Vector Type
//===----------------------------------------------------------------------===//

class ScalableVector_Type<Dialect dialect,string name> 
    : TypeDef<dialect, name> {
  let mnemonic = "vector";

  let summary = "Scalable vector type";
}

//===----------------------------------------------------------------------===//
// RVV dialect definition
//===----------------------------------------------------------------------===//

def RVV_Dialect : Dialect {
  let name = "rvv";
  let cppNamespace = "::rvv";
  let summary = "Basic dialect to target RISC-V Vector extension";
  let description = [{
    RISC-V vector extension (RVV) is the vector instruction set with scalable
    vector types, and the RVV instructions are vector length agnostic (VLA).
    For more details about RVV, please see the 
    [RVV specification](https://github.com/riscv/riscv-v-spec).
    This dialect contains the definitions of RVV operations and RVV 
    intrinsic operations. The former is used to interoperate with higher-level 
    dialects, and the latter is responsible for mapping to LLVM IR intrinsic.
  }];
}

class RVV_Op<string mnemonic, list<Trait> traits = []> :
  Op<RVV_Dialect, mnemonic, traits> {}

#endif // RVV_DIALECT_TD