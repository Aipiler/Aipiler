#ifndef RVV_TYPE_TD
#define RVV_TYPE_TD

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "rvvDialect.td"

class rvvType<string typename, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RVV_Dialect, typename, traits> {
    let mnemonic = typeMnemonic;
}

#endif // RVV_TYPE_TD