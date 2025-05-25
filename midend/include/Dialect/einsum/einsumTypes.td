#ifndef EINSUM_TYPE_TD
#define EINSUM_TYPE_TD

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "einsumDialect.td"

class einsumType<string typename, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<EINSUM_Dialect, typename, traits> {
    let mnemonic = typeMnemonic;
}

def einsumInt : AnyTypeOf<[AnyUnsignedInteger, AnySignlessInteger]>;
def einsumNumber : AnyTypeOf<[einsumInt, AnyFloat]>;
def einsumTensor : RankedTensorOf<[einsumNumber]>;





#endif // EINSUM_TYPE_TD