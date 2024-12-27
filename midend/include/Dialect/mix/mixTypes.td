#ifndef MIX_TYPE_TD
#define MIX_TYPE_TD

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mixDialect.td"

class mixType<string typename, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MIX_Dialect, typename, traits> {
    let mnemonic = typeMnemonic;
}

def mixInt : AnyTypeOf<[AnyUnsignedInteger, AnySignlessInteger]>;
def mixNumber : AnyTypeOf<[mixInt, AnyFloat]>;
def mixTensor : RankedTensorOf<[mixNumber]>;  
def mixData :AnyTypeOf<[mixTensor, mixNumber]>;


#endif // MIX_TYPE_TD