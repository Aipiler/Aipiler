#ifndef EINSUM_DIALECT_TD
#define EINSUM_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "einsumEnums.td"

def EINSUM_Dialect : Dialect {
    let name = "einsum";
    let summary = "The Einsum Dialect.";
    let description = [{
        einsum operations are defined in the Einsum Dialect!
    }];
    let cppNamespace = "::einsum";
}

class EINSUM_Op<string mnemonic, list<Trait> traits = []> :
    Op<EINSUM_Dialect, mnemonic, traits>;

// Define the function attribute enums matching the OpDSL functions.
def UnaryFnAttr : EnumAttr<EINSUM_Dialect, UnaryFn, "unary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}
def BinaryFnAttr : EnumAttr<EINSUM_Dialect, BinaryFn, "binary_fn"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // EINSUM_DIALECT_TD