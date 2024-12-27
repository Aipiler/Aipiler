#ifndef MIX_DIALECT_TD
#define MIX_DIALECT_TD

include "mlir/IR/OpBase.td"

def MIX_Dialect : Dialect {
    let name = "mix";
    let summary = "The Mix Dialect.";
    let description = [{
        Mix operations are defined in the Mix Dialect!
    }];
    let cppNamespace = "::mix";
}

class MIX_Op<string mnemonic, list<Trait> traits = []> :
    Op<MIX_Dialect, mnemonic, traits>;

#endif // MIX_DIALECT_TD