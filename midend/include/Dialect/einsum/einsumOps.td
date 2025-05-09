#ifndef EINSUM_OPS_TD
#define EINSUM_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"

include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

include "einsumDialect.td"
include "einsumTypes.td"
include "einsumEnums.td"



def MapOp : EINSUM_Op<"map", [Pure]> {
    let summary = "Self-Attention computation for input tensors";
    let description = [{
        Computes the self-attention mechanism, which includes key, query, and value projections, 
        followed by scaled dot-product attention and optional output projection.
    }];
    let arguments = (ins
        einsumTensor: $lhs, 
        einsumTensor: $rhs,       
        AffineMapArrayAttr:$indexing_maps,
        ArrayAttr: $target_ranks,
        Builtin_IntegerSetAttr: $constraints,
        BinaryFnAttr: $computeOp
    );
    let results = (outs
        einsumTensor: $result_tensor      
    );

}


def ReducepOp : EINSUM_Op<"reduce", [Pure]> {
    let summary = "Self-Attention computation for input tensors";
    let description = [{
        Computes the self-attention mechanism, which includes key, query, and value projections, 
        followed by scaled dot-product attention and optional output projection.
    }];
    let arguments = (ins
        einsumTensor: $input,      
        AffineMapArrayAttr:$indexing_maps,
        ArrayAttr: $target_ranks,
        Builtin_IntegerSetAttr: $constraints,
        BinaryFnAttr: $computeOp
    );
    let results = (outs
        einsumTensor: $result_tensor      
    );

}



def PopulateOp : EINSUM_Op<"populate", [Pure]> {
    let summary = "Self-Attention computation for input tensors";
    let description = [{
        Computes the self-attention mechanism, which includes key, query, and value projections, 
        followed by scaled dot-product attention and optional output projection.
    }];
    let arguments = (ins
        einsumTensor: $lhs, 
        einsumTensor: $rhs,       
        AffineMapArrayAttr:$indexing_maps,
        ArrayAttr: $target_ranks,
        Builtin_IntegerSetAttr: $constraints,
        BinaryFnAttr: $computeOp
    );
    let results = (outs
        einsumTensor: $result_tensor      
    );

}

def UnaryOp : EINSUM_Op<"unary", [Pure]> {
    let summary = "Self-Attention computation for input tensors";
    let description = [{
        Computes the self-attention mechanism, which includes key, query, and value projections, 
        followed by scaled dot-product attention and optional output projection.
    }];
    let arguments = (ins
        einsumTensor: $input,
        UnaryFnAttr: $computeOp
    );
    let results = (outs
        einsumTensor: $result_tensor      
    );

}

def CascadeOp : EINSUM_Op<"cascade", [Pure]> {
    let summary = "Self-Attention computation for input tensors";
    let description = [{
        Computes the self-attention mechanism, which includes key, query, and value projections, 
        followed by scaled dot-product attention and optional output projection.
    }];
    let arguments = (ins
        Variadic<einsumTensor>:$inputs,
        Variadic<AnyShaped>:$outputs
    );
    let results = (outs
        Variadic<einsumTensor>:$result_tensors
    );

}

def IterationOp: EINSUM_Op<"iteration", [Pure]> {
    let summary = "Self-Attention computation for input tensors";
    let description = [{
        Computes the self-attention mechanism, which includes key, query, and value projections, 
        followed by scaled dot-product attention and optional output projection.
    }];
    let arguments = (ins
        Variadic<einsumTensor>:$inputs,
        Variadic<AnyShaped>:$outputs
    );
    let results = (outs
        Variadic<einsumTensor>:$result_tensors
    );


}


#endif // EINSUM_OPS_TD