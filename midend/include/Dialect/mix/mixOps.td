#ifndef MIX_OPS_TD
#define MIX_OPS_TD

include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td" 
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/BuiltinTypes.td"

include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mixDialect.td"
include "mixTypes.td"

// 组合算子

def MLPOp : MIX_Op<"module.mlp"> {
    let summary = "MultiLayerPerceptron";
    let description = [{
        Multi Layer Perceptron.
    }];
    let arguments = (ins 
        mixTensor: $hidden_states,
        mixTensor: $residual
    );
    let results = (outs
        mixTensor: $output
    );
}

def LinearOp : MIX_Op<"module.linear", [Pure]> {
    let summary = "Linear";
    let description = [{
        Linear
    }];
    let arguments = (ins
        mixTensor: $input,
        mixTensor: $weight,
        Optional<mixTensor>: $bias
    );
    let results = (outs
        mixTensor: $output
    );
}

def SiLUOp : MIX_Op<"comp.silu", [Pure, SameOperandsAndResultElementType, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "SiLU";
    let description = [{
        SiLU
    }];
    let arguments = (ins
        mixTensor: $input
    );
    let results = (outs
        mixTensor: $output
    );
}

def SelfAttentionOp : MIX_Op<"module.self_attn", [Pure]> {
    let summary = "Self-Attention computation for input tensors";
    let description = [{
        Computes the self-attention mechanism, which includes key, query, and value projections, 
        followed by scaled dot-product attention and optional output projection.
    }];
    let arguments = (ins
        mixTensor: $hidden_states, 
        mixTensor: $residual,       
        mixTensor: $attention_mask
    );
    let results = (outs
        mixTensor: $output        // Resulting tensor after self-attention
    );
}

def SoftmaxOp : MIX_Op<"comp.softmax", [Pure]> {
    let summary = "Compute the softmax of a tensor along a specific axis";
    let description = [{
        Computes the softmax of the input tensor along the specified axis.
    }];
    let arguments = (ins
        mixTensor: $input,
        I64Attr: $axis
    );
    let results = (outs
        mixTensor: $output
    );
}

// 基础算子

def AddOp : MIX_Op<"prim.add", [Pure, Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "add";
    let description = [{
        add
    }];
    let arguments = (ins
        mixData: $lhs,
        mixData: $rhs
    );
    let results = (outs
        mixData: $output
    );
    let hasVerifier = 1;
}

def SubOp : MIX_Op<"prim.sub", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "sub";
    let description = [{
        sub
    }];
    let arguments = (ins
        mixData: $lhs,
        mixData: $rhs
    );
    let results = (outs
        mixData: $output
    );
    let hasVerifier = 1;
}

def MulOp : MIX_Op<"prim.mul", [Pure, Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "mul";
    let description = [{
        mul
    }];
    let arguments = (ins
        mixData: $lhs,
        mixData: $rhs
    );
    let results = (outs
        mixData: $output
    );
    let hasVerifier = 1;
}

def DivOp : MIX_Op<"prim.div", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "div";
    let description = [{
        div
    }];
    let arguments = (ins
        mixData: $lhs,
        mixData: $rhs
    );
    let results = (outs
        mixData: $output
    );
    let hasVerifier = 1;
}

def ConcatOp : MIX_Op<"prim.concat", [Pure]> {
  let summary = "Concatenate tensors along dimension";
  
  let arguments = (ins 
    Variadic<mixTensor>:$inputs,
    I64Attr:$axis
  );
  
  let results = (outs mixTensor:$output);
}

def MatMulOp : MIX_Op<"prim.matmul", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Matrix multiplication of tensors";
    let description = [{
        Performs matrix multiplication on the input tensors.
    }];
    let arguments = (ins
        mixTensor: $lhs,
        mixTensor: $rhs
    );
    let results = (outs
        mixTensor: $output
    );
    let hasVerifier = 1;
}

def BatchMatMulOp : MIX_Op<"unknown.batch_matmul", [Pure]> {
    let summary = "Batch matrix multiplication of tensors";
    let description = [{
        Performs batch matrix multiplication on the input tensors.
    }];
    let arguments = (ins
        mixTensor: $lhs,
        mixTensor: $rhs
    );
    let results = (outs
        mixTensor: $output
    );
}

def ReshapeOp : MIX_Op<"prim.reshape", [Pure]> {
    let summary = "Reshape a tensor to the specified shape";
    let description = [{
        Reshapes the input tensor to the specified shape without changing its data.
    }];
    let arguments = (ins
        mixTensor: $input,
        I64ArrayAttr: $shape
    );
    let results = (outs
        mixTensor: $output
    );
}

def PermuteOp : MIX_Op<"prim.permute", [Pure]> {
    let summary = "Permute the dimensions of a tensor";
    let description = [{
        Permutes the dimensions of the input tensor according to the specified order.
    }];
    let arguments = (ins
        mixTensor: $input,
        I64ArrayAttr: $dims
    );
    let results = (outs
        mixTensor: $output
    );
}



def PowOp : MIX_Op<"prim.pow", [Pure]> {
    let summary = "Element-wise power operation";
    let description = [{
        Computes the element-wise power of the input tensor to the specified exponent.
    }];
    let arguments = (ins
        mixTensor: $input,
        mixTensor: $exponent
    );
    let results = (outs
        mixTensor: $output
    );
}

def ExpOp : MIX_Op<"prim.exp", [Pure]> {
    let summary = "Element-wise exp operation";
    let description = [{
        Computes the element-wise exp of the input tensor to the specified exponent.
    }];
    let arguments = (ins
        mixTensor: $input
    );
    let results = (outs
        mixTensor: $output
    );
}

def ReciprocalOp : MIX_Op<"prim.reciprocal", [Pure]> {
    let summary = "Element-wise reciprocal of a tensor";
    let description = [{
        Computes the element-wise reciprocal (1/x) of the input tensor.
    }];
    let arguments = (ins
        mixTensor: $input
    );
    let results = (outs
        mixTensor: $output
    );
}

def CosOp : MIX_Op<"prim.cos", [Pure]> {
    let summary = "Element-wise cosine operation";
    let description = [{
        Computes the cosine of each element in the input tensor.
    }];
    let arguments = (ins
        mixTensor: $input
    );
    let results = (outs
        mixTensor: $output
    );
}

def SinOp : MIX_Op<"prim.sin", [Pure]> {
    let summary = "Element-wise sine operation";
    let description = [{
        Computes the sine of each element in the input tensor.
    }];
    let arguments = (ins
        mixTensor: $input
    );
    let results = (outs
        mixTensor: $output
    );
}

def NegOp : MIX_Op<"prim.neg", [Pure]> {
    let summary = "Element-wise negation of a tensor";
    let description = [{
        Computes the element-wise negation (-x) of the input tensor.
    }];
    let arguments = (ins
        mixTensor: $input
    );
    let results = (outs
        mixTensor: $output
    );
}


#endif // MIX_OPS_TD