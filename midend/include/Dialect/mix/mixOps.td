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

def LinearOp : MIX_Op<"module.linear", [Pure, InferTypeOpAdaptor]> {
    let summary = "Linear";
    let description = [{
        Linear
    }];
    let arguments = (ins
        mixTensor: $input,
        StrAttr: $params_loc,
        I32Attr: $in_feature,
        I32Attr: $out_feature,
        UnitAttr: $has_bias,
        OptionalAttr<TypeAttr>: $dtype
    );
    let results = (outs
        mixTensor: $output
    );

    let builders = [
        OpBuilder<(ins "mlir::Value": $input, "llvm::StringRef": $params_loc, "int": $in_feature, "int": $out_feature, "bool": $has_bias, "mlir::Type": $dtype), [{
            mlir::UnitAttr has_bisaUnit = has_bias ? $_builder.getUnitAttr() : nullptr;
            return build($_builder, $_state, input, $_builder.getStringAttr(params_loc), $_builder.getI32IntegerAttr(in_feature), $_builder.getI32IntegerAttr(out_feature), has_bisaUnit, mlir::TypeAttr::get(dtype));
        }]>

    ];

    let hasVerifier = 1;
}

def EmbeddingOp : MIX_Op<"module.embedding", [Pure, InferTypeOpAdaptor]> {
    let summary = "Embedding";
    let description = [{
        Embedding
    }];
    let arguments = (ins
        mixTensor: $input,
        StrAttr: $params_loc,
        I32Attr: $num_embeddings,
        I32Attr: $embedding_dim,
        OptionalAttr<I32Attr>: $padding_idx,
        OptionalAttr<F32Attr>: $max_norm,
        OptionalAttr<TypeAttr>: $dtype
    );
    let results = (outs
        mixTensor: $output
    );

    let builders = [
        OpBuilder<(ins "mlir::Value": $input, "llvm::StringRef": $params_loc, "int": $num_embeddings, "int": $embedding_dim, "mlir::IntegerAttr": $padding_idx, "mlir::FloatAttr": $max_norm, "mlir::Type": $dtype), [{
            return build($_builder, $_state, input, $_builder.getStringAttr(params_loc), $_builder.getI32IntegerAttr(num_embeddings), $_builder.getI32IntegerAttr(embedding_dim), padding_idx, max_norm, mlir::TypeAttr::get(dtype));
        }]>,
        OpBuilder<(ins "mlir::Value": $input, "llvm::StringRef": $params_loc, "int": $num_embeddings, "int": $embedding_dim, "mlir::Type": $dtype), [{
            return build($_builder, $_state, input, params_loc, num_embeddings, embedding_dim, mlir::IntegerAttr(), mlir::FloatAttr(), mlir::TypeAttr::get(dtype));
        }]>
    ];

    let hasVerifier = 1;
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

def SigmoidOp: MIX_Op<"comp.sigmoid", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Sigmoid";
    let description = [{
        Sigmoid
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

def SoftmaxOp : MIX_Op<"prim.softmax", [Pure, InferTypeOpAdaptor]> {
    let summary = "Compute the softmax of a tensor along a specific axis";
    let description = [{
        Computes the softmax of the input tensor along the specified axis.
    }];
    let arguments = (ins
        mixTensor: $input,
        SI32Attr: $axis
    );
    let results = (outs
        mixTensor: $output
    );
}

def MeanOp : MIX_Op<"comp.mean", [Pure, InferTypeOpAdaptor]> {
    let summary = "mean";
    let description = [{
        mean
    }];
    let arguments = (ins
        mixTensor: $input,
        I32ArrayAttr: $dims,
        BoolAttr: $keepDim
    );
    let results = (outs
        mixTensor: $output
    );
    let hasVerifier = 1;
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

def LtOp : MIX_Op<"prim.lt", [Pure, InferTypeOpAdaptor]> {
    let summary = "a letter than b";
    let description = [{
        a letter than b
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

def BitwiseNotOp : MIX_Op<"prim.bitwise_not", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "bitwise_not";
    let description = [{
        bitwise_not
    }];
    let arguments = (ins
        mixData: $input
    );
    let results = (outs
        mixData: $output
    );
}

def BitwiseOrOp : MIX_Op<"prim.bitwise_or", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "bitwise_or";
    let description = [{
        bitwise_or
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

def ExpandOp : MIX_Op<"prim.expand", [Pure, InferTypeOpAdaptor]> {
    let summary = "expand";
    let description = [{
        expand
    }];
    let arguments = (ins
        mixData: $input,
        I64ArrayAttr: $shape
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

def ConcatOp : MIX_Op<"prim.concat", [Pure, InferTypeOpAdaptor]> {
  let summary = "Concatenate tensors along dimension";
  
  let arguments = (ins 
    Variadic<mixTensor>:$inputs,
    I64Attr:$axis
  );
  
  let results = (outs mixTensor:$output);

  let hasVerifier = 1;
}

def SliceOp : MIX_Op<"prim.slice", [Pure, InferTypeOpAdaptor]> {
  let summary = "Slice";
  
  let arguments = (ins 
    mixTensor:$inputs,
    I64Attr:$dim,
    I64Attr:$start,
    I64Attr:$end,
    I64Attr:$step
  );
  
  let results = (outs mixTensor:$output);

  let hasVerifier = 1;
}

def TransposeOp : MIX_Op<"prim.transpose", [Pure, InferTypeOpAdaptor]> {
  let summary = "transpose";
  
  let arguments = (ins 
    mixTensor:$input,
    I32Attr:$dim1,
    I32Attr:$dim2
  );
  
  let results = (outs mixTensor:$output);

  let hasVerifier = 1;
}

def UnsqueezeOp : MIX_Op<"prim.unsqueeze", [Pure, InferTypeOpAdaptor]> {
  let summary = "unsqueeze";
  
  let arguments = (ins 
    mixTensor:$input,
    I32Attr:$axis
  );
  
  let results = (outs mixTensor:$output);

  let hasVerifier = 1;
}

def MaskedFillOp : MIX_Op<"prim.masked_fill", [Pure, InferTypeOpAdaptor]> {
  let summary = "Masked Fill";

  let arguments = (ins 
    mixTensor:$input,       // 输入张量
    mixTensor:$mask,        // 掩码张量 (布尔类型)
    mixNumber:$number       // 用于填充的标量值    
  );

  let results = (outs mixTensor:$output);  // 输出张量

  let hasVerifier = 1;
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

def BatchMatMulOp : MIX_Op<"prim.batch_matmul", [Pure, InferTypeOpAdaptor]> {
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

    let hasVerifier = 1;
}

def ReshapeOp : MIX_Op<"prim.reshape", [Pure, InferTypeOpAdaptor]> {
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

    let hasVerifier = 1;
}

def PermuteOp : MIX_Op<"prim.permute", [Pure, InferTypeOpAdaptor]> {
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

    let hasVerifier = 1;
}



def PowOp : MIX_Op<"prim.pow", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "Element-wise power operation";
    let description = [{
        Computes the element-wise power of the input tensor to the specified exponent.
    }];
    let arguments = (ins
        mixData: $input,
        mixData: $exponent
    );
    let results = (outs
        mixData: $output
    );
    let hasVerifier = 1;
}

def ExpOp : MIX_Op<"prim.exp", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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

def ReciprocalOp : MIX_Op<"prim.reciprocal", [Pure, InferTypeOpAdaptor]> {
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

def CosOp : MIX_Op<"prim.cos", [Pure, InferTypeOpAdaptor]> {
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

def SinOp : MIX_Op<"prim.sin", [Pure, InferTypeOpAdaptor]> {
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

def NegOp : MIX_Op<"prim.neg", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
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

def RsqrtOp : MIX_Op<"prim.rsqrt", [Pure, Commutative, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
    let summary = "rsqrt";
    let description = [{
        rsqrt
    }];
    let arguments = (ins
        mixData: $input
    );
    let results = (outs
        mixData: $output
    );
}

def ReduceSumOp : MIX_Op<"prim.reduce_sum", [InferTypeOpAdaptor]> {
  let summary = "Reduce Sum operator";

  let description = [{
    Reduce a tensor along the given axis by computing the sum of the axis.
  }];

  let arguments = (ins
    mixTensor:$input,
    I32Attr:$axis
  );

  let results = (outs
    mixTensor:$output
  );

  let hasVerifier = 1;
}

def TanhOp : MIX_Op<"prim.tanh", [Pure, InferTypeOpAdaptor]> {
    let summary = "Element-wise tanh operation";
    let description = [{
        Computes the tanh of each element in the input tensor.
    }];
    let arguments = (ins
        mixData: $input
    );
    let results = (outs
        mixData: $output
    );
}

def WeightOp : MIX_Op<"comp.weight", [Pure]> {
    let summary = "load param";
    let description = [{
        load param from model file.
    }];
    let arguments = (ins  
        StrAttr: $param_loc
    );
    let results = (outs
        mixTensor: $output
    );
}

def ConstantOp : MIX_Op<"prim.constant", [Pure, AllTypesMatch<["value", "output"]>]> {
    let summary = "constant op";
    let description = [{
        constant op
    }];
    let arguments = (ins
        TypedAttrInterface: $value
    );
    let results = (outs
        mixData: $output
    );
}

def ConvertOp : MIX_Op<"prim.convert",
    [Pure, SameOperandsAndResultShape, InferTypeOpAdaptor]> { 
  let summary = "Convert operation";
  let description = [{
    Performs an element-wise conversion from one element type to another on
    `operand` tensor and produces a `result` tensor.

    Example:
    ```mlir
    %result = prim.convert %operand : (tensor<3xi64>) -> tensor<3xcomplex<f64>>
    ```
  }];
  let arguments = (ins
      mixData: $value,
      TypeAttr: $element_ty
  );
  let results = (outs
      mixData: $output
  );
//   let builders = [
//     OpBuilder<(ins "mlir::Value": $value, "mlir::Type": $element_ty), [{
//         return build($_builder, $_state, value, mlir::TypeAttr::get(element_ty));
//     }]>
//   ];
}
def GetItemOp : MIX_Op<"comp.getitem", [Pure, InferTypeOpAdaptor]> {
    let summary = "GetItem";
    let description = [{
        Get item from tensor
    }];
    let arguments = (ins
        mixTensor: $value,
        AnyTypeOf<[mixTensor, mixInt]>: $indice
    );
    let results = (outs
        mixTensor: $output
    );
}

def GatherOp : MIX_Op<"prim.gather", [Pure, InferTypeOpAdaptor]> {
    let summary = "Gather";
    let description = [{
        gather op.
    }];
    let arguments = (ins
        mixTensor: $values,
        RankedTensorOf<[I32]>: $indices
    );
    let results = (outs
        mixTensor: $output
    );
}



#endif // MIX_OPS_TD