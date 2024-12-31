#ifndef RVV_OPS_TD
#define RVV_OPS_TD

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
include "rvvDialect.td"
include "rvvTypes.td"

//===----------------------------------------------------------------------===//
// RVV operation definitions
//===----------------------------------------------------------------------===//

def RVVSetVlOp :
    RVV_Op<"setvl",
              !listconcat([], [AllTypesMatch<["avl", "sew", "lmul", "vl"]>])>,
    Arguments<(ins AnySignlessIntegerOrIndex:$avl,
                  AnySignlessIntegerOrIndex:$sew,
                  AnySignlessIntegerOrIndex:$lmul)>,
    Results<(outs AnySignlessIntegerOrIndex:$vl)> {
  let summary = "Set vector length according to AVL, SEW, and LMUL";
  let description = [{
    SetVl operation sets the vector length according to AVL, SEW, and LMUL.
    RISC-V vector extension uses this to achieve a direct and portable
    strip-mining approach, which is purposed to handle a large number of
    elements. The return value of this operation is the number of elements
    for a single iteration.
  }];
  let assemblyFormat = "$avl `,` $sew `,` $lmul attr-dict `:` type($avl)";
}

def RVVLoadOp : RVV_Op<"load">,
    Arguments<(ins Arg<AnyMemRef, "", [MemRead]>:$base, Index:$index,
                       AnySignlessIntegerOrIndex:$length)>,
    Results<(outs ScalableVectorOf<[AnyType]>:$result)> {
  let summary = "Load scalable vector from memory";
  let description = [{
    Load a slice of memory into scalable vector with the given element length.
  }];
  let extraClassDeclaration = [{
    mlir::MemRefType getMemRefType() {
      return getBase().getType().cast<mlir::MemRefType>();
    }
  }];
  let assemblyFormat = "$base `[` $index `]` `,` $length attr-dict `:` "
    "type($base) `,` type($result) `,`  type($length)";
}

def RVVStoreOp : RVV_Op<"store">,
    Arguments<(ins ScalableVectorOf<[AnyType]>:$value, Arg<AnyMemRef, "",
                   [MemWrite]>:$base, Index:$index, AnySignlessIntegerOrIndex:$length)> {
  let summary = "Store scalable vector into memory";
  let description = [{
    Store the given element length of a scalable vector on a slice of memory.
  }];
  let extraClassDeclaration = [{
    mlir::MemRefType getMemRefType() {
      return getBase().getType().cast<mlir::MemRefType>();
    }
  }];
  let assemblyFormat = "$value `,` $base `[` $index `]` `,` $length attr-dict "
    "`:` type($value) `,` type($base) `,`  type($length)";
}

def RsqrtOp : RVV_Op<"rsqrt", [Pure, AllTypesMatch<["src", "dst"]>]>,
    Arguments<(ins ScalableVectorOf<[AnyType]>:$src, AnySignlessIntegerOrIndex:$length)>,
    Results<(outs ScalableVectorOf<[AnyType]>:$dst)> {
  let summary = "Reciprocal of the square-root";
  let description = "Floating-point reciprocal square-root estimated to 7 bits.";
  let assemblyFormat = "$src `,` $length attr-dict `:` type($dst) `,`  type($length)";
}

class RVV_UnaryAAXNoMaskRounding_Op<string mnemonic, string op_description,
                             list<Trait> traits = []> :
  RVV_Op<mnemonic, !listconcat(traits, [AllTypesMatch<["src1", "dst"]>])> {
  let summary = op_description # "for RVV scalable vectors";
  let description = [{ The `rvv.}] # mnemonic # [{` operation can be of 
    vector-vector form or vector-scalar form. It also takes the vector length 
    value and returns a scalable vector with the result of the }]
    # op_description # [{.}];
  let arguments = (ins 
    ScalableVectorOf<[AnyFloat]>:$src1,
    AnySignlessIntegerOrIndex:$frm,
    AnySignlessIntegerOrIndex:$length
  );
  let results = (outs ScalableVectorOf<[AnyFloat]>:$dst);
  let assemblyFormat = "$src1 `,` $frm `,` $length attr-dict `:` type($src1) "
    "`,` type($frm) `,` type($length)";
}

def RVVFsqrtOp : RVV_UnaryAAXNoMaskRounding_Op<"vfsqrt", "Floating-point "
    "square-root.">;
def RVVFrec7Op : RVV_UnaryAAXNoMaskRounding_Op<"vfrec", "Floating-point "
    "reciprocal square-root estimate to 7 bits.">;

class RVV_BinaryAAXNoMask_Op<string mnemonic, string op_description,
                             list<Trait> traits = []> :
  RVV_Op<mnemonic, !listconcat(traits, [AllTypesMatch<["src1", "dst"]>])> {
  let summary = op_description # "for RVV scalable vectors";
  let description = [{ The `rvv.}] # mnemonic # [{` operation can be of 
    vector-vector form or vector-scalar form. It also takes the vector length 
    value and returns a scalable vector with the result of the }]
    # op_description # [{.}];
  let arguments = (ins 
    ScalableVectorOf<[AnyInteger]>:$src1,
    AnyType:$src2,
    AnySignlessIntegerOrIndex:$length
  );
  let results = (outs ScalableVectorOf<[AnyInteger]>:$dst);
  let assemblyFormat = "$src1 `,` $src2 `,` $length attr-dict `:` type($src1) "
    "`,` type($src2) `,` type($length)";
}

def RVVAddOp : RVV_BinaryAAXNoMask_Op<"add", "addition">;
def RVVMulOp : RVV_BinaryAAXNoMask_Op<"mul", "multiplication">;

class RVV_FBinaryAAXNoMask_Op<string mnemonic, string op_description,
                             list<Trait> traits = []> :
  RVV_Op<mnemonic, !listconcat(traits, [AllTypesMatch<["src1", "dst"]>])> {
  let summary = op_description # "for RVV scalable vectors";
  let description = [{ The `rvv.}] # mnemonic # [{` operation can be of 
    vector-vector form or vector-scalar form. It also takes the vector length 
    value and returns a scalable vector with the result of the }]
    # op_description # [{.}];
  let arguments = (ins 
    ScalableVectorOf<[AnyFloat]>:$src1,
    AnyType:$src2,
    AnySignlessIntegerOrIndex:$length
  );
  let results = (outs ScalableVectorOf<[AnyFloat]>:$dst);
  let assemblyFormat = "$src1 `,` $src2 `,` $length attr-dict `:` type($src1) "
    "`,` type($src2) `,` type($length)";
}
def RVVFMaxOp : RVV_FBinaryAAXNoMask_Op<"vfmax", "floating-point maximum">;
def RVVFMinOp : RVV_FBinaryAAXNoMask_Op<"vfmin", "floating-point minimum">;

class RVV_BinaryAAXUnMaskedRounding_Op<string mnemonic, string op_description,
                             list<Trait> traits = [Pure]> :
  RVV_Op<mnemonic, !listconcat(traits, [AllTypesMatch<["src1", "dst"]>])> {
  let summary = op_description # "for RVV scalable vectors";
  let description = [{ The `rvv.}] # mnemonic # [{` operation can be of 
    vector-vector form or vector-scalar form. It also takes the vector length 
    value and returns a scalable vector with the result of the }]
    # op_description # [{.}];
  let arguments = (ins 
    ScalableVectorOf<[AnyFloat]>:$src1,
    AnyType:$src2,
    AnySignlessIntegerOrIndex:$frm,
    AnySignlessIntegerOrIndex:$length
  );
  let results = (outs ScalableVectorOf<[AnyFloat]>:$dst);
  let assemblyFormat = "$src1 `,` $src2 `,` $frm `,` $length attr-dict `:` type($src1) "
    "`,` type($src2)`,` type($frm) `,` type($length)";
}

def RVVFAddOp : RVV_BinaryAAXUnMaskedRounding_Op<"fadd", "floating-point addition">; 
def RVVFMulOp : RVV_BinaryAAXUnMaskedRounding_Op<"fmul", "floating-point multiplication">;


def RVVMAccOp :
  Op<RVV_Dialect, "vmacc", [Pure, AllTypesMatch<["src1", "dst"]>]>,
    Arguments<(ins ScalableVectorOf<[AnyInteger]>:$src1,
                   ScalableVectorOf<[AnyInteger]>:$src2,
                   ScalableVectorOf<[AnyInteger]>:$acc,
                   AnySignlessIntegerOrIndex:$length
                   )>,
    Results<(outs ScalableVectorOf<[AnyInteger]>:$dst)> {
  let summary = "vector fused multiply-add";
  let description = [{
    Multiply-add expressions operate on n-D vectors and compute a fused
    pointwise multiply-and-accumulate: `$dst = `$src1 * $src2 + $acc`.

    Example:
    ```mlir
    %3 = rvv.vmacc %0, %1, %2: vector<8x16xf32>
    ```
  }];
  let assemblyFormat = "$src1 `,` $src2 `,` $acc `,` $length attr-dict `:` type($src1) `,`  type($src2) `,` type($acc) `,` type($length)";
  // let extraClassDeclaration = [{
  //   VectorType getVectorType() { return ::llvm::cast<VectorType>(getLhs().getType()); }
  // }];
}

//===----------------------------------------------------------------------===//
// RVV intrinsic operation definitions
//===----------------------------------------------------------------------===//

class RVV_VSetVlI_IntrOp<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVV_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0], 
                  /*list<int> overloadedOperands=*/[], 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/1>;

class RVV_USLoad_IntrOp<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVV_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0], 
                  /*list<int> overloadedOperands=*/[2], 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/1>;

class RVV_USStore_IntrOp<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVV_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[], 
                  /*list<int> overloadedOperands=*/[0, 2], 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/0>;

class RVV_UnaryNoMask_IntrOp<string mnemonic, list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVV_Dialect,
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0], 
                  /*list<int> overloadedOperands=*/[2], 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/1>;

class RVV_BinaryAAXNoMask_IntrOp<string mnemonic,
                                 list<Trait> traits = []> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVV_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0], 
                  /*list<int> overloadedOperands=*/[2, 3], 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/1>;

class RVV_TernaryNoMask_IntrOp<string mnemonic,
                                 list<Trait> traits = [Pure]> :
  LLVM_IntrOpBase</*Dialect dialect=*/RVV_Dialect, 
                  /*string opName=*/"intr." # mnemonic,
                  /*string enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                  /*list<int> overloadedResults=*/[0], 
                  /*list<int> overloadedOperands=*/[1, 3], 
                  /*list<Trait> traits=*/traits, 
                  /*int numResults=*/1>;

def RVVIntrSetVlIOp : RVV_VSetVlI_IntrOp<"vsetvli">,
  Arguments<(ins AnyInteger, AnyInteger, AnyInteger)>;
def RVVIntrLoadEleOp : RVV_USLoad_IntrOp<"vle">,
  Arguments<(ins AnyScalableVector, LLVM_AnyPointer, AnyInteger)>;
def RVVIntrStoreEleOp : RVV_USStore_IntrOp<"vse">,
  Arguments<(ins AnyScalableVector, LLVM_AnyPointer, AnyInteger)>;
def IntrFrsqrt7Op : RVV_UnaryNoMask_IntrOp<"vfrsqrt7">,
  Arguments<(ins AnyScalableVector, AnyScalableVector, AnyInteger)>;
def RVVIntrFsqrtOp : RVV_UnaryNoMask_IntrOp<"vfsqrt">,
  Arguments<(ins AnyScalableVector, AnyScalableVector, AnyInteger, AnyInteger)>;
def RVVIntrFrec7Op : RVV_UnaryNoMask_IntrOp<"vfrec7">,
  Arguments<(ins AnyScalableVector, AnyScalableVector, AnyInteger, AnyInteger)>;
def RVVIntrAddOp : RVV_BinaryAAXNoMask_IntrOp<"vadd">, 
  Arguments<(ins AnyScalableVector, AnyScalableVector, AnyType, AnyInteger)>;
def RVVIntrMulOp : RVV_BinaryAAXNoMask_IntrOp<"vmul">, 
  Arguments<(ins AnyScalableVector, AnyScalableVector, AnyType, AnyInteger)>;
def RVVIntrFMaxOp : RVV_BinaryAAXNoMask_IntrOp<"vfmax">, 
  Arguments<(ins AnyScalableVector, AnyScalableVector, AnyType, AnyInteger)>;
def RVVIntrFMinOp : RVV_BinaryAAXNoMask_IntrOp<"vfmin">,
  Arguments<(ins AnyScalableVector, AnyScalableVector, AnyType, AnyInteger)>;
def RVVIntrFAddOp : RVV_BinaryAAXNoMask_IntrOp<"vfadd">, 
  Arguments<(ins AnyScalableVector, AnyScalableVector, AnyType, AnyInteger, AnyInteger)>;
def RVVIntrFMulOp : RVV_BinaryAAXNoMask_IntrOp<"vfmul">, 
  Arguments<(ins AnyScalableVector, AnyScalableVector, AnyType, AnyInteger, AnyInteger)>;
def RVVIntrMAccOp : RVV_TernaryNoMask_IntrOp<"vmacc">,
  Arguments<(ins AnyScalableVector, AnyType, AnyScalableVector, AnyInteger, AnyInteger)>;


#endif // RVV_OPS