#ifndef EINDUM_ENUMS
#define EINDUM_ENUMS

include "mlir/IR/EnumAttr.td"

// Define the function attribute enums matching the OpDSL functions.
def UnaryFn : I32EnumAttr<"UnaryFn", "", [
  I32EnumAttrCase<"exp", 0>,
  I32EnumAttrCase<"log", 1>,
  I32EnumAttrCase<"abs", 2>,
  I32EnumAttrCase<"ceil", 3>,
  I32EnumAttrCase<"floor", 4>,
  I32EnumAttrCase<"negf", 5>,
  I32EnumAttrCase<"reciprocal", 6>,
  I32EnumAttrCase<"round", 7>,
  I32EnumAttrCase<"sqrt", 8>,
  I32EnumAttrCase<"rsqrt", 9>,
  I32EnumAttrCase<"square", 10>,
  I32EnumAttrCase<"tanh", 11>,
  I32EnumAttrCase<"erf", 12>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::einsum";
}
def BinaryFn : I32EnumAttr<"BinaryFn", "", [
  I32EnumAttrCase<"add", 0>,
  I32EnumAttrCase<"sub", 1>,
  I32EnumAttrCase<"mul", 2>,
  I32EnumAttrCase<"div", 3>,
  I32EnumAttrCase<"div_unsigned", 4>,
  I32EnumAttrCase<"max_signed", 5>,
  I32EnumAttrCase<"min_signed", 6>,
  I32EnumAttrCase<"max_unsigned", 7>,
  I32EnumAttrCase<"min_unsigned", 8>,
  I32EnumAttrCase<"powf", 9>
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::einsum";
}


#endif // EINDUM_ENUMS
