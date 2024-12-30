module {
  func.func private @RotaryEmbedding() -> (tensor<40x1x160xf16>, tensor<40x1x160xf16>) {
    %0 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1 = "mix.prim.convert"(%0) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3 = "mix.prim.div"(%1, %2) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %5 = "mix.prim.pow"(%4, %3) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %6 = "mix.prim.reciprocal"(%5) : (tensor<80xf16>) -> tensor<80xf16>
    %7 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %8 = "mix.prim.mul"(%7, %6) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %9 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %10 = "mix.prim.unsqueeze"(%9) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %11 = "mix.prim.permute"(%10) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %12 = "mix.prim.unsqueeze"(%8) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %13 = "mix.prim.permute"(%12) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %14 = "mix.prim.mul"(%11, %13) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %15 = "mix.prim.concat"(%14, %14) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %16 = "mix.prim.cos"(%15) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %17 = "mix.prim.slice"(%16) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %18 = "mix.prim.unsqueeze"(%17) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %19 = "mix.prim.slice"(%18) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %20 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %21 = "mix.prim.mul"(%19, %20) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %22 = "mix.prim.sin"(%15) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %23 = "mix.prim.slice"(%22) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %24 = "mix.prim.unsqueeze"(%23) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %25 = "mix.prim.slice"(%24) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %26 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %27 = "mix.prim.mul"(%25, %26) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    return %21, %27 : tensor<40x1x160xf16>, tensor<40x1x160xf16>
  }
}