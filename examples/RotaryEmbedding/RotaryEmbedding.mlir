module {
  func.func private @_mlir_ciface_printMemrefF16(tensor<*xf16>)
  func.func private @RotaryEmbedding() -> (tensor<8192x1x160xf16>, tensor<8192x1x160xf16>) {
    %0 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1 = "mix.prim.convert"(%0) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %3 = "mix.prim.div"(%1, %2) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %cast = tensor.cast %1 : tensor<80xf16> to tensor<*xf16>
    call @_mlir_ciface_printMemrefF16(%cast) : (tensor<*xf16>) -> ()
    %4 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %5 = "mix.prim.pow"(%4, %3) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %6 = "mix.prim.reciprocal"(%5) : (tensor<80xf16>) -> tensor<80xf16>
    %7 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %8 = "mix.prim.mul"(%7, %6) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %9 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<8192xf16>}> : () -> tensor<8192xf16>
    %10 = "mix.prim.unsqueeze"(%9) <{axis = 1 : i32}> : (tensor<8192xf16>) -> tensor<8192x1xf16>
    %11 = "mix.prim.permute"(%10) <{dims = [0, 1]}> : (tensor<8192x1xf16>) -> tensor<8192x1xf16>
    %12 = "mix.prim.unsqueeze"(%8) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %13 = "mix.prim.permute"(%12) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %14 = "mix.prim.mul"(%11, %13) : (tensor<8192x1xf16>, tensor<1x80xf16>) -> tensor<8192x80xf16>
    %15 = "mix.prim.concat"(%14, %14) <{axis = 1 : i64}> : (tensor<8192x80xf16>, tensor<8192x80xf16>) -> tensor<8192x160xf16>
    %16 = "mix.prim.cos"(%15) : (tensor<8192x160xf16>) -> tensor<8192x160xf16>
    %17 = "mix.prim.slice"(%16) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<8192x160xf16>) -> tensor<8192x160xf16>
    %18 = "mix.prim.unsqueeze"(%17) <{axis = 1 : i32}> : (tensor<8192x160xf16>) -> tensor<8192x1x160xf16>
    %19 = "mix.prim.slice"(%18) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<8192x1x160xf16>) -> tensor<8192x1x160xf16>
    %20 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %21 = "mix.prim.mul"(%19, %20) : (tensor<8192x1x160xf16>, f16) -> tensor<8192x1x160xf16>
    %22 = "mix.prim.sin"(%15) : (tensor<8192x160xf16>) -> tensor<8192x160xf16>
    %23 = "mix.prim.slice"(%22) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<8192x160xf16>) -> tensor<8192x160xf16>
    %24 = "mix.prim.unsqueeze"(%23) <{axis = 1 : i32}> : (tensor<8192x160xf16>) -> tensor<8192x1x160xf16>
    %25 = "mix.prim.slice"(%24) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<8192x1x160xf16>) -> tensor<8192x1x160xf16>
    %26 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %27 = "mix.prim.mul"(%25, %26) : (tensor<8192x1x160xf16>, f16) -> tensor<8192x1x160xf16>
    return %21, %27 : tensor<8192x1x160xf16>, tensor<8192x1x160xf16>
  }
  func.func public @main() {
    %0:2 = call @RotaryEmbedding() : () -> (tensor<8192x1x160xf16>, tensor<8192x1x160xf16>)
    %cast = tensor.cast %0#0 : tensor<8192x1x160xf16> to tensor<*xf16>
    %cast_0 = tensor.cast %0#1 : tensor<8192x1x160xf16> to tensor<*xf16>
    call @_mlir_ciface_printMemrefF16(%cast) : (tensor<*xf16>) -> ()
    call @_mlir_ciface_printMemrefF16(%cast_0) : (tensor<*xf16>) -> ()
    return
  }
}