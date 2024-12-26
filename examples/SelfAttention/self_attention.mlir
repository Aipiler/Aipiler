"builtin.module"() ({
  "func.func"() <{function_type = (tensor<*xf32>) -> (), sym_name = "printMemrefF32", sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "self_attention", sym_visibility = "private"}> ({
    %0 = "mix.prim.weight"() <{param_loc = "hidden_states"}> : () -> tensor<1x40x5120xf32>
    %1 = "mix.prim.weight"() <{param_loc = "residual"}> : () -> tensor<1x40x5120xf32>
    %2 = "mix.prim.weight"() <{param_loc = "attention_mask"}> : () -> tensor<1x1x40x40xf32>
    %3 = "mix.prim.weight"() <{param_loc = "Self_attn.query.weight"}> : () -> tensor<5120x5120xf32>
    %4 = "mix.prim.weight"() <{param_loc = "Self_attn.key_value.weight"}> : () -> tensor<10240x5120xf32>
    %5 = "mix.prim.weight"() <{param_loc = "Self_attn.dense.weight"}> : () -> tensor<5120x5120xf32>
    %6 = "mix.prim.weight"() <{param_loc = "Self_attn.dense.bias"}> : () -> tensor<5120xf32>
    %7 = "mix.prim.transpose"(%0) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf32>) -> tensor<40x1x5120xf32>
    %8 = "mix.prim.transpose"(%3) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf32>) -> tensor<5120x5120xf32>
    %9 = "mix.prim.reshape"(%7) <{shape = [40, 5120]}> : (tensor<40x1x5120xf32>) -> tensor<40x5120xf32>
    %10 = "mix.prim.matmul"(%9, %8) : (tensor<40x5120xf32>, tensor<5120x5120xf32>) -> tensor<40x5120xf32>
    %11 = "mix.prim.reshape"(%10) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf32>) -> tensor<40x1x5120xf32>
    %12 = "mix.prim.reshape"(%11) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf32>) -> tensor<40x1x32x160xf32>
    %13 = "mix.prim.transpose"(%4) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf32>) -> tensor<5120x10240xf32>
    %14 = "mix.prim.reshape"(%7) <{shape = [40, 5120]}> : (tensor<40x1x5120xf32>) -> tensor<40x5120xf32>
    %15 = "mix.prim.matmul"(%14, %13) : (tensor<40x5120xf32>, tensor<5120x10240xf32>) -> tensor<40x10240xf32>
    %16 = "mix.prim.reshape"(%15) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf32>) -> tensor<40x1x10240xf32>
    %17 = "mix.prim.reshape"(%16) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf32>) -> tensor<40x1x32x320xf32>
    %18 = "mix.prim.slice"(%17) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf32>) -> tensor<40x1x32x160xf32>
    %19 = "mix.prim.slice"(%17) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf32>) -> tensor<40x1x32x160xf32>
    %20 = "mix.prim.reshape"(%12) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf32>) -> tensor<40x32x160xf32>
    %21 = "mix.prim.reshape"(%18) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf32>) -> tensor<40x32x160xf32>
    %22 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi64>}> : () -> tensor<80xi64>
    %23 = "mix.prim.convert"(%22) <{element_ty = f32}> : (tensor<80xi64>) -> tensor<80xf32>
    %24 = "mix.prim.constant"() <{value = 160 : i64}> : () -> i64
    %25 = "mix.prim.div"(%23, %24) : (tensor<80xf32>, i64) -> tensor<80xf32>
    %26 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %27 = "mix.prim.pow"(%26, %25) : (f16, tensor<80xf32>) -> tensor<80xf32>
    %28 = "mix.prim.reciprocal"(%27) : (tensor<80xf32>) -> tensor<80xf32>
    %29 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %30 = "mix.prim.mul"(%29, %28) : (f16, tensor<80xf32>) -> tensor<80xf32>
    %31 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf32>}> : () -> tensor<40xf32>
    %32 = "mix.prim.unsqueeze"(%31) <{axis = 1 : i32}> : (tensor<40xf32>) -> tensor<40x1xf32>
    %33 = "mix.prim.permute"(%32) <{dims = [0, 1]}> : (tensor<40x1xf32>) -> tensor<40x1xf32>
    %34 = "mix.prim.unsqueeze"(%30) <{axis = 1 : i32}> : (tensor<80xf32>) -> tensor<80x1xf32>
    %35 = "mix.prim.permute"(%34) <{dims = [1, 0]}> : (tensor<80x1xf32>) -> tensor<1x80xf32>
    %36 = "mix.prim.mul"(%33, %35) : (tensor<40x1xf32>, tensor<1x80xf32>) -> tensor<40x80xf32>
    %37 = "mix.prim.concat"(%36, %36) <{axis = 1 : i64}> : (tensor<40x80xf32>, tensor<40x80xf32>) -> tensor<40x160xf32>
    %38 = "mix.prim.cos"(%37) : (tensor<40x160xf32>) -> tensor<40x160xf32>
    %39 = "mix.prim.slice"(%38) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf32>) -> tensor<40x160xf32>
    %40 = "mix.prim.unsqueeze"(%39) <{axis = 1 : i32}> : (tensor<40x160xf32>) -> tensor<40x1x160xf32>
    %41 = "mix.prim.slice"(%40) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf32>) -> tensor<40x1x160xf32>
    %42 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %43 = "mix.prim.mul"(%41, %42) : (tensor<40x1x160xf32>, f16) -> tensor<40x1x160xf32>
    %44 = "mix.prim.sin"(%37) : (tensor<40x160xf32>) -> tensor<40x160xf32>
    %45 = "mix.prim.slice"(%44) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf32>) -> tensor<40x160xf32>
    %46 = "mix.prim.unsqueeze"(%45) <{axis = 1 : i32}> : (tensor<40x160xf32>) -> tensor<40x1x160xf32>
    %47 = "mix.prim.slice"(%46) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf32>) -> tensor<40x1x160xf32>
    %48 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %49 = "mix.prim.mul"(%47, %48) : (tensor<40x1x160xf32>, f16) -> tensor<40x1x160xf32>
    %50 = "mix.prim.slice"(%43) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf32>) -> tensor<40x1x160xf32>
    %51 = "mix.prim.slice"(%49) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf32>) -> tensor<40x1x160xf32>
    %52 = "mix.prim.mul"(%20, %50) : (tensor<40x32x160xf32>, tensor<40x1x160xf32>) -> tensor<40x32x160xf32>
    %53 = "mix.prim.slice"(%20) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf32>) -> tensor<40x32x80xf32>
    %54 = "mix.prim.slice"(%20) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf32>) -> tensor<40x32x80xf32>
    %55 = "mix.prim.neg"(%54) : (tensor<40x32x80xf32>) -> tensor<40x32x80xf32>
    %56 = "mix.prim.concat"(%55, %53) <{axis = 2 : i64}> : (tensor<40x32x80xf32>, tensor<40x32x80xf32>) -> tensor<40x32x160xf32>
    %57 = "mix.prim.mul"(%56, %51) : (tensor<40x32x160xf32>, tensor<40x1x160xf32>) -> tensor<40x32x160xf32>
    %58 = "mix.prim.add"(%52, %57) : (tensor<40x32x160xf32>, tensor<40x32x160xf32>) -> tensor<40x32x160xf32>
    %59 = "mix.prim.mul"(%21, %50) : (tensor<40x32x160xf32>, tensor<40x1x160xf32>) -> tensor<40x32x160xf32>
    %60 = "mix.prim.slice"(%21) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf32>) -> tensor<40x32x80xf32>
    %61 = "mix.prim.slice"(%21) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf32>) -> tensor<40x32x80xf32>
    %62 = "mix.prim.neg"(%61) : (tensor<40x32x80xf32>) -> tensor<40x32x80xf32>
    %63 = "mix.prim.concat"(%62, %60) <{axis = 2 : i64}> : (tensor<40x32x80xf32>, tensor<40x32x80xf32>) -> tensor<40x32x160xf32>
    %64 = "mix.prim.mul"(%63, %51) : (tensor<40x32x160xf32>, tensor<40x1x160xf32>) -> tensor<40x32x160xf32>
    %65 = "mix.prim.add"(%59, %64) : (tensor<40x32x160xf32>, tensor<40x32x160xf32>) -> tensor<40x32x160xf32>
    %66 = "mix.prim.reshape"(%58) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf32>) -> tensor<40x1x32x160xf32>
    %67 = "mix.prim.reshape"(%65) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf32>) -> tensor<40x1x32x160xf32>
    %68 = "mix.prim.reshape"(%66) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf32>) -> tensor<40x32x160xf32>
    %69 = "mix.prim.reshape"(%67) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf32>) -> tensor<40x32x160xf32>
    %70 = "mix.prim.transpose"(%68) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf32>) -> tensor<32x40x160xf32>
    %71 = "mix.prim.transpose"(%69) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf32>) -> tensor<32x40x160xf32>
    %72 = "mix.prim.transpose"(%71) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf32>) -> tensor<32x160x40xf32>
    %73 = "mix.prim.unsqueeze"(%70) <{axis = 3 : i32}> : (tensor<32x40x160xf32>) -> tensor<32x40x160x1xf32>
    %74 = "mix.prim.permute"(%73) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf32>) -> tensor<32x40x1x160xf32>
    %75 = "mix.prim.unsqueeze"(%72) <{axis = 3 : i32}> : (tensor<32x160x40xf32>) -> tensor<32x160x40x1xf32>
    %76 = "mix.prim.permute"(%75) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf32>) -> tensor<32x1x40x160xf32>
    %77 = "mix.prim.permute"(%74) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf32>) -> tensor<32x40x160x1xf32>
    %78 = "mix.prim.reshape"(%77) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf32>) -> tensor<32x40x160xf32>
    %79 = "mix.prim.permute"(%76) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf32>) -> tensor<32x160x40x1xf32>
    %80 = "mix.prim.reshape"(%79) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf32>) -> tensor<32x160x40xf32>
    %81 = "mix.unknown.batch_matmul"(%78, %80) : (tensor<32x40x160xf32>, tensor<32x160x40xf32>) -> tensor<32x40x40xf32>
    %82 = "mix.prim.reshape"(%81) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf32>) -> tensor<32x40x1x40xf32>
    %83 = "mix.prim.permute"(%82) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf32>) -> tensor<32x40x40x1xf32>
    %84 = "mix.prim.reshape"(%83) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf32>) -> tensor<32x40x40xf32>
    %85 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %86 = "mix.prim.mul"(%84, %85) : (tensor<32x40x40xf32>, f16) -> tensor<32x40x40xf32>
    %87 = "mix.prim.reshape"(%86) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %88 = "mix.comp.masked_fill"(%87, %2) <{value = -3.4028234663852886E+38 : f64}> : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xf32>) -> tensor<1x32x40x40xf32>
    %89 = "mix.comp.softmax"(%88) <{axis = -1 : i32}> : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %90 = "mix.prim.reshape"(%89) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %91 = "mix.prim.reshape"(%19) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf32>) -> tensor<40x32x160xf32>
    %92 = "mix.prim.transpose"(%91) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf32>) -> tensor<32x40x160xf32>
    %93 = "mix.unknown.batch_matmul"(%90, %92) : (tensor<32x40x40xf32>, tensor<32x40x160xf32>) -> tensor<32x40x160xf32>
    %94 = "mix.prim.reshape"(%93) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf32>) -> tensor<1x32x40x160xf32>
    %95 = "mix.prim.permute"(%94) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf32>) -> tensor<1x40x32x160xf32>
    %96 = "mix.prim.reshape"(%95) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf32>) -> tensor<1x40x5120xf32>
    %97 = "mix.prim.reshape"(%96) <{shape = [40, 5120]}> : (tensor<1x40x5120xf32>) -> tensor<40x5120xf32>
    %98 = "mix.prim.transpose"(%5) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf32>) -> tensor<5120x5120xf32>
    %99 = "mix.prim.matmul"(%97, %98) : (tensor<40x5120xf32>, tensor<5120x5120xf32>) -> tensor<40x5120xf32>
    %100 = "mix.prim.add"(%99, %6) : (tensor<40x5120xf32>, tensor<5120xf32>) -> tensor<40x5120xf32>
    %101 = "mix.prim.reshape"(%100) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf32>) -> tensor<1x40x5120xf32>
    %102 = "mix.prim.mul"(%1, %101) : (tensor<1x40x5120xf32>, tensor<1x40x5120xf32>) -> tensor<1x40x5120xf32>
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()