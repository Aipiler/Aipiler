module {
  func.func private @TransformerBlock() {
    %0 = "mix.comp.weight"() <{param_loc = "hidden_states"}> : () -> tensor<1x40x5120xf32>
    %1 = "mix.comp.weight"() <{param_loc = "attention_mask"}> : () -> tensor<1x1x40x40xi1>
    %2 = "mix.comp.weight"() <{param_loc = "model_parameters.rms.weight"}> : () -> tensor<5120xf32>
    %cst = arith.constant dense<2.000000e+00> : tensor<1xf32>
    %3 = "mix.prim.pow"(%0, %cst) : (tensor<1x40x5120xf32>, tensor<1xf32>) -> tensor<1x40x5120xf32>
    %4 = "mix.comp.mean"(%3) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf32>) -> tensor<1x40x1xf32>
    %cst_0 = arith.constant 9.99999997E-7 : f32
    %5 = "mix.prim.add"(%4, %cst_0) : (tensor<1x40x1xf32>, f32) -> tensor<1x40x1xf32>
    %6 = "mix.prim.rsqrt"(%5) : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %7 = "mix.prim.mul"(%0, %6) : (tensor<1x40x5120xf32>, tensor<1x40x1xf32>) -> tensor<1x40x5120xf32>
    %8 = "mix.prim.mul"(%2, %7) : (tensor<5120xf32>, tensor<1x40x5120xf32>) -> tensor<1x40x5120xf32>
    %9 = "mix.comp.weight"() <{param_loc = "Self_attn.query.weight"}> : () -> tensor<5120x5120xf32>
    %10 = "mix.comp.weight"() <{param_loc = "Self_attn.key_value.weight"}> : () -> tensor<10240x5120xf32>
    %11 = "mix.comp.weight"() <{param_loc = "Self_attn.dense.weight"}> : () -> tensor<5120x5120xf32>
    %12 = "mix.comp.weight"() <{param_loc = "Self_attn.dense.bias"}> : () -> tensor<5120xf32>
    %13 = "mix.prim.transpose"(%8) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf32>) -> tensor<40x1x5120xf32>
    %14 = "mix.prim.transpose"(%9) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf32>) -> tensor<5120x5120xf32>
    %15 = "mix.prim.reshape"(%13) <{shape = [40, 5120]}> : (tensor<40x1x5120xf32>) -> tensor<40x5120xf32>
    %16 = "mix.prim.matmul"(%15, %14) : (tensor<40x5120xf32>, tensor<5120x5120xf32>) -> tensor<40x5120xf32>
    %17 = "mix.prim.reshape"(%16) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf32>) -> tensor<40x1x5120xf32>
    %18 = "mix.prim.reshape"(%17) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf32>) -> tensor<40x1x32x160xf32>
    %19 = "mix.prim.transpose"(%10) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf32>) -> tensor<5120x10240xf32>
    %20 = "mix.prim.reshape"(%13) <{shape = [40, 5120]}> : (tensor<40x1x5120xf32>) -> tensor<40x5120xf32>
    %21 = "mix.prim.matmul"(%20, %19) : (tensor<40x5120xf32>, tensor<5120x10240xf32>) -> tensor<40x10240xf32>
    %22 = "mix.prim.reshape"(%21) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf32>) -> tensor<40x1x10240xf32>
    %23 = "mix.prim.reshape"(%22) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf32>) -> tensor<40x1x32x320xf32>
    %24 = "mix.prim.slice"(%23) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf32>) -> tensor<40x1x32x160xf32>
    %25 = "mix.prim.slice"(%23) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf32>) -> tensor<40x1x32x160xf32>
    %26 = "mix.prim.reshape"(%18) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf32>) -> tensor<40x32x160xf32>
    %27 = "mix.prim.reshape"(%24) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf32>) -> tensor<40x32x160xf32>
    %28 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi64>}> : () -> tensor<80xi64>
    %29 = "mix.prim.convert"(%28) <{element_ty = f32}> : (tensor<80xi64>) -> tensor<80xf32>
    %30 = "mix.prim.constant"() <{value = 160 : i64}> : () -> i64
    %31 = "mix.prim.div"(%29, %30) : (tensor<80xf32>, i64) -> tensor<80xf32>
    %32 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %33 = "mix.prim.pow"(%32, %31) : (f16, tensor<80xf32>) -> tensor<80xf32>
    %34 = "mix.prim.reciprocal"(%33) : (tensor<80xf32>) -> tensor<80xf32>
    %35 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %36 = "mix.prim.mul"(%35, %34) : (f16, tensor<80xf32>) -> tensor<80xf32>
    %37 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf32>}> : () -> tensor<40xf32>
    %38 = "mix.prim.unsqueeze"(%37) <{axis = 1 : i32}> : (tensor<40xf32>) -> tensor<40x1xf32>
    %39 = "mix.prim.permute"(%38) <{dims = [0, 1]}> : (tensor<40x1xf32>) -> tensor<40x1xf32>
    %40 = "mix.prim.unsqueeze"(%36) <{axis = 1 : i32}> : (tensor<80xf32>) -> tensor<80x1xf32>
    %41 = "mix.prim.permute"(%40) <{dims = [1, 0]}> : (tensor<80x1xf32>) -> tensor<1x80xf32>
    %42 = "mix.prim.mul"(%39, %41) : (tensor<40x1xf32>, tensor<1x80xf32>) -> tensor<40x80xf32>
    %43 = "mix.prim.concat"(%42, %42) <{axis = 1 : i64}> : (tensor<40x80xf32>, tensor<40x80xf32>) -> tensor<40x160xf32>
    %44 = "mix.prim.cos"(%43) : (tensor<40x160xf32>) -> tensor<40x160xf32>
    %45 = "mix.prim.slice"(%44) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf32>) -> tensor<40x160xf32>
    %46 = "mix.prim.unsqueeze"(%45) <{axis = 1 : i32}> : (tensor<40x160xf32>) -> tensor<40x1x160xf32>
    %47 = "mix.prim.slice"(%46) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf32>) -> tensor<40x1x160xf32>
    %48 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %49 = "mix.prim.mul"(%47, %48) : (tensor<40x1x160xf32>, f16) -> tensor<40x1x160xf32>
    %50 = "mix.prim.sin"(%43) : (tensor<40x160xf32>) -> tensor<40x160xf32>
    %51 = "mix.prim.slice"(%50) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf32>) -> tensor<40x160xf32>
    %52 = "mix.prim.unsqueeze"(%51) <{axis = 1 : i32}> : (tensor<40x160xf32>) -> tensor<40x1x160xf32>
    %53 = "mix.prim.slice"(%52) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf32>) -> tensor<40x1x160xf32>
    %54 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %55 = "mix.prim.mul"(%53, %54) : (tensor<40x1x160xf32>, f16) -> tensor<40x1x160xf32>
    %56 = "mix.prim.slice"(%49) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf32>) -> tensor<40x1x160xf32>
    %57 = "mix.prim.slice"(%55) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf32>) -> tensor<40x1x160xf32>
    %58 = "mix.prim.mul"(%26, %56) : (tensor<40x32x160xf32>, tensor<40x1x160xf32>) -> tensor<40x32x160xf32>
    %59 = "mix.prim.slice"(%26) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf32>) -> tensor<40x32x80xf32>
    %60 = "mix.prim.slice"(%26) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf32>) -> tensor<40x32x80xf32>
    %61 = "mix.prim.neg"(%60) : (tensor<40x32x80xf32>) -> tensor<40x32x80xf32>
    %62 = "mix.prim.concat"(%61, %59) <{axis = 2 : i64}> : (tensor<40x32x80xf32>, tensor<40x32x80xf32>) -> tensor<40x32x160xf32>
    %63 = "mix.prim.mul"(%62, %57) : (tensor<40x32x160xf32>, tensor<40x1x160xf32>) -> tensor<40x32x160xf32>
    %64 = "mix.prim.add"(%58, %63) : (tensor<40x32x160xf32>, tensor<40x32x160xf32>) -> tensor<40x32x160xf32>
    %65 = "mix.prim.mul"(%27, %56) : (tensor<40x32x160xf32>, tensor<40x1x160xf32>) -> tensor<40x32x160xf32>
    %66 = "mix.prim.slice"(%27) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf32>) -> tensor<40x32x80xf32>
    %67 = "mix.prim.slice"(%27) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf32>) -> tensor<40x32x80xf32>
    %68 = "mix.prim.neg"(%67) : (tensor<40x32x80xf32>) -> tensor<40x32x80xf32>
    %69 = "mix.prim.concat"(%68, %66) <{axis = 2 : i64}> : (tensor<40x32x80xf32>, tensor<40x32x80xf32>) -> tensor<40x32x160xf32>
    %70 = "mix.prim.mul"(%69, %57) : (tensor<40x32x160xf32>, tensor<40x1x160xf32>) -> tensor<40x32x160xf32>
    %71 = "mix.prim.add"(%65, %70) : (tensor<40x32x160xf32>, tensor<40x32x160xf32>) -> tensor<40x32x160xf32>
    %72 = "mix.prim.reshape"(%64) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf32>) -> tensor<40x1x32x160xf32>
    %73 = "mix.prim.reshape"(%71) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf32>) -> tensor<40x1x32x160xf32>
    %74 = "mix.prim.reshape"(%72) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf32>) -> tensor<40x32x160xf32>
    %75 = "mix.prim.reshape"(%73) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf32>) -> tensor<40x32x160xf32>
    %76 = "mix.prim.transpose"(%74) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf32>) -> tensor<32x40x160xf32>
    %77 = "mix.prim.transpose"(%75) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf32>) -> tensor<32x40x160xf32>
    %78 = "mix.prim.transpose"(%77) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf32>) -> tensor<32x160x40xf32>
    %79 = "mix.prim.unsqueeze"(%76) <{axis = 3 : i32}> : (tensor<32x40x160xf32>) -> tensor<32x40x160x1xf32>
    %80 = "mix.prim.permute"(%79) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf32>) -> tensor<32x40x1x160xf32>
    %81 = "mix.prim.unsqueeze"(%78) <{axis = 3 : i32}> : (tensor<32x160x40xf32>) -> tensor<32x160x40x1xf32>
    %82 = "mix.prim.permute"(%81) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf32>) -> tensor<32x1x40x160xf32>
    %83 = "mix.prim.permute"(%80) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf32>) -> tensor<32x40x160x1xf32>
    %84 = "mix.prim.reshape"(%83) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf32>) -> tensor<32x40x160xf32>
    %85 = "mix.prim.permute"(%82) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf32>) -> tensor<32x160x40x1xf32>
    %86 = "mix.prim.reshape"(%85) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf32>) -> tensor<32x160x40xf32>
    %87 = "mix.prim.batch_matmul"(%84, %86) : (tensor<32x40x160xf32>, tensor<32x160x40xf32>) -> tensor<32x40x40xf32>
    %88 = "mix.prim.reshape"(%87) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf32>) -> tensor<32x40x1x40xf32>
    %89 = "mix.prim.permute"(%88) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf32>) -> tensor<32x40x40x1xf32>
    %90 = "mix.prim.reshape"(%89) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf32>) -> tensor<32x40x40xf32>
    %91 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %92 = "mix.prim.mul"(%90, %91) : (tensor<32x40x40xf32>, f16) -> tensor<32x40x40xf32>
    %93 = "mix.prim.reshape"(%92) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %94 = "mix.comp.masked_fill"(%93, %1) <{value = -3.4028234663852886E+38 : f64}> : (tensor<1x32x40x40xf32>, tensor<1x1x40x40xi1>) -> tensor<1x32x40x40xf32>
    %95 = "mix.comp.softmax"(%94) <{axis = -1 : si32}> : (tensor<1x32x40x40xf32>) -> tensor<1x32x40x40xf32>
    %96 = "mix.prim.reshape"(%95) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf32>) -> tensor<32x40x40xf32>
    %97 = "mix.prim.reshape"(%25) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf32>) -> tensor<40x32x160xf32>
    %98 = "mix.prim.transpose"(%97) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf32>) -> tensor<32x40x160xf32>
    %99 = "mix.prim.batch_matmul"(%96, %98) : (tensor<32x40x40xf32>, tensor<32x40x160xf32>) -> tensor<32x40x160xf32>
    %100 = "mix.prim.reshape"(%99) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf32>) -> tensor<1x32x40x160xf32>
    %101 = "mix.prim.permute"(%100) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf32>) -> tensor<1x40x32x160xf32>
    %102 = "mix.prim.reshape"(%101) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf32>) -> tensor<1x40x5120xf32>
    %103 = "mix.prim.reshape"(%102) <{shape = [40, 5120]}> : (tensor<1x40x5120xf32>) -> tensor<40x5120xf32>
    %104 = "mix.prim.transpose"(%11) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf32>) -> tensor<5120x5120xf32>
    %105 = "mix.prim.matmul"(%103, %104) : (tensor<40x5120xf32>, tensor<5120x5120xf32>) -> tensor<40x5120xf32>
    %106 = "mix.prim.add"(%105, %12) : (tensor<40x5120xf32>, tensor<5120xf32>) -> tensor<40x5120xf32>
    %107 = "mix.prim.reshape"(%106) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf32>) -> tensor<1x40x5120xf32>
    %108 = "mix.prim.mul"(%0, %107) : (tensor<1x40x5120xf32>, tensor<1x40x5120xf32>) -> tensor<1x40x5120xf32>
    %109 = "mix.comp.weight"() <{param_loc = "model_parameters.rms.weight"}> : () -> tensor<5120xf32>
    %cst_1 = arith.constant dense<2.000000e+00> : tensor<1xf32>
    %110 = "mix.prim.pow"(%108, %cst_1) : (tensor<1x40x5120xf32>, tensor<1xf32>) -> tensor<1x40x5120xf32>
    %111 = "mix.comp.mean"(%110) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf32>) -> tensor<1x40x1xf32>
    %cst_2 = arith.constant 9.99999997E-7 : f32
    %112 = "mix.prim.add"(%111, %cst_2) : (tensor<1x40x1xf32>, f32) -> tensor<1x40x1xf32>
    %113 = "mix.prim.rsqrt"(%112) : (tensor<1x40x1xf32>) -> tensor<1x40x1xf32>
    %114 = "mix.prim.mul"(%108, %113) : (tensor<1x40x5120xf32>, tensor<1x40x1xf32>) -> tensor<1x40x5120xf32>
    %115 = "mix.prim.mul"(%109, %114) : (tensor<5120xf32>, tensor<1x40x5120xf32>) -> tensor<1x40x5120xf32>
    %116 = "mix.module.linear"(%115) <{dtype = f32, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "model_parameters.mlp.linear0"}> : (tensor<1x40x5120xf32>) -> tensor<1x40x12288xf32>
    %117 = "mix.comp.silu"(%116) : (tensor<1x40x12288xf32>) -> tensor<1x40x12288xf32>
    %118 = "mix.module.linear"(%115) <{dtype = f32, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "model_parameters.mlp.linear1"}> : (tensor<1x40x5120xf32>) -> tensor<1x40x12288xf32>
    %119 = "mix.prim.mul"(%117, %118) : (tensor<1x40x12288xf32>, tensor<1x40x12288xf32>) -> tensor<1x40x12288xf32>
    %120 = "mix.module.linear"(%119) <{dtype = f32, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "model_parameters.mlp.linear2"}> : (tensor<1x40x12288xf32>) -> tensor<1x40x5120xf32>
    %121 = "mix.prim.add"(%120, %108) : (tensor<1x40x5120xf32>, tensor<1x40x5120xf32>) -> tensor<1x40x5120xf32>
    return
  }
}