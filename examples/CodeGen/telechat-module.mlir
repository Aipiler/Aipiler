module {
  func.func private @Telechat(%arg0: tensor<1x40xi32>) -> tensor<1x40x120000xf16> {
    %0 = "mix.comp.weight"() <{param_loc = "transformer.word_embeddings.weight.weight"}> : () -> tensor<1x120000x5120xf16>
    %1 = "mix.prim.reshape"(%arg0) <{shape = [1, 40]}> : (tensor<1x40xi32>) -> tensor<1x40xi32>
    %2 = "mix.prim.gather"(%0, %1) : (tensor<1x120000x5120xf16>, tensor<1x40xi32>) -> tensor<1x40x5120xf16>
    %3 = "mix.prim.reshape"(%2) <{shape = [1, 40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4 = "mix.prim.constant"() <{value = dense<true> : tensor<1x1x40x40xi1>}> : () -> tensor<1x1x40x40xi1>
    %5 = "mix.comp.weight"() <{param_loc = "transformer.h.0.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %6 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %7 = "mix.prim.pow"(%3, %6) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %8 = "mix.comp.mean"(%7) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %9 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %10 = "mix.prim.add"(%8, %9) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %11 = "mix.prim.rsqrt"(%10) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %12 = "mix.prim.mul"(%3, %11) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %13 = "mix.prim.mul"(%5, %12) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %14 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %15 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %16 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %17 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %18 = "mix.prim.transpose"(%13) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %19 = "mix.prim.transpose"(%14) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %20 = "mix.prim.reshape"(%18) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %21 = "mix.prim.matmul"(%20, %19) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %22 = "mix.prim.reshape"(%21) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %23 = "mix.prim.reshape"(%22) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %24 = "mix.prim.transpose"(%15) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %25 = "mix.prim.reshape"(%18) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %26 = "mix.prim.matmul"(%25, %24) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %27 = "mix.prim.reshape"(%26) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %28 = "mix.prim.reshape"(%27) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %29 = "mix.prim.slice"(%28) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %30 = "mix.prim.slice"(%28) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %31 = "mix.prim.reshape"(%23) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %32 = "mix.prim.reshape"(%29) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %33 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %34 = "mix.prim.convert"(%33) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %35 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %36 = "mix.prim.div"(%34, %35) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %37 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %38 = "mix.prim.pow"(%37, %36) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %39 = "mix.prim.reciprocal"(%38) : (tensor<80xf16>) -> tensor<80xf16>
    %40 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %41 = "mix.prim.mul"(%40, %39) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %42 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %43 = "mix.prim.unsqueeze"(%42) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %44 = "mix.prim.permute"(%43) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %45 = "mix.prim.unsqueeze"(%41) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %46 = "mix.prim.permute"(%45) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %47 = "mix.prim.mul"(%44, %46) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %48 = "mix.prim.concat"(%47, %47) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %49 = "mix.prim.cos"(%48) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %50 = "mix.prim.slice"(%49) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %51 = "mix.prim.unsqueeze"(%50) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %52 = "mix.prim.slice"(%51) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %53 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %54 = "mix.prim.mul"(%52, %53) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %55 = "mix.prim.sin"(%48) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %56 = "mix.prim.slice"(%55) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %57 = "mix.prim.unsqueeze"(%56) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %58 = "mix.prim.slice"(%57) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %59 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %60 = "mix.prim.mul"(%58, %59) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %61 = "mix.prim.slice"(%54) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %62 = "mix.prim.slice"(%60) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %63 = "mix.prim.mul"(%31, %61) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %64 = "mix.prim.slice"(%31) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %65 = "mix.prim.slice"(%31) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %66 = "mix.prim.neg"(%65) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %67 = "mix.prim.concat"(%66, %64) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %68 = "mix.prim.mul"(%67, %62) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %69 = "mix.prim.add"(%63, %68) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %70 = "mix.prim.mul"(%32, %61) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %71 = "mix.prim.slice"(%32) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %72 = "mix.prim.slice"(%32) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %73 = "mix.prim.neg"(%72) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %74 = "mix.prim.concat"(%73, %71) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %75 = "mix.prim.mul"(%74, %62) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %76 = "mix.prim.add"(%70, %75) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %77 = "mix.prim.reshape"(%69) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %78 = "mix.prim.reshape"(%76) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %79 = "mix.prim.reshape"(%77) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %80 = "mix.prim.reshape"(%78) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %81 = "mix.prim.transpose"(%79) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %82 = "mix.prim.transpose"(%80) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %83 = "mix.prim.transpose"(%82) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %84 = "mix.prim.unsqueeze"(%81) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %85 = "mix.prim.permute"(%84) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %86 = "mix.prim.unsqueeze"(%83) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %87 = "mix.prim.permute"(%86) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %88 = "mix.prim.permute"(%85) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %89 = "mix.prim.reshape"(%88) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %90 = "mix.prim.permute"(%87) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %91 = "mix.prim.reshape"(%90) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %92 = "mix.prim.batch_matmul"(%89, %91) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %93 = "mix.prim.reshape"(%92) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %94 = "mix.prim.permute"(%93) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %95 = "mix.prim.reshape"(%94) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %96 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %97 = "mix.prim.mul"(%95, %96) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %98 = "mix.prim.reshape"(%97) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %99 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %100 = "mix.comp.masked_fill"(%98, %4, %99) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %101 = "mix.comp.softmax"(%100) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %102 = "mix.prim.reshape"(%101) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %103 = "mix.prim.reshape"(%30) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %104 = "mix.prim.transpose"(%103) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %105 = "mix.prim.batch_matmul"(%102, %104) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %106 = "mix.prim.reshape"(%105) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %107 = "mix.prim.permute"(%106) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %108 = "mix.prim.reshape"(%107) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %109 = "mix.prim.reshape"(%108) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %110 = "mix.prim.transpose"(%16) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %111 = "mix.prim.matmul"(%109, %110) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %112 = "mix.prim.add"(%111, %17) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %113 = "mix.prim.reshape"(%112) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %114 = "mix.prim.mul"(%3, %113) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %115 = "mix.comp.weight"() <{param_loc = "transformer.h.0.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %116 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %117 = "mix.prim.pow"(%114, %116) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %118 = "mix.comp.mean"(%117) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %119 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %120 = "mix.prim.add"(%118, %119) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %121 = "mix.prim.rsqrt"(%120) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %122 = "mix.prim.mul"(%114, %121) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %123 = "mix.prim.mul"(%115, %122) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %124 = "mix.comp.weight"() <{param_loc = "transformer.h.0.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %125 = "mix.prim.reshape"(%124) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %126 = "mix.prim.batch_matmul"(%123, %125) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %127 = "mix.comp.silu"(%126) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %128 = "mix.comp.weight"() <{param_loc = "transformer.h.0.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %129 = "mix.prim.reshape"(%128) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %130 = "mix.prim.batch_matmul"(%123, %129) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %131 = "mix.prim.mul"(%127, %130) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %132 = "mix.comp.weight"() <{param_loc = "transformer.h.0.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %133 = "mix.prim.reshape"(%132) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %134 = "mix.prim.batch_matmul"(%131, %133) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %135 = "mix.comp.weight"() <{param_loc = "transformer.h.0.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %136 = "mix.prim.add"(%134, %135) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %137 = "mix.prim.add"(%136, %114) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %138 = "mix.comp.weight"() <{param_loc = "transformer.h.1.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %139 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %140 = "mix.prim.pow"(%137, %139) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %141 = "mix.comp.mean"(%140) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %142 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %143 = "mix.prim.add"(%141, %142) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %144 = "mix.prim.rsqrt"(%143) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %145 = "mix.prim.mul"(%137, %144) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %146 = "mix.prim.mul"(%138, %145) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %147 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %148 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %149 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %150 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %151 = "mix.prim.transpose"(%146) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %152 = "mix.prim.transpose"(%147) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %153 = "mix.prim.reshape"(%151) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %154 = "mix.prim.matmul"(%153, %152) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %155 = "mix.prim.reshape"(%154) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %156 = "mix.prim.reshape"(%155) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %157 = "mix.prim.transpose"(%148) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %158 = "mix.prim.reshape"(%151) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %159 = "mix.prim.matmul"(%158, %157) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %160 = "mix.prim.reshape"(%159) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %161 = "mix.prim.reshape"(%160) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %162 = "mix.prim.slice"(%161) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %163 = "mix.prim.slice"(%161) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %164 = "mix.prim.reshape"(%156) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %165 = "mix.prim.reshape"(%162) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %166 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %167 = "mix.prim.convert"(%166) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %168 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %169 = "mix.prim.div"(%167, %168) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %170 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %171 = "mix.prim.pow"(%170, %169) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %172 = "mix.prim.reciprocal"(%171) : (tensor<80xf16>) -> tensor<80xf16>
    %173 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %174 = "mix.prim.mul"(%173, %172) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %175 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %176 = "mix.prim.unsqueeze"(%175) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %177 = "mix.prim.permute"(%176) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %178 = "mix.prim.unsqueeze"(%174) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %179 = "mix.prim.permute"(%178) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %180 = "mix.prim.mul"(%177, %179) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %181 = "mix.prim.concat"(%180, %180) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %182 = "mix.prim.cos"(%181) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %183 = "mix.prim.slice"(%182) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %184 = "mix.prim.unsqueeze"(%183) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %185 = "mix.prim.slice"(%184) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %186 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %187 = "mix.prim.mul"(%185, %186) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %188 = "mix.prim.sin"(%181) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %189 = "mix.prim.slice"(%188) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %190 = "mix.prim.unsqueeze"(%189) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %191 = "mix.prim.slice"(%190) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %192 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %193 = "mix.prim.mul"(%191, %192) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %194 = "mix.prim.slice"(%187) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %195 = "mix.prim.slice"(%193) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %196 = "mix.prim.mul"(%164, %194) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %197 = "mix.prim.slice"(%164) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %198 = "mix.prim.slice"(%164) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %199 = "mix.prim.neg"(%198) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %200 = "mix.prim.concat"(%199, %197) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %201 = "mix.prim.mul"(%200, %195) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %202 = "mix.prim.add"(%196, %201) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %203 = "mix.prim.mul"(%165, %194) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %204 = "mix.prim.slice"(%165) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %205 = "mix.prim.slice"(%165) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %206 = "mix.prim.neg"(%205) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %207 = "mix.prim.concat"(%206, %204) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %208 = "mix.prim.mul"(%207, %195) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %209 = "mix.prim.add"(%203, %208) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %210 = "mix.prim.reshape"(%202) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %211 = "mix.prim.reshape"(%209) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %212 = "mix.prim.reshape"(%210) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %213 = "mix.prim.reshape"(%211) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %214 = "mix.prim.transpose"(%212) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %215 = "mix.prim.transpose"(%213) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %216 = "mix.prim.transpose"(%215) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %217 = "mix.prim.unsqueeze"(%214) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %218 = "mix.prim.permute"(%217) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %219 = "mix.prim.unsqueeze"(%216) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %220 = "mix.prim.permute"(%219) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %221 = "mix.prim.permute"(%218) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %222 = "mix.prim.reshape"(%221) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %223 = "mix.prim.permute"(%220) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %224 = "mix.prim.reshape"(%223) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %225 = "mix.prim.batch_matmul"(%222, %224) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %226 = "mix.prim.reshape"(%225) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %227 = "mix.prim.permute"(%226) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %228 = "mix.prim.reshape"(%227) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %229 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %230 = "mix.prim.mul"(%228, %229) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %231 = "mix.prim.reshape"(%230) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %232 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %233 = "mix.comp.masked_fill"(%231, %4, %232) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %234 = "mix.comp.softmax"(%233) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %235 = "mix.prim.reshape"(%234) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %236 = "mix.prim.reshape"(%163) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %237 = "mix.prim.transpose"(%236) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %238 = "mix.prim.batch_matmul"(%235, %237) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %239 = "mix.prim.reshape"(%238) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %240 = "mix.prim.permute"(%239) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %241 = "mix.prim.reshape"(%240) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %242 = "mix.prim.reshape"(%241) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %243 = "mix.prim.transpose"(%149) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %244 = "mix.prim.matmul"(%242, %243) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %245 = "mix.prim.add"(%244, %150) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %246 = "mix.prim.reshape"(%245) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %247 = "mix.prim.mul"(%137, %246) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %248 = "mix.comp.weight"() <{param_loc = "transformer.h.1.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %249 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %250 = "mix.prim.pow"(%247, %249) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %251 = "mix.comp.mean"(%250) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %252 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %253 = "mix.prim.add"(%251, %252) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %254 = "mix.prim.rsqrt"(%253) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %255 = "mix.prim.mul"(%247, %254) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %256 = "mix.prim.mul"(%248, %255) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %257 = "mix.comp.weight"() <{param_loc = "transformer.h.1.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %258 = "mix.prim.reshape"(%257) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %259 = "mix.prim.batch_matmul"(%256, %258) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %260 = "mix.comp.silu"(%259) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %261 = "mix.comp.weight"() <{param_loc = "transformer.h.1.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %262 = "mix.prim.reshape"(%261) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %263 = "mix.prim.batch_matmul"(%256, %262) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %264 = "mix.prim.mul"(%260, %263) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %265 = "mix.comp.weight"() <{param_loc = "transformer.h.1.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %266 = "mix.prim.reshape"(%265) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %267 = "mix.prim.batch_matmul"(%264, %266) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %268 = "mix.comp.weight"() <{param_loc = "transformer.h.1.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %269 = "mix.prim.add"(%267, %268) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %270 = "mix.prim.add"(%269, %247) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %271 = "mix.comp.weight"() <{param_loc = "transformer.h.2.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %272 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %273 = "mix.prim.pow"(%270, %272) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %274 = "mix.comp.mean"(%273) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %275 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %276 = "mix.prim.add"(%274, %275) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %277 = "mix.prim.rsqrt"(%276) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %278 = "mix.prim.mul"(%270, %277) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %279 = "mix.prim.mul"(%271, %278) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %280 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %281 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %282 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %283 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %284 = "mix.prim.transpose"(%279) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %285 = "mix.prim.transpose"(%280) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %286 = "mix.prim.reshape"(%284) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %287 = "mix.prim.matmul"(%286, %285) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %288 = "mix.prim.reshape"(%287) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %289 = "mix.prim.reshape"(%288) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %290 = "mix.prim.transpose"(%281) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %291 = "mix.prim.reshape"(%284) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %292 = "mix.prim.matmul"(%291, %290) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %293 = "mix.prim.reshape"(%292) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %294 = "mix.prim.reshape"(%293) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %295 = "mix.prim.slice"(%294) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %296 = "mix.prim.slice"(%294) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %297 = "mix.prim.reshape"(%289) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %298 = "mix.prim.reshape"(%295) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %299 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %300 = "mix.prim.convert"(%299) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %301 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %302 = "mix.prim.div"(%300, %301) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %303 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %304 = "mix.prim.pow"(%303, %302) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %305 = "mix.prim.reciprocal"(%304) : (tensor<80xf16>) -> tensor<80xf16>
    %306 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %307 = "mix.prim.mul"(%306, %305) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %308 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %309 = "mix.prim.unsqueeze"(%308) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %310 = "mix.prim.permute"(%309) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %311 = "mix.prim.unsqueeze"(%307) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %312 = "mix.prim.permute"(%311) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %313 = "mix.prim.mul"(%310, %312) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %314 = "mix.prim.concat"(%313, %313) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %315 = "mix.prim.cos"(%314) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %316 = "mix.prim.slice"(%315) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %317 = "mix.prim.unsqueeze"(%316) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %318 = "mix.prim.slice"(%317) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %319 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %320 = "mix.prim.mul"(%318, %319) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %321 = "mix.prim.sin"(%314) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %322 = "mix.prim.slice"(%321) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %323 = "mix.prim.unsqueeze"(%322) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %324 = "mix.prim.slice"(%323) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %325 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %326 = "mix.prim.mul"(%324, %325) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %327 = "mix.prim.slice"(%320) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %328 = "mix.prim.slice"(%326) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %329 = "mix.prim.mul"(%297, %327) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %330 = "mix.prim.slice"(%297) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %331 = "mix.prim.slice"(%297) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %332 = "mix.prim.neg"(%331) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %333 = "mix.prim.concat"(%332, %330) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %334 = "mix.prim.mul"(%333, %328) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %335 = "mix.prim.add"(%329, %334) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %336 = "mix.prim.mul"(%298, %327) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %337 = "mix.prim.slice"(%298) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %338 = "mix.prim.slice"(%298) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %339 = "mix.prim.neg"(%338) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %340 = "mix.prim.concat"(%339, %337) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %341 = "mix.prim.mul"(%340, %328) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %342 = "mix.prim.add"(%336, %341) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %343 = "mix.prim.reshape"(%335) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %344 = "mix.prim.reshape"(%342) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %345 = "mix.prim.reshape"(%343) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %346 = "mix.prim.reshape"(%344) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %347 = "mix.prim.transpose"(%345) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %348 = "mix.prim.transpose"(%346) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %349 = "mix.prim.transpose"(%348) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %350 = "mix.prim.unsqueeze"(%347) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %351 = "mix.prim.permute"(%350) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %352 = "mix.prim.unsqueeze"(%349) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %353 = "mix.prim.permute"(%352) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %354 = "mix.prim.permute"(%351) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %355 = "mix.prim.reshape"(%354) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %356 = "mix.prim.permute"(%353) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %357 = "mix.prim.reshape"(%356) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %358 = "mix.prim.batch_matmul"(%355, %357) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %359 = "mix.prim.reshape"(%358) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %360 = "mix.prim.permute"(%359) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %361 = "mix.prim.reshape"(%360) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %362 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %363 = "mix.prim.mul"(%361, %362) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %364 = "mix.prim.reshape"(%363) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %365 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %366 = "mix.comp.masked_fill"(%364, %4, %365) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %367 = "mix.comp.softmax"(%366) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %368 = "mix.prim.reshape"(%367) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %369 = "mix.prim.reshape"(%296) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %370 = "mix.prim.transpose"(%369) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %371 = "mix.prim.batch_matmul"(%368, %370) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %372 = "mix.prim.reshape"(%371) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %373 = "mix.prim.permute"(%372) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %374 = "mix.prim.reshape"(%373) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %375 = "mix.prim.reshape"(%374) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %376 = "mix.prim.transpose"(%282) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %377 = "mix.prim.matmul"(%375, %376) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %378 = "mix.prim.add"(%377, %283) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %379 = "mix.prim.reshape"(%378) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %380 = "mix.prim.mul"(%270, %379) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %381 = "mix.comp.weight"() <{param_loc = "transformer.h.2.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %382 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %383 = "mix.prim.pow"(%380, %382) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %384 = "mix.comp.mean"(%383) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %385 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %386 = "mix.prim.add"(%384, %385) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %387 = "mix.prim.rsqrt"(%386) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %388 = "mix.prim.mul"(%380, %387) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %389 = "mix.prim.mul"(%381, %388) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %390 = "mix.comp.weight"() <{param_loc = "transformer.h.2.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %391 = "mix.prim.reshape"(%390) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %392 = "mix.prim.batch_matmul"(%389, %391) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %393 = "mix.comp.silu"(%392) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %394 = "mix.comp.weight"() <{param_loc = "transformer.h.2.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %395 = "mix.prim.reshape"(%394) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %396 = "mix.prim.batch_matmul"(%389, %395) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %397 = "mix.prim.mul"(%393, %396) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %398 = "mix.comp.weight"() <{param_loc = "transformer.h.2.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %399 = "mix.prim.reshape"(%398) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %400 = "mix.prim.batch_matmul"(%397, %399) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %401 = "mix.comp.weight"() <{param_loc = "transformer.h.2.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %402 = "mix.prim.add"(%400, %401) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %403 = "mix.prim.add"(%402, %380) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %404 = "mix.comp.weight"() <{param_loc = "transformer.h.3.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %405 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %406 = "mix.prim.pow"(%403, %405) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %407 = "mix.comp.mean"(%406) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %408 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %409 = "mix.prim.add"(%407, %408) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %410 = "mix.prim.rsqrt"(%409) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %411 = "mix.prim.mul"(%403, %410) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %412 = "mix.prim.mul"(%404, %411) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %413 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %414 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %415 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %416 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %417 = "mix.prim.transpose"(%412) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %418 = "mix.prim.transpose"(%413) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %419 = "mix.prim.reshape"(%417) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %420 = "mix.prim.matmul"(%419, %418) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %421 = "mix.prim.reshape"(%420) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %422 = "mix.prim.reshape"(%421) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %423 = "mix.prim.transpose"(%414) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %424 = "mix.prim.reshape"(%417) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %425 = "mix.prim.matmul"(%424, %423) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %426 = "mix.prim.reshape"(%425) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %427 = "mix.prim.reshape"(%426) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %428 = "mix.prim.slice"(%427) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %429 = "mix.prim.slice"(%427) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %430 = "mix.prim.reshape"(%422) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %431 = "mix.prim.reshape"(%428) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %432 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %433 = "mix.prim.convert"(%432) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %434 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %435 = "mix.prim.div"(%433, %434) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %436 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %437 = "mix.prim.pow"(%436, %435) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %438 = "mix.prim.reciprocal"(%437) : (tensor<80xf16>) -> tensor<80xf16>
    %439 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %440 = "mix.prim.mul"(%439, %438) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %441 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %442 = "mix.prim.unsqueeze"(%441) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %443 = "mix.prim.permute"(%442) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %444 = "mix.prim.unsqueeze"(%440) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %445 = "mix.prim.permute"(%444) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %446 = "mix.prim.mul"(%443, %445) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %447 = "mix.prim.concat"(%446, %446) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %448 = "mix.prim.cos"(%447) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %449 = "mix.prim.slice"(%448) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %450 = "mix.prim.unsqueeze"(%449) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %451 = "mix.prim.slice"(%450) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %452 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %453 = "mix.prim.mul"(%451, %452) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %454 = "mix.prim.sin"(%447) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %455 = "mix.prim.slice"(%454) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %456 = "mix.prim.unsqueeze"(%455) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %457 = "mix.prim.slice"(%456) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %458 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %459 = "mix.prim.mul"(%457, %458) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %460 = "mix.prim.slice"(%453) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %461 = "mix.prim.slice"(%459) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %462 = "mix.prim.mul"(%430, %460) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %463 = "mix.prim.slice"(%430) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %464 = "mix.prim.slice"(%430) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %465 = "mix.prim.neg"(%464) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %466 = "mix.prim.concat"(%465, %463) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %467 = "mix.prim.mul"(%466, %461) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %468 = "mix.prim.add"(%462, %467) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %469 = "mix.prim.mul"(%431, %460) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %470 = "mix.prim.slice"(%431) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %471 = "mix.prim.slice"(%431) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %472 = "mix.prim.neg"(%471) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %473 = "mix.prim.concat"(%472, %470) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %474 = "mix.prim.mul"(%473, %461) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %475 = "mix.prim.add"(%469, %474) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %476 = "mix.prim.reshape"(%468) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %477 = "mix.prim.reshape"(%475) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %478 = "mix.prim.reshape"(%476) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %479 = "mix.prim.reshape"(%477) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %480 = "mix.prim.transpose"(%478) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %481 = "mix.prim.transpose"(%479) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %482 = "mix.prim.transpose"(%481) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %483 = "mix.prim.unsqueeze"(%480) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %484 = "mix.prim.permute"(%483) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %485 = "mix.prim.unsqueeze"(%482) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %486 = "mix.prim.permute"(%485) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %487 = "mix.prim.permute"(%484) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %488 = "mix.prim.reshape"(%487) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %489 = "mix.prim.permute"(%486) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %490 = "mix.prim.reshape"(%489) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %491 = "mix.prim.batch_matmul"(%488, %490) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %492 = "mix.prim.reshape"(%491) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %493 = "mix.prim.permute"(%492) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %494 = "mix.prim.reshape"(%493) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %495 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %496 = "mix.prim.mul"(%494, %495) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %497 = "mix.prim.reshape"(%496) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %498 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %499 = "mix.comp.masked_fill"(%497, %4, %498) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %500 = "mix.comp.softmax"(%499) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %501 = "mix.prim.reshape"(%500) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %502 = "mix.prim.reshape"(%429) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %503 = "mix.prim.transpose"(%502) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %504 = "mix.prim.batch_matmul"(%501, %503) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %505 = "mix.prim.reshape"(%504) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %506 = "mix.prim.permute"(%505) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %507 = "mix.prim.reshape"(%506) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %508 = "mix.prim.reshape"(%507) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %509 = "mix.prim.transpose"(%415) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %510 = "mix.prim.matmul"(%508, %509) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %511 = "mix.prim.add"(%510, %416) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %512 = "mix.prim.reshape"(%511) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %513 = "mix.prim.mul"(%403, %512) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %514 = "mix.comp.weight"() <{param_loc = "transformer.h.3.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %515 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %516 = "mix.prim.pow"(%513, %515) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %517 = "mix.comp.mean"(%516) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %518 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %519 = "mix.prim.add"(%517, %518) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %520 = "mix.prim.rsqrt"(%519) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %521 = "mix.prim.mul"(%513, %520) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %522 = "mix.prim.mul"(%514, %521) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %523 = "mix.comp.weight"() <{param_loc = "transformer.h.3.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %524 = "mix.prim.reshape"(%523) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %525 = "mix.prim.batch_matmul"(%522, %524) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %526 = "mix.comp.silu"(%525) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %527 = "mix.comp.weight"() <{param_loc = "transformer.h.3.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %528 = "mix.prim.reshape"(%527) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %529 = "mix.prim.batch_matmul"(%522, %528) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %530 = "mix.prim.mul"(%526, %529) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %531 = "mix.comp.weight"() <{param_loc = "transformer.h.3.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %532 = "mix.prim.reshape"(%531) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %533 = "mix.prim.batch_matmul"(%530, %532) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %534 = "mix.comp.weight"() <{param_loc = "transformer.h.3.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %535 = "mix.prim.add"(%533, %534) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %536 = "mix.prim.add"(%535, %513) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %537 = "mix.comp.weight"() <{param_loc = "transformer.h.4.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %538 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %539 = "mix.prim.pow"(%536, %538) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %540 = "mix.comp.mean"(%539) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %541 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %542 = "mix.prim.add"(%540, %541) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %543 = "mix.prim.rsqrt"(%542) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %544 = "mix.prim.mul"(%536, %543) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %545 = "mix.prim.mul"(%537, %544) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %546 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %547 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %548 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %549 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %550 = "mix.prim.transpose"(%545) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %551 = "mix.prim.transpose"(%546) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %552 = "mix.prim.reshape"(%550) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %553 = "mix.prim.matmul"(%552, %551) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %554 = "mix.prim.reshape"(%553) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %555 = "mix.prim.reshape"(%554) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %556 = "mix.prim.transpose"(%547) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %557 = "mix.prim.reshape"(%550) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %558 = "mix.prim.matmul"(%557, %556) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %559 = "mix.prim.reshape"(%558) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %560 = "mix.prim.reshape"(%559) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %561 = "mix.prim.slice"(%560) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %562 = "mix.prim.slice"(%560) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %563 = "mix.prim.reshape"(%555) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %564 = "mix.prim.reshape"(%561) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %565 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %566 = "mix.prim.convert"(%565) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %567 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %568 = "mix.prim.div"(%566, %567) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %569 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %570 = "mix.prim.pow"(%569, %568) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %571 = "mix.prim.reciprocal"(%570) : (tensor<80xf16>) -> tensor<80xf16>
    %572 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %573 = "mix.prim.mul"(%572, %571) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %574 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %575 = "mix.prim.unsqueeze"(%574) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %576 = "mix.prim.permute"(%575) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %577 = "mix.prim.unsqueeze"(%573) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %578 = "mix.prim.permute"(%577) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %579 = "mix.prim.mul"(%576, %578) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %580 = "mix.prim.concat"(%579, %579) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %581 = "mix.prim.cos"(%580) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %582 = "mix.prim.slice"(%581) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %583 = "mix.prim.unsqueeze"(%582) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %584 = "mix.prim.slice"(%583) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %585 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %586 = "mix.prim.mul"(%584, %585) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %587 = "mix.prim.sin"(%580) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %588 = "mix.prim.slice"(%587) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %589 = "mix.prim.unsqueeze"(%588) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %590 = "mix.prim.slice"(%589) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %591 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %592 = "mix.prim.mul"(%590, %591) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %593 = "mix.prim.slice"(%586) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %594 = "mix.prim.slice"(%592) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %595 = "mix.prim.mul"(%563, %593) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %596 = "mix.prim.slice"(%563) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %597 = "mix.prim.slice"(%563) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %598 = "mix.prim.neg"(%597) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %599 = "mix.prim.concat"(%598, %596) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %600 = "mix.prim.mul"(%599, %594) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %601 = "mix.prim.add"(%595, %600) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %602 = "mix.prim.mul"(%564, %593) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %603 = "mix.prim.slice"(%564) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %604 = "mix.prim.slice"(%564) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %605 = "mix.prim.neg"(%604) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %606 = "mix.prim.concat"(%605, %603) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %607 = "mix.prim.mul"(%606, %594) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %608 = "mix.prim.add"(%602, %607) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %609 = "mix.prim.reshape"(%601) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %610 = "mix.prim.reshape"(%608) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %611 = "mix.prim.reshape"(%609) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %612 = "mix.prim.reshape"(%610) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %613 = "mix.prim.transpose"(%611) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %614 = "mix.prim.transpose"(%612) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %615 = "mix.prim.transpose"(%614) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %616 = "mix.prim.unsqueeze"(%613) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %617 = "mix.prim.permute"(%616) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %618 = "mix.prim.unsqueeze"(%615) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %619 = "mix.prim.permute"(%618) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %620 = "mix.prim.permute"(%617) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %621 = "mix.prim.reshape"(%620) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %622 = "mix.prim.permute"(%619) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %623 = "mix.prim.reshape"(%622) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %624 = "mix.prim.batch_matmul"(%621, %623) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %625 = "mix.prim.reshape"(%624) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %626 = "mix.prim.permute"(%625) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %627 = "mix.prim.reshape"(%626) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %628 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %629 = "mix.prim.mul"(%627, %628) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %630 = "mix.prim.reshape"(%629) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %631 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %632 = "mix.comp.masked_fill"(%630, %4, %631) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %633 = "mix.comp.softmax"(%632) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %634 = "mix.prim.reshape"(%633) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %635 = "mix.prim.reshape"(%562) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %636 = "mix.prim.transpose"(%635) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %637 = "mix.prim.batch_matmul"(%634, %636) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %638 = "mix.prim.reshape"(%637) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %639 = "mix.prim.permute"(%638) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %640 = "mix.prim.reshape"(%639) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %641 = "mix.prim.reshape"(%640) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %642 = "mix.prim.transpose"(%548) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %643 = "mix.prim.matmul"(%641, %642) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %644 = "mix.prim.add"(%643, %549) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %645 = "mix.prim.reshape"(%644) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %646 = "mix.prim.mul"(%536, %645) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %647 = "mix.comp.weight"() <{param_loc = "transformer.h.4.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %648 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %649 = "mix.prim.pow"(%646, %648) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %650 = "mix.comp.mean"(%649) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %651 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %652 = "mix.prim.add"(%650, %651) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %653 = "mix.prim.rsqrt"(%652) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %654 = "mix.prim.mul"(%646, %653) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %655 = "mix.prim.mul"(%647, %654) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %656 = "mix.comp.weight"() <{param_loc = "transformer.h.4.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %657 = "mix.prim.reshape"(%656) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %658 = "mix.prim.batch_matmul"(%655, %657) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %659 = "mix.comp.silu"(%658) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %660 = "mix.comp.weight"() <{param_loc = "transformer.h.4.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %661 = "mix.prim.reshape"(%660) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %662 = "mix.prim.batch_matmul"(%655, %661) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %663 = "mix.prim.mul"(%659, %662) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %664 = "mix.comp.weight"() <{param_loc = "transformer.h.4.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %665 = "mix.prim.reshape"(%664) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %666 = "mix.prim.batch_matmul"(%663, %665) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %667 = "mix.comp.weight"() <{param_loc = "transformer.h.4.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %668 = "mix.prim.add"(%666, %667) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %669 = "mix.prim.add"(%668, %646) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %670 = "mix.comp.weight"() <{param_loc = "transformer.h.5.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %671 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %672 = "mix.prim.pow"(%669, %671) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %673 = "mix.comp.mean"(%672) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %674 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %675 = "mix.prim.add"(%673, %674) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %676 = "mix.prim.rsqrt"(%675) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %677 = "mix.prim.mul"(%669, %676) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %678 = "mix.prim.mul"(%670, %677) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %679 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %680 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %681 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %682 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %683 = "mix.prim.transpose"(%678) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %684 = "mix.prim.transpose"(%679) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %685 = "mix.prim.reshape"(%683) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %686 = "mix.prim.matmul"(%685, %684) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %687 = "mix.prim.reshape"(%686) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %688 = "mix.prim.reshape"(%687) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %689 = "mix.prim.transpose"(%680) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %690 = "mix.prim.reshape"(%683) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %691 = "mix.prim.matmul"(%690, %689) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %692 = "mix.prim.reshape"(%691) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %693 = "mix.prim.reshape"(%692) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %694 = "mix.prim.slice"(%693) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %695 = "mix.prim.slice"(%693) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %696 = "mix.prim.reshape"(%688) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %697 = "mix.prim.reshape"(%694) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %698 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %699 = "mix.prim.convert"(%698) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %700 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %701 = "mix.prim.div"(%699, %700) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %702 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %703 = "mix.prim.pow"(%702, %701) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %704 = "mix.prim.reciprocal"(%703) : (tensor<80xf16>) -> tensor<80xf16>
    %705 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %706 = "mix.prim.mul"(%705, %704) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %707 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %708 = "mix.prim.unsqueeze"(%707) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %709 = "mix.prim.permute"(%708) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %710 = "mix.prim.unsqueeze"(%706) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %711 = "mix.prim.permute"(%710) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %712 = "mix.prim.mul"(%709, %711) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %713 = "mix.prim.concat"(%712, %712) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %714 = "mix.prim.cos"(%713) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %715 = "mix.prim.slice"(%714) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %716 = "mix.prim.unsqueeze"(%715) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %717 = "mix.prim.slice"(%716) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %718 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %719 = "mix.prim.mul"(%717, %718) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %720 = "mix.prim.sin"(%713) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %721 = "mix.prim.slice"(%720) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %722 = "mix.prim.unsqueeze"(%721) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %723 = "mix.prim.slice"(%722) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %724 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %725 = "mix.prim.mul"(%723, %724) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %726 = "mix.prim.slice"(%719) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %727 = "mix.prim.slice"(%725) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %728 = "mix.prim.mul"(%696, %726) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %729 = "mix.prim.slice"(%696) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %730 = "mix.prim.slice"(%696) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %731 = "mix.prim.neg"(%730) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %732 = "mix.prim.concat"(%731, %729) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %733 = "mix.prim.mul"(%732, %727) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %734 = "mix.prim.add"(%728, %733) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %735 = "mix.prim.mul"(%697, %726) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %736 = "mix.prim.slice"(%697) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %737 = "mix.prim.slice"(%697) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %738 = "mix.prim.neg"(%737) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %739 = "mix.prim.concat"(%738, %736) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %740 = "mix.prim.mul"(%739, %727) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %741 = "mix.prim.add"(%735, %740) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %742 = "mix.prim.reshape"(%734) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %743 = "mix.prim.reshape"(%741) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %744 = "mix.prim.reshape"(%742) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %745 = "mix.prim.reshape"(%743) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %746 = "mix.prim.transpose"(%744) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %747 = "mix.prim.transpose"(%745) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %748 = "mix.prim.transpose"(%747) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %749 = "mix.prim.unsqueeze"(%746) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %750 = "mix.prim.permute"(%749) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %751 = "mix.prim.unsqueeze"(%748) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %752 = "mix.prim.permute"(%751) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %753 = "mix.prim.permute"(%750) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %754 = "mix.prim.reshape"(%753) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %755 = "mix.prim.permute"(%752) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %756 = "mix.prim.reshape"(%755) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %757 = "mix.prim.batch_matmul"(%754, %756) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %758 = "mix.prim.reshape"(%757) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %759 = "mix.prim.permute"(%758) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %760 = "mix.prim.reshape"(%759) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %761 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %762 = "mix.prim.mul"(%760, %761) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %763 = "mix.prim.reshape"(%762) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %764 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %765 = "mix.comp.masked_fill"(%763, %4, %764) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %766 = "mix.comp.softmax"(%765) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %767 = "mix.prim.reshape"(%766) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %768 = "mix.prim.reshape"(%695) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %769 = "mix.prim.transpose"(%768) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %770 = "mix.prim.batch_matmul"(%767, %769) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %771 = "mix.prim.reshape"(%770) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %772 = "mix.prim.permute"(%771) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %773 = "mix.prim.reshape"(%772) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %774 = "mix.prim.reshape"(%773) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %775 = "mix.prim.transpose"(%681) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %776 = "mix.prim.matmul"(%774, %775) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %777 = "mix.prim.add"(%776, %682) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %778 = "mix.prim.reshape"(%777) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %779 = "mix.prim.mul"(%669, %778) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %780 = "mix.comp.weight"() <{param_loc = "transformer.h.5.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %781 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %782 = "mix.prim.pow"(%779, %781) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %783 = "mix.comp.mean"(%782) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %784 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %785 = "mix.prim.add"(%783, %784) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %786 = "mix.prim.rsqrt"(%785) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %787 = "mix.prim.mul"(%779, %786) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %788 = "mix.prim.mul"(%780, %787) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %789 = "mix.comp.weight"() <{param_loc = "transformer.h.5.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %790 = "mix.prim.reshape"(%789) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %791 = "mix.prim.batch_matmul"(%788, %790) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %792 = "mix.comp.silu"(%791) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %793 = "mix.comp.weight"() <{param_loc = "transformer.h.5.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %794 = "mix.prim.reshape"(%793) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %795 = "mix.prim.batch_matmul"(%788, %794) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %796 = "mix.prim.mul"(%792, %795) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %797 = "mix.comp.weight"() <{param_loc = "transformer.h.5.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %798 = "mix.prim.reshape"(%797) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %799 = "mix.prim.batch_matmul"(%796, %798) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %800 = "mix.comp.weight"() <{param_loc = "transformer.h.5.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %801 = "mix.prim.add"(%799, %800) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %802 = "mix.prim.add"(%801, %779) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %803 = "mix.comp.weight"() <{param_loc = "transformer.h.6.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %804 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %805 = "mix.prim.pow"(%802, %804) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %806 = "mix.comp.mean"(%805) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %807 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %808 = "mix.prim.add"(%806, %807) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %809 = "mix.prim.rsqrt"(%808) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %810 = "mix.prim.mul"(%802, %809) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %811 = "mix.prim.mul"(%803, %810) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %812 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %813 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %814 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %815 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %816 = "mix.prim.transpose"(%811) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %817 = "mix.prim.transpose"(%812) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %818 = "mix.prim.reshape"(%816) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %819 = "mix.prim.matmul"(%818, %817) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %820 = "mix.prim.reshape"(%819) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %821 = "mix.prim.reshape"(%820) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %822 = "mix.prim.transpose"(%813) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %823 = "mix.prim.reshape"(%816) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %824 = "mix.prim.matmul"(%823, %822) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %825 = "mix.prim.reshape"(%824) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %826 = "mix.prim.reshape"(%825) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %827 = "mix.prim.slice"(%826) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %828 = "mix.prim.slice"(%826) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %829 = "mix.prim.reshape"(%821) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %830 = "mix.prim.reshape"(%827) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %831 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %832 = "mix.prim.convert"(%831) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %833 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %834 = "mix.prim.div"(%832, %833) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %835 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %836 = "mix.prim.pow"(%835, %834) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %837 = "mix.prim.reciprocal"(%836) : (tensor<80xf16>) -> tensor<80xf16>
    %838 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %839 = "mix.prim.mul"(%838, %837) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %840 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %841 = "mix.prim.unsqueeze"(%840) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %842 = "mix.prim.permute"(%841) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %843 = "mix.prim.unsqueeze"(%839) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %844 = "mix.prim.permute"(%843) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %845 = "mix.prim.mul"(%842, %844) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %846 = "mix.prim.concat"(%845, %845) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %847 = "mix.prim.cos"(%846) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %848 = "mix.prim.slice"(%847) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %849 = "mix.prim.unsqueeze"(%848) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %850 = "mix.prim.slice"(%849) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %851 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %852 = "mix.prim.mul"(%850, %851) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %853 = "mix.prim.sin"(%846) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %854 = "mix.prim.slice"(%853) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %855 = "mix.prim.unsqueeze"(%854) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %856 = "mix.prim.slice"(%855) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %857 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %858 = "mix.prim.mul"(%856, %857) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %859 = "mix.prim.slice"(%852) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %860 = "mix.prim.slice"(%858) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %861 = "mix.prim.mul"(%829, %859) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %862 = "mix.prim.slice"(%829) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %863 = "mix.prim.slice"(%829) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %864 = "mix.prim.neg"(%863) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %865 = "mix.prim.concat"(%864, %862) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %866 = "mix.prim.mul"(%865, %860) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %867 = "mix.prim.add"(%861, %866) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %868 = "mix.prim.mul"(%830, %859) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %869 = "mix.prim.slice"(%830) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %870 = "mix.prim.slice"(%830) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %871 = "mix.prim.neg"(%870) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %872 = "mix.prim.concat"(%871, %869) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %873 = "mix.prim.mul"(%872, %860) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %874 = "mix.prim.add"(%868, %873) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %875 = "mix.prim.reshape"(%867) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %876 = "mix.prim.reshape"(%874) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %877 = "mix.prim.reshape"(%875) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %878 = "mix.prim.reshape"(%876) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %879 = "mix.prim.transpose"(%877) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %880 = "mix.prim.transpose"(%878) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %881 = "mix.prim.transpose"(%880) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %882 = "mix.prim.unsqueeze"(%879) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %883 = "mix.prim.permute"(%882) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %884 = "mix.prim.unsqueeze"(%881) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %885 = "mix.prim.permute"(%884) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %886 = "mix.prim.permute"(%883) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %887 = "mix.prim.reshape"(%886) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %888 = "mix.prim.permute"(%885) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %889 = "mix.prim.reshape"(%888) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %890 = "mix.prim.batch_matmul"(%887, %889) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %891 = "mix.prim.reshape"(%890) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %892 = "mix.prim.permute"(%891) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %893 = "mix.prim.reshape"(%892) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %894 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %895 = "mix.prim.mul"(%893, %894) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %896 = "mix.prim.reshape"(%895) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %897 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %898 = "mix.comp.masked_fill"(%896, %4, %897) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %899 = "mix.comp.softmax"(%898) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %900 = "mix.prim.reshape"(%899) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %901 = "mix.prim.reshape"(%828) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %902 = "mix.prim.transpose"(%901) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %903 = "mix.prim.batch_matmul"(%900, %902) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %904 = "mix.prim.reshape"(%903) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %905 = "mix.prim.permute"(%904) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %906 = "mix.prim.reshape"(%905) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %907 = "mix.prim.reshape"(%906) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %908 = "mix.prim.transpose"(%814) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %909 = "mix.prim.matmul"(%907, %908) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %910 = "mix.prim.add"(%909, %815) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %911 = "mix.prim.reshape"(%910) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %912 = "mix.prim.mul"(%802, %911) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %913 = "mix.comp.weight"() <{param_loc = "transformer.h.6.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %914 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %915 = "mix.prim.pow"(%912, %914) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %916 = "mix.comp.mean"(%915) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %917 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %918 = "mix.prim.add"(%916, %917) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %919 = "mix.prim.rsqrt"(%918) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %920 = "mix.prim.mul"(%912, %919) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %921 = "mix.prim.mul"(%913, %920) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %922 = "mix.comp.weight"() <{param_loc = "transformer.h.6.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %923 = "mix.prim.reshape"(%922) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %924 = "mix.prim.batch_matmul"(%921, %923) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %925 = "mix.comp.silu"(%924) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %926 = "mix.comp.weight"() <{param_loc = "transformer.h.6.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %927 = "mix.prim.reshape"(%926) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %928 = "mix.prim.batch_matmul"(%921, %927) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %929 = "mix.prim.mul"(%925, %928) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %930 = "mix.comp.weight"() <{param_loc = "transformer.h.6.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %931 = "mix.prim.reshape"(%930) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %932 = "mix.prim.batch_matmul"(%929, %931) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %933 = "mix.comp.weight"() <{param_loc = "transformer.h.6.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %934 = "mix.prim.add"(%932, %933) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %935 = "mix.prim.add"(%934, %912) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %936 = "mix.comp.weight"() <{param_loc = "transformer.h.7.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %937 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %938 = "mix.prim.pow"(%935, %937) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %939 = "mix.comp.mean"(%938) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %940 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %941 = "mix.prim.add"(%939, %940) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %942 = "mix.prim.rsqrt"(%941) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %943 = "mix.prim.mul"(%935, %942) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %944 = "mix.prim.mul"(%936, %943) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %945 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %946 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %947 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %948 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %949 = "mix.prim.transpose"(%944) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %950 = "mix.prim.transpose"(%945) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %951 = "mix.prim.reshape"(%949) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %952 = "mix.prim.matmul"(%951, %950) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %953 = "mix.prim.reshape"(%952) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %954 = "mix.prim.reshape"(%953) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %955 = "mix.prim.transpose"(%946) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %956 = "mix.prim.reshape"(%949) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %957 = "mix.prim.matmul"(%956, %955) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %958 = "mix.prim.reshape"(%957) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %959 = "mix.prim.reshape"(%958) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %960 = "mix.prim.slice"(%959) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %961 = "mix.prim.slice"(%959) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %962 = "mix.prim.reshape"(%954) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %963 = "mix.prim.reshape"(%960) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %964 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %965 = "mix.prim.convert"(%964) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %966 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %967 = "mix.prim.div"(%965, %966) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %968 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %969 = "mix.prim.pow"(%968, %967) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %970 = "mix.prim.reciprocal"(%969) : (tensor<80xf16>) -> tensor<80xf16>
    %971 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %972 = "mix.prim.mul"(%971, %970) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %973 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %974 = "mix.prim.unsqueeze"(%973) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %975 = "mix.prim.permute"(%974) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %976 = "mix.prim.unsqueeze"(%972) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %977 = "mix.prim.permute"(%976) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %978 = "mix.prim.mul"(%975, %977) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %979 = "mix.prim.concat"(%978, %978) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %980 = "mix.prim.cos"(%979) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %981 = "mix.prim.slice"(%980) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %982 = "mix.prim.unsqueeze"(%981) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %983 = "mix.prim.slice"(%982) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %984 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %985 = "mix.prim.mul"(%983, %984) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %986 = "mix.prim.sin"(%979) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %987 = "mix.prim.slice"(%986) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %988 = "mix.prim.unsqueeze"(%987) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %989 = "mix.prim.slice"(%988) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %990 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %991 = "mix.prim.mul"(%989, %990) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %992 = "mix.prim.slice"(%985) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %993 = "mix.prim.slice"(%991) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %994 = "mix.prim.mul"(%962, %992) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %995 = "mix.prim.slice"(%962) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %996 = "mix.prim.slice"(%962) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %997 = "mix.prim.neg"(%996) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %998 = "mix.prim.concat"(%997, %995) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %999 = "mix.prim.mul"(%998, %993) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1000 = "mix.prim.add"(%994, %999) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1001 = "mix.prim.mul"(%963, %992) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1002 = "mix.prim.slice"(%963) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1003 = "mix.prim.slice"(%963) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1004 = "mix.prim.neg"(%1003) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1005 = "mix.prim.concat"(%1004, %1002) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1006 = "mix.prim.mul"(%1005, %993) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1007 = "mix.prim.add"(%1001, %1006) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1008 = "mix.prim.reshape"(%1000) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1009 = "mix.prim.reshape"(%1007) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1010 = "mix.prim.reshape"(%1008) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1011 = "mix.prim.reshape"(%1009) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1012 = "mix.prim.transpose"(%1010) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1013 = "mix.prim.transpose"(%1011) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1014 = "mix.prim.transpose"(%1013) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1015 = "mix.prim.unsqueeze"(%1012) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1016 = "mix.prim.permute"(%1015) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1017 = "mix.prim.unsqueeze"(%1014) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1018 = "mix.prim.permute"(%1017) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1019 = "mix.prim.permute"(%1016) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1020 = "mix.prim.reshape"(%1019) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1021 = "mix.prim.permute"(%1018) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1022 = "mix.prim.reshape"(%1021) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1023 = "mix.prim.batch_matmul"(%1020, %1022) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1024 = "mix.prim.reshape"(%1023) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1025 = "mix.prim.permute"(%1024) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1026 = "mix.prim.reshape"(%1025) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1027 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1028 = "mix.prim.mul"(%1026, %1027) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1029 = "mix.prim.reshape"(%1028) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1030 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1031 = "mix.comp.masked_fill"(%1029, %4, %1030) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1032 = "mix.comp.softmax"(%1031) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1033 = "mix.prim.reshape"(%1032) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1034 = "mix.prim.reshape"(%961) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1035 = "mix.prim.transpose"(%1034) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1036 = "mix.prim.batch_matmul"(%1033, %1035) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1037 = "mix.prim.reshape"(%1036) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1038 = "mix.prim.permute"(%1037) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1039 = "mix.prim.reshape"(%1038) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1040 = "mix.prim.reshape"(%1039) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1041 = "mix.prim.transpose"(%947) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1042 = "mix.prim.matmul"(%1040, %1041) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1043 = "mix.prim.add"(%1042, %948) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1044 = "mix.prim.reshape"(%1043) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1045 = "mix.prim.mul"(%935, %1044) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1046 = "mix.comp.weight"() <{param_loc = "transformer.h.7.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1047 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1048 = "mix.prim.pow"(%1045, %1047) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1049 = "mix.comp.mean"(%1048) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1050 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1051 = "mix.prim.add"(%1049, %1050) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1052 = "mix.prim.rsqrt"(%1051) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1053 = "mix.prim.mul"(%1045, %1052) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1054 = "mix.prim.mul"(%1046, %1053) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1055 = "mix.comp.weight"() <{param_loc = "transformer.h.7.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1056 = "mix.prim.reshape"(%1055) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1057 = "mix.prim.batch_matmul"(%1054, %1056) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1058 = "mix.comp.silu"(%1057) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1059 = "mix.comp.weight"() <{param_loc = "transformer.h.7.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1060 = "mix.prim.reshape"(%1059) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1061 = "mix.prim.batch_matmul"(%1054, %1060) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1062 = "mix.prim.mul"(%1058, %1061) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1063 = "mix.comp.weight"() <{param_loc = "transformer.h.7.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %1064 = "mix.prim.reshape"(%1063) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %1065 = "mix.prim.batch_matmul"(%1062, %1064) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %1066 = "mix.comp.weight"() <{param_loc = "transformer.h.7.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %1067 = "mix.prim.add"(%1065, %1066) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %1068 = "mix.prim.add"(%1067, %1045) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1069 = "mix.comp.weight"() <{param_loc = "transformer.h.8.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1070 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1071 = "mix.prim.pow"(%1068, %1070) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1072 = "mix.comp.mean"(%1071) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1073 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1074 = "mix.prim.add"(%1072, %1073) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1075 = "mix.prim.rsqrt"(%1074) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1076 = "mix.prim.mul"(%1068, %1075) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1077 = "mix.prim.mul"(%1069, %1076) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1078 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1079 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1080 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1081 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1082 = "mix.prim.transpose"(%1077) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1083 = "mix.prim.transpose"(%1078) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1084 = "mix.prim.reshape"(%1082) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1085 = "mix.prim.matmul"(%1084, %1083) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1086 = "mix.prim.reshape"(%1085) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1087 = "mix.prim.reshape"(%1086) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1088 = "mix.prim.transpose"(%1079) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1089 = "mix.prim.reshape"(%1082) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1090 = "mix.prim.matmul"(%1089, %1088) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1091 = "mix.prim.reshape"(%1090) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1092 = "mix.prim.reshape"(%1091) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1093 = "mix.prim.slice"(%1092) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1094 = "mix.prim.slice"(%1092) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1095 = "mix.prim.reshape"(%1087) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1096 = "mix.prim.reshape"(%1093) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1097 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1098 = "mix.prim.convert"(%1097) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1099 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1100 = "mix.prim.div"(%1098, %1099) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1101 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1102 = "mix.prim.pow"(%1101, %1100) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1103 = "mix.prim.reciprocal"(%1102) : (tensor<80xf16>) -> tensor<80xf16>
    %1104 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1105 = "mix.prim.mul"(%1104, %1103) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1106 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1107 = "mix.prim.unsqueeze"(%1106) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1108 = "mix.prim.permute"(%1107) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1109 = "mix.prim.unsqueeze"(%1105) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1110 = "mix.prim.permute"(%1109) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1111 = "mix.prim.mul"(%1108, %1110) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1112 = "mix.prim.concat"(%1111, %1111) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1113 = "mix.prim.cos"(%1112) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1114 = "mix.prim.slice"(%1113) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1115 = "mix.prim.unsqueeze"(%1114) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1116 = "mix.prim.slice"(%1115) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1117 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1118 = "mix.prim.mul"(%1116, %1117) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1119 = "mix.prim.sin"(%1112) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1120 = "mix.prim.slice"(%1119) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1121 = "mix.prim.unsqueeze"(%1120) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1122 = "mix.prim.slice"(%1121) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1123 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1124 = "mix.prim.mul"(%1122, %1123) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1125 = "mix.prim.slice"(%1118) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1126 = "mix.prim.slice"(%1124) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1127 = "mix.prim.mul"(%1095, %1125) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1128 = "mix.prim.slice"(%1095) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1129 = "mix.prim.slice"(%1095) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1130 = "mix.prim.neg"(%1129) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1131 = "mix.prim.concat"(%1130, %1128) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1132 = "mix.prim.mul"(%1131, %1126) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1133 = "mix.prim.add"(%1127, %1132) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1134 = "mix.prim.mul"(%1096, %1125) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1135 = "mix.prim.slice"(%1096) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1136 = "mix.prim.slice"(%1096) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1137 = "mix.prim.neg"(%1136) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1138 = "mix.prim.concat"(%1137, %1135) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1139 = "mix.prim.mul"(%1138, %1126) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1140 = "mix.prim.add"(%1134, %1139) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1141 = "mix.prim.reshape"(%1133) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1142 = "mix.prim.reshape"(%1140) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1143 = "mix.prim.reshape"(%1141) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1144 = "mix.prim.reshape"(%1142) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1145 = "mix.prim.transpose"(%1143) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1146 = "mix.prim.transpose"(%1144) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1147 = "mix.prim.transpose"(%1146) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1148 = "mix.prim.unsqueeze"(%1145) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1149 = "mix.prim.permute"(%1148) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1150 = "mix.prim.unsqueeze"(%1147) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1151 = "mix.prim.permute"(%1150) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1152 = "mix.prim.permute"(%1149) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1153 = "mix.prim.reshape"(%1152) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1154 = "mix.prim.permute"(%1151) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1155 = "mix.prim.reshape"(%1154) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1156 = "mix.prim.batch_matmul"(%1153, %1155) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1157 = "mix.prim.reshape"(%1156) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1158 = "mix.prim.permute"(%1157) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1159 = "mix.prim.reshape"(%1158) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1160 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1161 = "mix.prim.mul"(%1159, %1160) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1162 = "mix.prim.reshape"(%1161) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1163 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1164 = "mix.comp.masked_fill"(%1162, %4, %1163) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1165 = "mix.comp.softmax"(%1164) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1166 = "mix.prim.reshape"(%1165) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1167 = "mix.prim.reshape"(%1094) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1168 = "mix.prim.transpose"(%1167) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1169 = "mix.prim.batch_matmul"(%1166, %1168) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1170 = "mix.prim.reshape"(%1169) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1171 = "mix.prim.permute"(%1170) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1172 = "mix.prim.reshape"(%1171) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1173 = "mix.prim.reshape"(%1172) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1174 = "mix.prim.transpose"(%1080) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1175 = "mix.prim.matmul"(%1173, %1174) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1176 = "mix.prim.add"(%1175, %1081) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1177 = "mix.prim.reshape"(%1176) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1178 = "mix.prim.mul"(%1068, %1177) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1179 = "mix.comp.weight"() <{param_loc = "transformer.h.8.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1180 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1181 = "mix.prim.pow"(%1178, %1180) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1182 = "mix.comp.mean"(%1181) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1183 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1184 = "mix.prim.add"(%1182, %1183) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1185 = "mix.prim.rsqrt"(%1184) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1186 = "mix.prim.mul"(%1178, %1185) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1187 = "mix.prim.mul"(%1179, %1186) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1188 = "mix.comp.weight"() <{param_loc = "transformer.h.8.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1189 = "mix.prim.reshape"(%1188) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1190 = "mix.prim.batch_matmul"(%1187, %1189) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1191 = "mix.comp.silu"(%1190) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1192 = "mix.comp.weight"() <{param_loc = "transformer.h.8.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1193 = "mix.prim.reshape"(%1192) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1194 = "mix.prim.batch_matmul"(%1187, %1193) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1195 = "mix.prim.mul"(%1191, %1194) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1196 = "mix.comp.weight"() <{param_loc = "transformer.h.8.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %1197 = "mix.prim.reshape"(%1196) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %1198 = "mix.prim.batch_matmul"(%1195, %1197) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %1199 = "mix.comp.weight"() <{param_loc = "transformer.h.8.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %1200 = "mix.prim.add"(%1198, %1199) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %1201 = "mix.prim.add"(%1200, %1178) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1202 = "mix.comp.weight"() <{param_loc = "transformer.h.9.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1203 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1204 = "mix.prim.pow"(%1201, %1203) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1205 = "mix.comp.mean"(%1204) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1206 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1207 = "mix.prim.add"(%1205, %1206) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1208 = "mix.prim.rsqrt"(%1207) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1209 = "mix.prim.mul"(%1201, %1208) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1210 = "mix.prim.mul"(%1202, %1209) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1211 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1212 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1213 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1214 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1215 = "mix.prim.transpose"(%1210) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1216 = "mix.prim.transpose"(%1211) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1217 = "mix.prim.reshape"(%1215) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1218 = "mix.prim.matmul"(%1217, %1216) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1219 = "mix.prim.reshape"(%1218) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1220 = "mix.prim.reshape"(%1219) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1221 = "mix.prim.transpose"(%1212) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1222 = "mix.prim.reshape"(%1215) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1223 = "mix.prim.matmul"(%1222, %1221) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1224 = "mix.prim.reshape"(%1223) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1225 = "mix.prim.reshape"(%1224) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1226 = "mix.prim.slice"(%1225) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1227 = "mix.prim.slice"(%1225) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1228 = "mix.prim.reshape"(%1220) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1229 = "mix.prim.reshape"(%1226) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1230 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1231 = "mix.prim.convert"(%1230) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1232 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1233 = "mix.prim.div"(%1231, %1232) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1234 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1235 = "mix.prim.pow"(%1234, %1233) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1236 = "mix.prim.reciprocal"(%1235) : (tensor<80xf16>) -> tensor<80xf16>
    %1237 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1238 = "mix.prim.mul"(%1237, %1236) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1239 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1240 = "mix.prim.unsqueeze"(%1239) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1241 = "mix.prim.permute"(%1240) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1242 = "mix.prim.unsqueeze"(%1238) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1243 = "mix.prim.permute"(%1242) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1244 = "mix.prim.mul"(%1241, %1243) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1245 = "mix.prim.concat"(%1244, %1244) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1246 = "mix.prim.cos"(%1245) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1247 = "mix.prim.slice"(%1246) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1248 = "mix.prim.unsqueeze"(%1247) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1249 = "mix.prim.slice"(%1248) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1250 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1251 = "mix.prim.mul"(%1249, %1250) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1252 = "mix.prim.sin"(%1245) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1253 = "mix.prim.slice"(%1252) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1254 = "mix.prim.unsqueeze"(%1253) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1255 = "mix.prim.slice"(%1254) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1256 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1257 = "mix.prim.mul"(%1255, %1256) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1258 = "mix.prim.slice"(%1251) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1259 = "mix.prim.slice"(%1257) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1260 = "mix.prim.mul"(%1228, %1258) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1261 = "mix.prim.slice"(%1228) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1262 = "mix.prim.slice"(%1228) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1263 = "mix.prim.neg"(%1262) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1264 = "mix.prim.concat"(%1263, %1261) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1265 = "mix.prim.mul"(%1264, %1259) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1266 = "mix.prim.add"(%1260, %1265) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1267 = "mix.prim.mul"(%1229, %1258) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1268 = "mix.prim.slice"(%1229) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1269 = "mix.prim.slice"(%1229) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1270 = "mix.prim.neg"(%1269) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1271 = "mix.prim.concat"(%1270, %1268) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1272 = "mix.prim.mul"(%1271, %1259) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1273 = "mix.prim.add"(%1267, %1272) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1274 = "mix.prim.reshape"(%1266) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1275 = "mix.prim.reshape"(%1273) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1276 = "mix.prim.reshape"(%1274) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1277 = "mix.prim.reshape"(%1275) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1278 = "mix.prim.transpose"(%1276) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1279 = "mix.prim.transpose"(%1277) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1280 = "mix.prim.transpose"(%1279) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1281 = "mix.prim.unsqueeze"(%1278) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1282 = "mix.prim.permute"(%1281) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1283 = "mix.prim.unsqueeze"(%1280) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1284 = "mix.prim.permute"(%1283) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1285 = "mix.prim.permute"(%1282) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1286 = "mix.prim.reshape"(%1285) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1287 = "mix.prim.permute"(%1284) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1288 = "mix.prim.reshape"(%1287) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1289 = "mix.prim.batch_matmul"(%1286, %1288) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1290 = "mix.prim.reshape"(%1289) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1291 = "mix.prim.permute"(%1290) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1292 = "mix.prim.reshape"(%1291) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1293 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1294 = "mix.prim.mul"(%1292, %1293) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1295 = "mix.prim.reshape"(%1294) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1296 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1297 = "mix.comp.masked_fill"(%1295, %4, %1296) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1298 = "mix.comp.softmax"(%1297) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1299 = "mix.prim.reshape"(%1298) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1300 = "mix.prim.reshape"(%1227) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1301 = "mix.prim.transpose"(%1300) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1302 = "mix.prim.batch_matmul"(%1299, %1301) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1303 = "mix.prim.reshape"(%1302) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1304 = "mix.prim.permute"(%1303) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1305 = "mix.prim.reshape"(%1304) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1306 = "mix.prim.reshape"(%1305) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1307 = "mix.prim.transpose"(%1213) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1308 = "mix.prim.matmul"(%1306, %1307) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1309 = "mix.prim.add"(%1308, %1214) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1310 = "mix.prim.reshape"(%1309) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1311 = "mix.prim.mul"(%1201, %1310) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1312 = "mix.comp.weight"() <{param_loc = "transformer.h.9.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1313 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1314 = "mix.prim.pow"(%1311, %1313) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1315 = "mix.comp.mean"(%1314) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1316 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1317 = "mix.prim.add"(%1315, %1316) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1318 = "mix.prim.rsqrt"(%1317) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1319 = "mix.prim.mul"(%1311, %1318) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1320 = "mix.prim.mul"(%1312, %1319) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1321 = "mix.comp.weight"() <{param_loc = "transformer.h.9.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1322 = "mix.prim.reshape"(%1321) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1323 = "mix.prim.batch_matmul"(%1320, %1322) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1324 = "mix.comp.silu"(%1323) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1325 = "mix.comp.weight"() <{param_loc = "transformer.h.9.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1326 = "mix.prim.reshape"(%1325) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1327 = "mix.prim.batch_matmul"(%1320, %1326) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1328 = "mix.prim.mul"(%1324, %1327) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1329 = "mix.comp.weight"() <{param_loc = "transformer.h.9.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %1330 = "mix.prim.reshape"(%1329) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %1331 = "mix.prim.batch_matmul"(%1328, %1330) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %1332 = "mix.comp.weight"() <{param_loc = "transformer.h.9.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %1333 = "mix.prim.add"(%1331, %1332) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %1334 = "mix.prim.add"(%1333, %1311) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1335 = "mix.comp.weight"() <{param_loc = "transformer.h.10.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1336 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1337 = "mix.prim.pow"(%1334, %1336) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1338 = "mix.comp.mean"(%1337) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1339 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1340 = "mix.prim.add"(%1338, %1339) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1341 = "mix.prim.rsqrt"(%1340) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1342 = "mix.prim.mul"(%1334, %1341) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1343 = "mix.prim.mul"(%1335, %1342) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1344 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1345 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1346 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1347 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1348 = "mix.prim.transpose"(%1343) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1349 = "mix.prim.transpose"(%1344) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1350 = "mix.prim.reshape"(%1348) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1351 = "mix.prim.matmul"(%1350, %1349) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1352 = "mix.prim.reshape"(%1351) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1353 = "mix.prim.reshape"(%1352) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1354 = "mix.prim.transpose"(%1345) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1355 = "mix.prim.reshape"(%1348) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1356 = "mix.prim.matmul"(%1355, %1354) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1357 = "mix.prim.reshape"(%1356) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1358 = "mix.prim.reshape"(%1357) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1359 = "mix.prim.slice"(%1358) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1360 = "mix.prim.slice"(%1358) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1361 = "mix.prim.reshape"(%1353) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1362 = "mix.prim.reshape"(%1359) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1363 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1364 = "mix.prim.convert"(%1363) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1365 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1366 = "mix.prim.div"(%1364, %1365) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1367 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1368 = "mix.prim.pow"(%1367, %1366) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1369 = "mix.prim.reciprocal"(%1368) : (tensor<80xf16>) -> tensor<80xf16>
    %1370 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1371 = "mix.prim.mul"(%1370, %1369) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1372 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1373 = "mix.prim.unsqueeze"(%1372) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1374 = "mix.prim.permute"(%1373) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1375 = "mix.prim.unsqueeze"(%1371) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1376 = "mix.prim.permute"(%1375) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1377 = "mix.prim.mul"(%1374, %1376) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1378 = "mix.prim.concat"(%1377, %1377) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1379 = "mix.prim.cos"(%1378) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1380 = "mix.prim.slice"(%1379) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1381 = "mix.prim.unsqueeze"(%1380) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1382 = "mix.prim.slice"(%1381) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1383 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1384 = "mix.prim.mul"(%1382, %1383) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1385 = "mix.prim.sin"(%1378) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1386 = "mix.prim.slice"(%1385) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1387 = "mix.prim.unsqueeze"(%1386) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1388 = "mix.prim.slice"(%1387) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1389 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1390 = "mix.prim.mul"(%1388, %1389) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1391 = "mix.prim.slice"(%1384) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1392 = "mix.prim.slice"(%1390) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1393 = "mix.prim.mul"(%1361, %1391) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1394 = "mix.prim.slice"(%1361) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1395 = "mix.prim.slice"(%1361) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1396 = "mix.prim.neg"(%1395) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1397 = "mix.prim.concat"(%1396, %1394) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1398 = "mix.prim.mul"(%1397, %1392) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1399 = "mix.prim.add"(%1393, %1398) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1400 = "mix.prim.mul"(%1362, %1391) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1401 = "mix.prim.slice"(%1362) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1402 = "mix.prim.slice"(%1362) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1403 = "mix.prim.neg"(%1402) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1404 = "mix.prim.concat"(%1403, %1401) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1405 = "mix.prim.mul"(%1404, %1392) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1406 = "mix.prim.add"(%1400, %1405) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1407 = "mix.prim.reshape"(%1399) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1408 = "mix.prim.reshape"(%1406) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1409 = "mix.prim.reshape"(%1407) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1410 = "mix.prim.reshape"(%1408) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1411 = "mix.prim.transpose"(%1409) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1412 = "mix.prim.transpose"(%1410) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1413 = "mix.prim.transpose"(%1412) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1414 = "mix.prim.unsqueeze"(%1411) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1415 = "mix.prim.permute"(%1414) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1416 = "mix.prim.unsqueeze"(%1413) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1417 = "mix.prim.permute"(%1416) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1418 = "mix.prim.permute"(%1415) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1419 = "mix.prim.reshape"(%1418) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1420 = "mix.prim.permute"(%1417) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1421 = "mix.prim.reshape"(%1420) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1422 = "mix.prim.batch_matmul"(%1419, %1421) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1423 = "mix.prim.reshape"(%1422) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1424 = "mix.prim.permute"(%1423) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1425 = "mix.prim.reshape"(%1424) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1426 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1427 = "mix.prim.mul"(%1425, %1426) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1428 = "mix.prim.reshape"(%1427) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1429 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1430 = "mix.comp.masked_fill"(%1428, %4, %1429) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1431 = "mix.comp.softmax"(%1430) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1432 = "mix.prim.reshape"(%1431) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1433 = "mix.prim.reshape"(%1360) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1434 = "mix.prim.transpose"(%1433) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1435 = "mix.prim.batch_matmul"(%1432, %1434) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1436 = "mix.prim.reshape"(%1435) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1437 = "mix.prim.permute"(%1436) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1438 = "mix.prim.reshape"(%1437) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1439 = "mix.prim.reshape"(%1438) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1440 = "mix.prim.transpose"(%1346) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1441 = "mix.prim.matmul"(%1439, %1440) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1442 = "mix.prim.add"(%1441, %1347) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1443 = "mix.prim.reshape"(%1442) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1444 = "mix.prim.mul"(%1334, %1443) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1445 = "mix.comp.weight"() <{param_loc = "transformer.h.10.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1446 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1447 = "mix.prim.pow"(%1444, %1446) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1448 = "mix.comp.mean"(%1447) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1449 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1450 = "mix.prim.add"(%1448, %1449) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1451 = "mix.prim.rsqrt"(%1450) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1452 = "mix.prim.mul"(%1444, %1451) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1453 = "mix.prim.mul"(%1445, %1452) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1454 = "mix.comp.weight"() <{param_loc = "transformer.h.10.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1455 = "mix.prim.reshape"(%1454) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1456 = "mix.prim.batch_matmul"(%1453, %1455) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1457 = "mix.comp.silu"(%1456) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1458 = "mix.comp.weight"() <{param_loc = "transformer.h.10.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1459 = "mix.prim.reshape"(%1458) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1460 = "mix.prim.batch_matmul"(%1453, %1459) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1461 = "mix.prim.mul"(%1457, %1460) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1462 = "mix.comp.weight"() <{param_loc = "transformer.h.10.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %1463 = "mix.prim.reshape"(%1462) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %1464 = "mix.prim.batch_matmul"(%1461, %1463) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %1465 = "mix.comp.weight"() <{param_loc = "transformer.h.10.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %1466 = "mix.prim.add"(%1464, %1465) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %1467 = "mix.prim.add"(%1466, %1444) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1468 = "mix.comp.weight"() <{param_loc = "transformer.h.11.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1469 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1470 = "mix.prim.pow"(%1467, %1469) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1471 = "mix.comp.mean"(%1470) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1472 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1473 = "mix.prim.add"(%1471, %1472) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1474 = "mix.prim.rsqrt"(%1473) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1475 = "mix.prim.mul"(%1467, %1474) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1476 = "mix.prim.mul"(%1468, %1475) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1477 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1478 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1479 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1480 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1481 = "mix.prim.transpose"(%1476) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1482 = "mix.prim.transpose"(%1477) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1483 = "mix.prim.reshape"(%1481) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1484 = "mix.prim.matmul"(%1483, %1482) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1485 = "mix.prim.reshape"(%1484) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1486 = "mix.prim.reshape"(%1485) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1487 = "mix.prim.transpose"(%1478) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1488 = "mix.prim.reshape"(%1481) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1489 = "mix.prim.matmul"(%1488, %1487) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1490 = "mix.prim.reshape"(%1489) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1491 = "mix.prim.reshape"(%1490) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1492 = "mix.prim.slice"(%1491) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1493 = "mix.prim.slice"(%1491) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1494 = "mix.prim.reshape"(%1486) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1495 = "mix.prim.reshape"(%1492) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1496 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1497 = "mix.prim.convert"(%1496) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1498 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1499 = "mix.prim.div"(%1497, %1498) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1500 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1501 = "mix.prim.pow"(%1500, %1499) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1502 = "mix.prim.reciprocal"(%1501) : (tensor<80xf16>) -> tensor<80xf16>
    %1503 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1504 = "mix.prim.mul"(%1503, %1502) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1505 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1506 = "mix.prim.unsqueeze"(%1505) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1507 = "mix.prim.permute"(%1506) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1508 = "mix.prim.unsqueeze"(%1504) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1509 = "mix.prim.permute"(%1508) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1510 = "mix.prim.mul"(%1507, %1509) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1511 = "mix.prim.concat"(%1510, %1510) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1512 = "mix.prim.cos"(%1511) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1513 = "mix.prim.slice"(%1512) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1514 = "mix.prim.unsqueeze"(%1513) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1515 = "mix.prim.slice"(%1514) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1516 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1517 = "mix.prim.mul"(%1515, %1516) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1518 = "mix.prim.sin"(%1511) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1519 = "mix.prim.slice"(%1518) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1520 = "mix.prim.unsqueeze"(%1519) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1521 = "mix.prim.slice"(%1520) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1522 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1523 = "mix.prim.mul"(%1521, %1522) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1524 = "mix.prim.slice"(%1517) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1525 = "mix.prim.slice"(%1523) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1526 = "mix.prim.mul"(%1494, %1524) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1527 = "mix.prim.slice"(%1494) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1528 = "mix.prim.slice"(%1494) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1529 = "mix.prim.neg"(%1528) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1530 = "mix.prim.concat"(%1529, %1527) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1531 = "mix.prim.mul"(%1530, %1525) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1532 = "mix.prim.add"(%1526, %1531) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1533 = "mix.prim.mul"(%1495, %1524) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1534 = "mix.prim.slice"(%1495) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1535 = "mix.prim.slice"(%1495) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1536 = "mix.prim.neg"(%1535) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1537 = "mix.prim.concat"(%1536, %1534) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1538 = "mix.prim.mul"(%1537, %1525) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1539 = "mix.prim.add"(%1533, %1538) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1540 = "mix.prim.reshape"(%1532) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1541 = "mix.prim.reshape"(%1539) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1542 = "mix.prim.reshape"(%1540) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1543 = "mix.prim.reshape"(%1541) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1544 = "mix.prim.transpose"(%1542) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1545 = "mix.prim.transpose"(%1543) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1546 = "mix.prim.transpose"(%1545) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1547 = "mix.prim.unsqueeze"(%1544) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1548 = "mix.prim.permute"(%1547) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1549 = "mix.prim.unsqueeze"(%1546) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1550 = "mix.prim.permute"(%1549) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1551 = "mix.prim.permute"(%1548) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1552 = "mix.prim.reshape"(%1551) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1553 = "mix.prim.permute"(%1550) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1554 = "mix.prim.reshape"(%1553) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1555 = "mix.prim.batch_matmul"(%1552, %1554) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1556 = "mix.prim.reshape"(%1555) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1557 = "mix.prim.permute"(%1556) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1558 = "mix.prim.reshape"(%1557) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1559 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1560 = "mix.prim.mul"(%1558, %1559) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1561 = "mix.prim.reshape"(%1560) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1562 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1563 = "mix.comp.masked_fill"(%1561, %4, %1562) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1564 = "mix.comp.softmax"(%1563) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1565 = "mix.prim.reshape"(%1564) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1566 = "mix.prim.reshape"(%1493) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1567 = "mix.prim.transpose"(%1566) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1568 = "mix.prim.batch_matmul"(%1565, %1567) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1569 = "mix.prim.reshape"(%1568) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1570 = "mix.prim.permute"(%1569) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1571 = "mix.prim.reshape"(%1570) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1572 = "mix.prim.reshape"(%1571) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1573 = "mix.prim.transpose"(%1479) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1574 = "mix.prim.matmul"(%1572, %1573) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1575 = "mix.prim.add"(%1574, %1480) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1576 = "mix.prim.reshape"(%1575) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1577 = "mix.prim.mul"(%1467, %1576) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1578 = "mix.comp.weight"() <{param_loc = "transformer.h.11.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1579 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1580 = "mix.prim.pow"(%1577, %1579) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1581 = "mix.comp.mean"(%1580) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1582 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1583 = "mix.prim.add"(%1581, %1582) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1584 = "mix.prim.rsqrt"(%1583) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1585 = "mix.prim.mul"(%1577, %1584) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1586 = "mix.prim.mul"(%1578, %1585) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1587 = "mix.comp.weight"() <{param_loc = "transformer.h.11.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1588 = "mix.prim.reshape"(%1587) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1589 = "mix.prim.batch_matmul"(%1586, %1588) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1590 = "mix.comp.silu"(%1589) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1591 = "mix.comp.weight"() <{param_loc = "transformer.h.11.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1592 = "mix.prim.reshape"(%1591) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1593 = "mix.prim.batch_matmul"(%1586, %1592) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1594 = "mix.prim.mul"(%1590, %1593) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1595 = "mix.comp.weight"() <{param_loc = "transformer.h.11.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %1596 = "mix.prim.reshape"(%1595) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %1597 = "mix.prim.batch_matmul"(%1594, %1596) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %1598 = "mix.comp.weight"() <{param_loc = "transformer.h.11.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %1599 = "mix.prim.add"(%1597, %1598) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %1600 = "mix.prim.add"(%1599, %1577) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1601 = "mix.comp.weight"() <{param_loc = "transformer.h.12.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1602 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1603 = "mix.prim.pow"(%1600, %1602) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1604 = "mix.comp.mean"(%1603) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1605 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1606 = "mix.prim.add"(%1604, %1605) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1607 = "mix.prim.rsqrt"(%1606) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1608 = "mix.prim.mul"(%1600, %1607) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1609 = "mix.prim.mul"(%1601, %1608) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1610 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1611 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1612 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1613 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1614 = "mix.prim.transpose"(%1609) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1615 = "mix.prim.transpose"(%1610) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1616 = "mix.prim.reshape"(%1614) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1617 = "mix.prim.matmul"(%1616, %1615) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1618 = "mix.prim.reshape"(%1617) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1619 = "mix.prim.reshape"(%1618) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1620 = "mix.prim.transpose"(%1611) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1621 = "mix.prim.reshape"(%1614) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1622 = "mix.prim.matmul"(%1621, %1620) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1623 = "mix.prim.reshape"(%1622) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1624 = "mix.prim.reshape"(%1623) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1625 = "mix.prim.slice"(%1624) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1626 = "mix.prim.slice"(%1624) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1627 = "mix.prim.reshape"(%1619) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1628 = "mix.prim.reshape"(%1625) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1629 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1630 = "mix.prim.convert"(%1629) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1631 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1632 = "mix.prim.div"(%1630, %1631) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1633 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1634 = "mix.prim.pow"(%1633, %1632) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1635 = "mix.prim.reciprocal"(%1634) : (tensor<80xf16>) -> tensor<80xf16>
    %1636 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1637 = "mix.prim.mul"(%1636, %1635) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1638 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1639 = "mix.prim.unsqueeze"(%1638) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1640 = "mix.prim.permute"(%1639) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1641 = "mix.prim.unsqueeze"(%1637) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1642 = "mix.prim.permute"(%1641) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1643 = "mix.prim.mul"(%1640, %1642) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1644 = "mix.prim.concat"(%1643, %1643) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1645 = "mix.prim.cos"(%1644) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1646 = "mix.prim.slice"(%1645) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1647 = "mix.prim.unsqueeze"(%1646) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1648 = "mix.prim.slice"(%1647) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1649 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1650 = "mix.prim.mul"(%1648, %1649) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1651 = "mix.prim.sin"(%1644) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1652 = "mix.prim.slice"(%1651) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1653 = "mix.prim.unsqueeze"(%1652) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1654 = "mix.prim.slice"(%1653) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1655 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1656 = "mix.prim.mul"(%1654, %1655) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1657 = "mix.prim.slice"(%1650) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1658 = "mix.prim.slice"(%1656) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1659 = "mix.prim.mul"(%1627, %1657) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1660 = "mix.prim.slice"(%1627) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1661 = "mix.prim.slice"(%1627) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1662 = "mix.prim.neg"(%1661) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1663 = "mix.prim.concat"(%1662, %1660) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1664 = "mix.prim.mul"(%1663, %1658) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1665 = "mix.prim.add"(%1659, %1664) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1666 = "mix.prim.mul"(%1628, %1657) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1667 = "mix.prim.slice"(%1628) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1668 = "mix.prim.slice"(%1628) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1669 = "mix.prim.neg"(%1668) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1670 = "mix.prim.concat"(%1669, %1667) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1671 = "mix.prim.mul"(%1670, %1658) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1672 = "mix.prim.add"(%1666, %1671) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1673 = "mix.prim.reshape"(%1665) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1674 = "mix.prim.reshape"(%1672) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1675 = "mix.prim.reshape"(%1673) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1676 = "mix.prim.reshape"(%1674) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1677 = "mix.prim.transpose"(%1675) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1678 = "mix.prim.transpose"(%1676) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1679 = "mix.prim.transpose"(%1678) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1680 = "mix.prim.unsqueeze"(%1677) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1681 = "mix.prim.permute"(%1680) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1682 = "mix.prim.unsqueeze"(%1679) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1683 = "mix.prim.permute"(%1682) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1684 = "mix.prim.permute"(%1681) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1685 = "mix.prim.reshape"(%1684) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1686 = "mix.prim.permute"(%1683) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1687 = "mix.prim.reshape"(%1686) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1688 = "mix.prim.batch_matmul"(%1685, %1687) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1689 = "mix.prim.reshape"(%1688) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1690 = "mix.prim.permute"(%1689) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1691 = "mix.prim.reshape"(%1690) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1692 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1693 = "mix.prim.mul"(%1691, %1692) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1694 = "mix.prim.reshape"(%1693) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1695 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1696 = "mix.comp.masked_fill"(%1694, %4, %1695) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1697 = "mix.comp.softmax"(%1696) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1698 = "mix.prim.reshape"(%1697) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1699 = "mix.prim.reshape"(%1626) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1700 = "mix.prim.transpose"(%1699) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1701 = "mix.prim.batch_matmul"(%1698, %1700) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1702 = "mix.prim.reshape"(%1701) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1703 = "mix.prim.permute"(%1702) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1704 = "mix.prim.reshape"(%1703) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1705 = "mix.prim.reshape"(%1704) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1706 = "mix.prim.transpose"(%1612) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1707 = "mix.prim.matmul"(%1705, %1706) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1708 = "mix.prim.add"(%1707, %1613) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1709 = "mix.prim.reshape"(%1708) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1710 = "mix.prim.mul"(%1600, %1709) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1711 = "mix.comp.weight"() <{param_loc = "transformer.h.12.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1712 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1713 = "mix.prim.pow"(%1710, %1712) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1714 = "mix.comp.mean"(%1713) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1715 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1716 = "mix.prim.add"(%1714, %1715) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1717 = "mix.prim.rsqrt"(%1716) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1718 = "mix.prim.mul"(%1710, %1717) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1719 = "mix.prim.mul"(%1711, %1718) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1720 = "mix.comp.weight"() <{param_loc = "transformer.h.12.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1721 = "mix.prim.reshape"(%1720) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1722 = "mix.prim.batch_matmul"(%1719, %1721) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1723 = "mix.comp.silu"(%1722) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1724 = "mix.comp.weight"() <{param_loc = "transformer.h.12.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1725 = "mix.prim.reshape"(%1724) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1726 = "mix.prim.batch_matmul"(%1719, %1725) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1727 = "mix.prim.mul"(%1723, %1726) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1728 = "mix.comp.weight"() <{param_loc = "transformer.h.12.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %1729 = "mix.prim.reshape"(%1728) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %1730 = "mix.prim.batch_matmul"(%1727, %1729) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %1731 = "mix.comp.weight"() <{param_loc = "transformer.h.12.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %1732 = "mix.prim.add"(%1730, %1731) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %1733 = "mix.prim.add"(%1732, %1710) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1734 = "mix.comp.weight"() <{param_loc = "transformer.h.13.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1735 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1736 = "mix.prim.pow"(%1733, %1735) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1737 = "mix.comp.mean"(%1736) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1738 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1739 = "mix.prim.add"(%1737, %1738) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1740 = "mix.prim.rsqrt"(%1739) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1741 = "mix.prim.mul"(%1733, %1740) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1742 = "mix.prim.mul"(%1734, %1741) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1743 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1744 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1745 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1746 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1747 = "mix.prim.transpose"(%1742) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1748 = "mix.prim.transpose"(%1743) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1749 = "mix.prim.reshape"(%1747) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1750 = "mix.prim.matmul"(%1749, %1748) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1751 = "mix.prim.reshape"(%1750) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1752 = "mix.prim.reshape"(%1751) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1753 = "mix.prim.transpose"(%1744) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1754 = "mix.prim.reshape"(%1747) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1755 = "mix.prim.matmul"(%1754, %1753) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1756 = "mix.prim.reshape"(%1755) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1757 = "mix.prim.reshape"(%1756) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1758 = "mix.prim.slice"(%1757) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1759 = "mix.prim.slice"(%1757) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1760 = "mix.prim.reshape"(%1752) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1761 = "mix.prim.reshape"(%1758) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1762 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1763 = "mix.prim.convert"(%1762) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1764 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1765 = "mix.prim.div"(%1763, %1764) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1766 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1767 = "mix.prim.pow"(%1766, %1765) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1768 = "mix.prim.reciprocal"(%1767) : (tensor<80xf16>) -> tensor<80xf16>
    %1769 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1770 = "mix.prim.mul"(%1769, %1768) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1771 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1772 = "mix.prim.unsqueeze"(%1771) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1773 = "mix.prim.permute"(%1772) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1774 = "mix.prim.unsqueeze"(%1770) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1775 = "mix.prim.permute"(%1774) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1776 = "mix.prim.mul"(%1773, %1775) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1777 = "mix.prim.concat"(%1776, %1776) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1778 = "mix.prim.cos"(%1777) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1779 = "mix.prim.slice"(%1778) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1780 = "mix.prim.unsqueeze"(%1779) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1781 = "mix.prim.slice"(%1780) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1782 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1783 = "mix.prim.mul"(%1781, %1782) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1784 = "mix.prim.sin"(%1777) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1785 = "mix.prim.slice"(%1784) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1786 = "mix.prim.unsqueeze"(%1785) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1787 = "mix.prim.slice"(%1786) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1788 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1789 = "mix.prim.mul"(%1787, %1788) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1790 = "mix.prim.slice"(%1783) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1791 = "mix.prim.slice"(%1789) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1792 = "mix.prim.mul"(%1760, %1790) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1793 = "mix.prim.slice"(%1760) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1794 = "mix.prim.slice"(%1760) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1795 = "mix.prim.neg"(%1794) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1796 = "mix.prim.concat"(%1795, %1793) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1797 = "mix.prim.mul"(%1796, %1791) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1798 = "mix.prim.add"(%1792, %1797) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1799 = "mix.prim.mul"(%1761, %1790) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1800 = "mix.prim.slice"(%1761) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1801 = "mix.prim.slice"(%1761) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1802 = "mix.prim.neg"(%1801) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1803 = "mix.prim.concat"(%1802, %1800) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1804 = "mix.prim.mul"(%1803, %1791) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1805 = "mix.prim.add"(%1799, %1804) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1806 = "mix.prim.reshape"(%1798) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1807 = "mix.prim.reshape"(%1805) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1808 = "mix.prim.reshape"(%1806) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1809 = "mix.prim.reshape"(%1807) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1810 = "mix.prim.transpose"(%1808) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1811 = "mix.prim.transpose"(%1809) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1812 = "mix.prim.transpose"(%1811) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1813 = "mix.prim.unsqueeze"(%1810) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1814 = "mix.prim.permute"(%1813) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1815 = "mix.prim.unsqueeze"(%1812) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1816 = "mix.prim.permute"(%1815) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1817 = "mix.prim.permute"(%1814) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1818 = "mix.prim.reshape"(%1817) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1819 = "mix.prim.permute"(%1816) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1820 = "mix.prim.reshape"(%1819) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1821 = "mix.prim.batch_matmul"(%1818, %1820) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1822 = "mix.prim.reshape"(%1821) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1823 = "mix.prim.permute"(%1822) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1824 = "mix.prim.reshape"(%1823) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1825 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1826 = "mix.prim.mul"(%1824, %1825) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1827 = "mix.prim.reshape"(%1826) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1828 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1829 = "mix.comp.masked_fill"(%1827, %4, %1828) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1830 = "mix.comp.softmax"(%1829) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1831 = "mix.prim.reshape"(%1830) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1832 = "mix.prim.reshape"(%1759) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1833 = "mix.prim.transpose"(%1832) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1834 = "mix.prim.batch_matmul"(%1831, %1833) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1835 = "mix.prim.reshape"(%1834) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1836 = "mix.prim.permute"(%1835) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1837 = "mix.prim.reshape"(%1836) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1838 = "mix.prim.reshape"(%1837) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1839 = "mix.prim.transpose"(%1745) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1840 = "mix.prim.matmul"(%1838, %1839) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1841 = "mix.prim.add"(%1840, %1746) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1842 = "mix.prim.reshape"(%1841) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1843 = "mix.prim.mul"(%1733, %1842) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1844 = "mix.comp.weight"() <{param_loc = "transformer.h.13.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1845 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1846 = "mix.prim.pow"(%1843, %1845) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1847 = "mix.comp.mean"(%1846) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1848 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1849 = "mix.prim.add"(%1847, %1848) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1850 = "mix.prim.rsqrt"(%1849) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1851 = "mix.prim.mul"(%1843, %1850) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1852 = "mix.prim.mul"(%1844, %1851) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1853 = "mix.comp.weight"() <{param_loc = "transformer.h.13.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1854 = "mix.prim.reshape"(%1853) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1855 = "mix.prim.batch_matmul"(%1852, %1854) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1856 = "mix.comp.silu"(%1855) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1857 = "mix.comp.weight"() <{param_loc = "transformer.h.13.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1858 = "mix.prim.reshape"(%1857) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1859 = "mix.prim.batch_matmul"(%1852, %1858) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1860 = "mix.prim.mul"(%1856, %1859) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1861 = "mix.comp.weight"() <{param_loc = "transformer.h.13.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %1862 = "mix.prim.reshape"(%1861) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %1863 = "mix.prim.batch_matmul"(%1860, %1862) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %1864 = "mix.comp.weight"() <{param_loc = "transformer.h.13.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %1865 = "mix.prim.add"(%1863, %1864) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %1866 = "mix.prim.add"(%1865, %1843) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1867 = "mix.comp.weight"() <{param_loc = "transformer.h.14.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1868 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1869 = "mix.prim.pow"(%1866, %1868) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1870 = "mix.comp.mean"(%1869) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1871 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1872 = "mix.prim.add"(%1870, %1871) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1873 = "mix.prim.rsqrt"(%1872) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1874 = "mix.prim.mul"(%1866, %1873) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1875 = "mix.prim.mul"(%1867, %1874) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1876 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1877 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1878 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1879 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1880 = "mix.prim.transpose"(%1875) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1881 = "mix.prim.transpose"(%1876) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1882 = "mix.prim.reshape"(%1880) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1883 = "mix.prim.matmul"(%1882, %1881) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1884 = "mix.prim.reshape"(%1883) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1885 = "mix.prim.reshape"(%1884) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1886 = "mix.prim.transpose"(%1877) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1887 = "mix.prim.reshape"(%1880) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1888 = "mix.prim.matmul"(%1887, %1886) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1889 = "mix.prim.reshape"(%1888) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1890 = "mix.prim.reshape"(%1889) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1891 = "mix.prim.slice"(%1890) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1892 = "mix.prim.slice"(%1890) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1893 = "mix.prim.reshape"(%1885) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1894 = "mix.prim.reshape"(%1891) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1895 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1896 = "mix.prim.convert"(%1895) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1897 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1898 = "mix.prim.div"(%1896, %1897) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1899 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1900 = "mix.prim.pow"(%1899, %1898) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1901 = "mix.prim.reciprocal"(%1900) : (tensor<80xf16>) -> tensor<80xf16>
    %1902 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1903 = "mix.prim.mul"(%1902, %1901) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1904 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1905 = "mix.prim.unsqueeze"(%1904) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1906 = "mix.prim.permute"(%1905) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1907 = "mix.prim.unsqueeze"(%1903) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1908 = "mix.prim.permute"(%1907) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1909 = "mix.prim.mul"(%1906, %1908) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1910 = "mix.prim.concat"(%1909, %1909) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1911 = "mix.prim.cos"(%1910) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1912 = "mix.prim.slice"(%1911) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1913 = "mix.prim.unsqueeze"(%1912) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1914 = "mix.prim.slice"(%1913) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1915 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1916 = "mix.prim.mul"(%1914, %1915) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1917 = "mix.prim.sin"(%1910) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1918 = "mix.prim.slice"(%1917) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1919 = "mix.prim.unsqueeze"(%1918) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1920 = "mix.prim.slice"(%1919) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1921 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1922 = "mix.prim.mul"(%1920, %1921) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1923 = "mix.prim.slice"(%1916) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1924 = "mix.prim.slice"(%1922) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1925 = "mix.prim.mul"(%1893, %1923) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1926 = "mix.prim.slice"(%1893) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1927 = "mix.prim.slice"(%1893) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1928 = "mix.prim.neg"(%1927) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1929 = "mix.prim.concat"(%1928, %1926) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1930 = "mix.prim.mul"(%1929, %1924) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1931 = "mix.prim.add"(%1925, %1930) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1932 = "mix.prim.mul"(%1894, %1923) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1933 = "mix.prim.slice"(%1894) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1934 = "mix.prim.slice"(%1894) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1935 = "mix.prim.neg"(%1934) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1936 = "mix.prim.concat"(%1935, %1933) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1937 = "mix.prim.mul"(%1936, %1924) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1938 = "mix.prim.add"(%1932, %1937) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1939 = "mix.prim.reshape"(%1931) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1940 = "mix.prim.reshape"(%1938) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1941 = "mix.prim.reshape"(%1939) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1942 = "mix.prim.reshape"(%1940) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1943 = "mix.prim.transpose"(%1941) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1944 = "mix.prim.transpose"(%1942) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1945 = "mix.prim.transpose"(%1944) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1946 = "mix.prim.unsqueeze"(%1943) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1947 = "mix.prim.permute"(%1946) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1948 = "mix.prim.unsqueeze"(%1945) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1949 = "mix.prim.permute"(%1948) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1950 = "mix.prim.permute"(%1947) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1951 = "mix.prim.reshape"(%1950) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1952 = "mix.prim.permute"(%1949) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1953 = "mix.prim.reshape"(%1952) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1954 = "mix.prim.batch_matmul"(%1951, %1953) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1955 = "mix.prim.reshape"(%1954) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1956 = "mix.prim.permute"(%1955) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1957 = "mix.prim.reshape"(%1956) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1958 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1959 = "mix.prim.mul"(%1957, %1958) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1960 = "mix.prim.reshape"(%1959) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1961 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1962 = "mix.comp.masked_fill"(%1960, %4, %1961) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1963 = "mix.comp.softmax"(%1962) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1964 = "mix.prim.reshape"(%1963) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1965 = "mix.prim.reshape"(%1892) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1966 = "mix.prim.transpose"(%1965) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1967 = "mix.prim.batch_matmul"(%1964, %1966) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1968 = "mix.prim.reshape"(%1967) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1969 = "mix.prim.permute"(%1968) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1970 = "mix.prim.reshape"(%1969) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1971 = "mix.prim.reshape"(%1970) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1972 = "mix.prim.transpose"(%1878) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1973 = "mix.prim.matmul"(%1971, %1972) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1974 = "mix.prim.add"(%1973, %1879) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1975 = "mix.prim.reshape"(%1974) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1976 = "mix.prim.mul"(%1866, %1975) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1977 = "mix.comp.weight"() <{param_loc = "transformer.h.14.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1978 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1979 = "mix.prim.pow"(%1976, %1978) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1980 = "mix.comp.mean"(%1979) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1981 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1982 = "mix.prim.add"(%1980, %1981) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1983 = "mix.prim.rsqrt"(%1982) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1984 = "mix.prim.mul"(%1976, %1983) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1985 = "mix.prim.mul"(%1977, %1984) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1986 = "mix.comp.weight"() <{param_loc = "transformer.h.14.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1987 = "mix.prim.reshape"(%1986) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1988 = "mix.prim.batch_matmul"(%1985, %1987) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1989 = "mix.comp.silu"(%1988) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1990 = "mix.comp.weight"() <{param_loc = "transformer.h.14.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %1991 = "mix.prim.reshape"(%1990) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %1992 = "mix.prim.batch_matmul"(%1985, %1991) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %1993 = "mix.prim.mul"(%1989, %1992) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1994 = "mix.comp.weight"() <{param_loc = "transformer.h.14.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %1995 = "mix.prim.reshape"(%1994) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %1996 = "mix.prim.batch_matmul"(%1993, %1995) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %1997 = "mix.comp.weight"() <{param_loc = "transformer.h.14.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %1998 = "mix.prim.add"(%1996, %1997) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %1999 = "mix.prim.add"(%1998, %1976) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2000 = "mix.comp.weight"() <{param_loc = "transformer.h.15.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2001 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2002 = "mix.prim.pow"(%1999, %2001) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2003 = "mix.comp.mean"(%2002) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2004 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2005 = "mix.prim.add"(%2003, %2004) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2006 = "mix.prim.rsqrt"(%2005) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2007 = "mix.prim.mul"(%1999, %2006) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2008 = "mix.prim.mul"(%2000, %2007) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2009 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2010 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2011 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2012 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2013 = "mix.prim.transpose"(%2008) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2014 = "mix.prim.transpose"(%2009) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2015 = "mix.prim.reshape"(%2013) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2016 = "mix.prim.matmul"(%2015, %2014) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2017 = "mix.prim.reshape"(%2016) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2018 = "mix.prim.reshape"(%2017) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2019 = "mix.prim.transpose"(%2010) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2020 = "mix.prim.reshape"(%2013) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2021 = "mix.prim.matmul"(%2020, %2019) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2022 = "mix.prim.reshape"(%2021) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2023 = "mix.prim.reshape"(%2022) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2024 = "mix.prim.slice"(%2023) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2025 = "mix.prim.slice"(%2023) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2026 = "mix.prim.reshape"(%2018) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2027 = "mix.prim.reshape"(%2024) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2028 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2029 = "mix.prim.convert"(%2028) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2030 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2031 = "mix.prim.div"(%2029, %2030) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2032 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2033 = "mix.prim.pow"(%2032, %2031) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2034 = "mix.prim.reciprocal"(%2033) : (tensor<80xf16>) -> tensor<80xf16>
    %2035 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2036 = "mix.prim.mul"(%2035, %2034) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2037 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2038 = "mix.prim.unsqueeze"(%2037) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2039 = "mix.prim.permute"(%2038) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2040 = "mix.prim.unsqueeze"(%2036) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2041 = "mix.prim.permute"(%2040) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2042 = "mix.prim.mul"(%2039, %2041) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2043 = "mix.prim.concat"(%2042, %2042) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2044 = "mix.prim.cos"(%2043) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2045 = "mix.prim.slice"(%2044) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2046 = "mix.prim.unsqueeze"(%2045) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2047 = "mix.prim.slice"(%2046) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2048 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2049 = "mix.prim.mul"(%2047, %2048) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2050 = "mix.prim.sin"(%2043) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2051 = "mix.prim.slice"(%2050) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2052 = "mix.prim.unsqueeze"(%2051) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2053 = "mix.prim.slice"(%2052) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2054 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2055 = "mix.prim.mul"(%2053, %2054) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2056 = "mix.prim.slice"(%2049) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2057 = "mix.prim.slice"(%2055) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2058 = "mix.prim.mul"(%2026, %2056) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2059 = "mix.prim.slice"(%2026) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2060 = "mix.prim.slice"(%2026) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2061 = "mix.prim.neg"(%2060) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2062 = "mix.prim.concat"(%2061, %2059) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2063 = "mix.prim.mul"(%2062, %2057) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2064 = "mix.prim.add"(%2058, %2063) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2065 = "mix.prim.mul"(%2027, %2056) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2066 = "mix.prim.slice"(%2027) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2067 = "mix.prim.slice"(%2027) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2068 = "mix.prim.neg"(%2067) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2069 = "mix.prim.concat"(%2068, %2066) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2070 = "mix.prim.mul"(%2069, %2057) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2071 = "mix.prim.add"(%2065, %2070) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2072 = "mix.prim.reshape"(%2064) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2073 = "mix.prim.reshape"(%2071) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2074 = "mix.prim.reshape"(%2072) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2075 = "mix.prim.reshape"(%2073) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2076 = "mix.prim.transpose"(%2074) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2077 = "mix.prim.transpose"(%2075) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2078 = "mix.prim.transpose"(%2077) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2079 = "mix.prim.unsqueeze"(%2076) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2080 = "mix.prim.permute"(%2079) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2081 = "mix.prim.unsqueeze"(%2078) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2082 = "mix.prim.permute"(%2081) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2083 = "mix.prim.permute"(%2080) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2084 = "mix.prim.reshape"(%2083) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2085 = "mix.prim.permute"(%2082) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2086 = "mix.prim.reshape"(%2085) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2087 = "mix.prim.batch_matmul"(%2084, %2086) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2088 = "mix.prim.reshape"(%2087) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2089 = "mix.prim.permute"(%2088) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2090 = "mix.prim.reshape"(%2089) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2091 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2092 = "mix.prim.mul"(%2090, %2091) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2093 = "mix.prim.reshape"(%2092) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2094 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2095 = "mix.comp.masked_fill"(%2093, %4, %2094) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2096 = "mix.comp.softmax"(%2095) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2097 = "mix.prim.reshape"(%2096) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2098 = "mix.prim.reshape"(%2025) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2099 = "mix.prim.transpose"(%2098) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2100 = "mix.prim.batch_matmul"(%2097, %2099) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2101 = "mix.prim.reshape"(%2100) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2102 = "mix.prim.permute"(%2101) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2103 = "mix.prim.reshape"(%2102) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2104 = "mix.prim.reshape"(%2103) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2105 = "mix.prim.transpose"(%2011) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2106 = "mix.prim.matmul"(%2104, %2105) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2107 = "mix.prim.add"(%2106, %2012) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2108 = "mix.prim.reshape"(%2107) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2109 = "mix.prim.mul"(%1999, %2108) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2110 = "mix.comp.weight"() <{param_loc = "transformer.h.15.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2111 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2112 = "mix.prim.pow"(%2109, %2111) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2113 = "mix.comp.mean"(%2112) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2114 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2115 = "mix.prim.add"(%2113, %2114) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2116 = "mix.prim.rsqrt"(%2115) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2117 = "mix.prim.mul"(%2109, %2116) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2118 = "mix.prim.mul"(%2110, %2117) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2119 = "mix.comp.weight"() <{param_loc = "transformer.h.15.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2120 = "mix.prim.reshape"(%2119) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2121 = "mix.prim.batch_matmul"(%2118, %2120) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2122 = "mix.comp.silu"(%2121) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2123 = "mix.comp.weight"() <{param_loc = "transformer.h.15.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2124 = "mix.prim.reshape"(%2123) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2125 = "mix.prim.batch_matmul"(%2118, %2124) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2126 = "mix.prim.mul"(%2122, %2125) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2127 = "mix.comp.weight"() <{param_loc = "transformer.h.15.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %2128 = "mix.prim.reshape"(%2127) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %2129 = "mix.prim.batch_matmul"(%2126, %2128) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %2130 = "mix.comp.weight"() <{param_loc = "transformer.h.15.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %2131 = "mix.prim.add"(%2129, %2130) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %2132 = "mix.prim.add"(%2131, %2109) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2133 = "mix.comp.weight"() <{param_loc = "transformer.h.16.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2134 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2135 = "mix.prim.pow"(%2132, %2134) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2136 = "mix.comp.mean"(%2135) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2137 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2138 = "mix.prim.add"(%2136, %2137) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2139 = "mix.prim.rsqrt"(%2138) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2140 = "mix.prim.mul"(%2132, %2139) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2141 = "mix.prim.mul"(%2133, %2140) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2142 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2143 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2144 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2145 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2146 = "mix.prim.transpose"(%2141) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2147 = "mix.prim.transpose"(%2142) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2148 = "mix.prim.reshape"(%2146) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2149 = "mix.prim.matmul"(%2148, %2147) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2150 = "mix.prim.reshape"(%2149) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2151 = "mix.prim.reshape"(%2150) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2152 = "mix.prim.transpose"(%2143) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2153 = "mix.prim.reshape"(%2146) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2154 = "mix.prim.matmul"(%2153, %2152) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2155 = "mix.prim.reshape"(%2154) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2156 = "mix.prim.reshape"(%2155) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2157 = "mix.prim.slice"(%2156) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2158 = "mix.prim.slice"(%2156) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2159 = "mix.prim.reshape"(%2151) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2160 = "mix.prim.reshape"(%2157) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2161 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2162 = "mix.prim.convert"(%2161) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2163 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2164 = "mix.prim.div"(%2162, %2163) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2165 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2166 = "mix.prim.pow"(%2165, %2164) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2167 = "mix.prim.reciprocal"(%2166) : (tensor<80xf16>) -> tensor<80xf16>
    %2168 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2169 = "mix.prim.mul"(%2168, %2167) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2170 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2171 = "mix.prim.unsqueeze"(%2170) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2172 = "mix.prim.permute"(%2171) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2173 = "mix.prim.unsqueeze"(%2169) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2174 = "mix.prim.permute"(%2173) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2175 = "mix.prim.mul"(%2172, %2174) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2176 = "mix.prim.concat"(%2175, %2175) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2177 = "mix.prim.cos"(%2176) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2178 = "mix.prim.slice"(%2177) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2179 = "mix.prim.unsqueeze"(%2178) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2180 = "mix.prim.slice"(%2179) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2181 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2182 = "mix.prim.mul"(%2180, %2181) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2183 = "mix.prim.sin"(%2176) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2184 = "mix.prim.slice"(%2183) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2185 = "mix.prim.unsqueeze"(%2184) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2186 = "mix.prim.slice"(%2185) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2187 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2188 = "mix.prim.mul"(%2186, %2187) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2189 = "mix.prim.slice"(%2182) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2190 = "mix.prim.slice"(%2188) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2191 = "mix.prim.mul"(%2159, %2189) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2192 = "mix.prim.slice"(%2159) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2193 = "mix.prim.slice"(%2159) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2194 = "mix.prim.neg"(%2193) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2195 = "mix.prim.concat"(%2194, %2192) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2196 = "mix.prim.mul"(%2195, %2190) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2197 = "mix.prim.add"(%2191, %2196) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2198 = "mix.prim.mul"(%2160, %2189) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2199 = "mix.prim.slice"(%2160) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2200 = "mix.prim.slice"(%2160) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2201 = "mix.prim.neg"(%2200) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2202 = "mix.prim.concat"(%2201, %2199) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2203 = "mix.prim.mul"(%2202, %2190) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2204 = "mix.prim.add"(%2198, %2203) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2205 = "mix.prim.reshape"(%2197) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2206 = "mix.prim.reshape"(%2204) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2207 = "mix.prim.reshape"(%2205) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2208 = "mix.prim.reshape"(%2206) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2209 = "mix.prim.transpose"(%2207) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2210 = "mix.prim.transpose"(%2208) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2211 = "mix.prim.transpose"(%2210) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2212 = "mix.prim.unsqueeze"(%2209) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2213 = "mix.prim.permute"(%2212) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2214 = "mix.prim.unsqueeze"(%2211) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2215 = "mix.prim.permute"(%2214) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2216 = "mix.prim.permute"(%2213) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2217 = "mix.prim.reshape"(%2216) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2218 = "mix.prim.permute"(%2215) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2219 = "mix.prim.reshape"(%2218) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2220 = "mix.prim.batch_matmul"(%2217, %2219) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2221 = "mix.prim.reshape"(%2220) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2222 = "mix.prim.permute"(%2221) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2223 = "mix.prim.reshape"(%2222) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2224 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2225 = "mix.prim.mul"(%2223, %2224) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2226 = "mix.prim.reshape"(%2225) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2227 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2228 = "mix.comp.masked_fill"(%2226, %4, %2227) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2229 = "mix.comp.softmax"(%2228) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2230 = "mix.prim.reshape"(%2229) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2231 = "mix.prim.reshape"(%2158) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2232 = "mix.prim.transpose"(%2231) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2233 = "mix.prim.batch_matmul"(%2230, %2232) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2234 = "mix.prim.reshape"(%2233) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2235 = "mix.prim.permute"(%2234) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2236 = "mix.prim.reshape"(%2235) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2237 = "mix.prim.reshape"(%2236) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2238 = "mix.prim.transpose"(%2144) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2239 = "mix.prim.matmul"(%2237, %2238) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2240 = "mix.prim.add"(%2239, %2145) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2241 = "mix.prim.reshape"(%2240) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2242 = "mix.prim.mul"(%2132, %2241) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2243 = "mix.comp.weight"() <{param_loc = "transformer.h.16.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2244 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2245 = "mix.prim.pow"(%2242, %2244) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2246 = "mix.comp.mean"(%2245) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2247 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2248 = "mix.prim.add"(%2246, %2247) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2249 = "mix.prim.rsqrt"(%2248) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2250 = "mix.prim.mul"(%2242, %2249) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2251 = "mix.prim.mul"(%2243, %2250) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2252 = "mix.comp.weight"() <{param_loc = "transformer.h.16.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2253 = "mix.prim.reshape"(%2252) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2254 = "mix.prim.batch_matmul"(%2251, %2253) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2255 = "mix.comp.silu"(%2254) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2256 = "mix.comp.weight"() <{param_loc = "transformer.h.16.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2257 = "mix.prim.reshape"(%2256) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2258 = "mix.prim.batch_matmul"(%2251, %2257) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2259 = "mix.prim.mul"(%2255, %2258) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2260 = "mix.comp.weight"() <{param_loc = "transformer.h.16.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %2261 = "mix.prim.reshape"(%2260) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %2262 = "mix.prim.batch_matmul"(%2259, %2261) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %2263 = "mix.comp.weight"() <{param_loc = "transformer.h.16.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %2264 = "mix.prim.add"(%2262, %2263) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %2265 = "mix.prim.add"(%2264, %2242) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2266 = "mix.comp.weight"() <{param_loc = "transformer.h.17.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2267 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2268 = "mix.prim.pow"(%2265, %2267) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2269 = "mix.comp.mean"(%2268) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2270 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2271 = "mix.prim.add"(%2269, %2270) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2272 = "mix.prim.rsqrt"(%2271) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2273 = "mix.prim.mul"(%2265, %2272) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2274 = "mix.prim.mul"(%2266, %2273) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2275 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2276 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2277 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2278 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2279 = "mix.prim.transpose"(%2274) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2280 = "mix.prim.transpose"(%2275) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2281 = "mix.prim.reshape"(%2279) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2282 = "mix.prim.matmul"(%2281, %2280) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2283 = "mix.prim.reshape"(%2282) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2284 = "mix.prim.reshape"(%2283) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2285 = "mix.prim.transpose"(%2276) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2286 = "mix.prim.reshape"(%2279) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2287 = "mix.prim.matmul"(%2286, %2285) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2288 = "mix.prim.reshape"(%2287) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2289 = "mix.prim.reshape"(%2288) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2290 = "mix.prim.slice"(%2289) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2291 = "mix.prim.slice"(%2289) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2292 = "mix.prim.reshape"(%2284) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2293 = "mix.prim.reshape"(%2290) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2294 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2295 = "mix.prim.convert"(%2294) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2296 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2297 = "mix.prim.div"(%2295, %2296) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2298 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2299 = "mix.prim.pow"(%2298, %2297) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2300 = "mix.prim.reciprocal"(%2299) : (tensor<80xf16>) -> tensor<80xf16>
    %2301 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2302 = "mix.prim.mul"(%2301, %2300) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2303 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2304 = "mix.prim.unsqueeze"(%2303) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2305 = "mix.prim.permute"(%2304) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2306 = "mix.prim.unsqueeze"(%2302) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2307 = "mix.prim.permute"(%2306) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2308 = "mix.prim.mul"(%2305, %2307) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2309 = "mix.prim.concat"(%2308, %2308) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2310 = "mix.prim.cos"(%2309) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2311 = "mix.prim.slice"(%2310) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2312 = "mix.prim.unsqueeze"(%2311) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2313 = "mix.prim.slice"(%2312) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2314 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2315 = "mix.prim.mul"(%2313, %2314) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2316 = "mix.prim.sin"(%2309) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2317 = "mix.prim.slice"(%2316) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2318 = "mix.prim.unsqueeze"(%2317) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2319 = "mix.prim.slice"(%2318) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2320 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2321 = "mix.prim.mul"(%2319, %2320) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2322 = "mix.prim.slice"(%2315) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2323 = "mix.prim.slice"(%2321) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2324 = "mix.prim.mul"(%2292, %2322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2325 = "mix.prim.slice"(%2292) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2326 = "mix.prim.slice"(%2292) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2327 = "mix.prim.neg"(%2326) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2328 = "mix.prim.concat"(%2327, %2325) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2329 = "mix.prim.mul"(%2328, %2323) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2330 = "mix.prim.add"(%2324, %2329) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2331 = "mix.prim.mul"(%2293, %2322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2332 = "mix.prim.slice"(%2293) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2333 = "mix.prim.slice"(%2293) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2334 = "mix.prim.neg"(%2333) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2335 = "mix.prim.concat"(%2334, %2332) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2336 = "mix.prim.mul"(%2335, %2323) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2337 = "mix.prim.add"(%2331, %2336) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2338 = "mix.prim.reshape"(%2330) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2339 = "mix.prim.reshape"(%2337) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2340 = "mix.prim.reshape"(%2338) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2341 = "mix.prim.reshape"(%2339) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2342 = "mix.prim.transpose"(%2340) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2343 = "mix.prim.transpose"(%2341) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2344 = "mix.prim.transpose"(%2343) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2345 = "mix.prim.unsqueeze"(%2342) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2346 = "mix.prim.permute"(%2345) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2347 = "mix.prim.unsqueeze"(%2344) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2348 = "mix.prim.permute"(%2347) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2349 = "mix.prim.permute"(%2346) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2350 = "mix.prim.reshape"(%2349) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2351 = "mix.prim.permute"(%2348) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2352 = "mix.prim.reshape"(%2351) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2353 = "mix.prim.batch_matmul"(%2350, %2352) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2354 = "mix.prim.reshape"(%2353) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2355 = "mix.prim.permute"(%2354) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2356 = "mix.prim.reshape"(%2355) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2357 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2358 = "mix.prim.mul"(%2356, %2357) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2359 = "mix.prim.reshape"(%2358) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2360 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2361 = "mix.comp.masked_fill"(%2359, %4, %2360) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2362 = "mix.comp.softmax"(%2361) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2363 = "mix.prim.reshape"(%2362) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2364 = "mix.prim.reshape"(%2291) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2365 = "mix.prim.transpose"(%2364) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2366 = "mix.prim.batch_matmul"(%2363, %2365) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2367 = "mix.prim.reshape"(%2366) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2368 = "mix.prim.permute"(%2367) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2369 = "mix.prim.reshape"(%2368) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2370 = "mix.prim.reshape"(%2369) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2371 = "mix.prim.transpose"(%2277) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2372 = "mix.prim.matmul"(%2370, %2371) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2373 = "mix.prim.add"(%2372, %2278) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2374 = "mix.prim.reshape"(%2373) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2375 = "mix.prim.mul"(%2265, %2374) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2376 = "mix.comp.weight"() <{param_loc = "transformer.h.17.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2377 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2378 = "mix.prim.pow"(%2375, %2377) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2379 = "mix.comp.mean"(%2378) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2380 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2381 = "mix.prim.add"(%2379, %2380) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2382 = "mix.prim.rsqrt"(%2381) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2383 = "mix.prim.mul"(%2375, %2382) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2384 = "mix.prim.mul"(%2376, %2383) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2385 = "mix.comp.weight"() <{param_loc = "transformer.h.17.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2386 = "mix.prim.reshape"(%2385) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2387 = "mix.prim.batch_matmul"(%2384, %2386) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2388 = "mix.comp.silu"(%2387) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2389 = "mix.comp.weight"() <{param_loc = "transformer.h.17.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2390 = "mix.prim.reshape"(%2389) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2391 = "mix.prim.batch_matmul"(%2384, %2390) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2392 = "mix.prim.mul"(%2388, %2391) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2393 = "mix.comp.weight"() <{param_loc = "transformer.h.17.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %2394 = "mix.prim.reshape"(%2393) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %2395 = "mix.prim.batch_matmul"(%2392, %2394) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %2396 = "mix.comp.weight"() <{param_loc = "transformer.h.17.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %2397 = "mix.prim.add"(%2395, %2396) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %2398 = "mix.prim.add"(%2397, %2375) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2399 = "mix.comp.weight"() <{param_loc = "transformer.h.18.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2400 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2401 = "mix.prim.pow"(%2398, %2400) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2402 = "mix.comp.mean"(%2401) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2403 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2404 = "mix.prim.add"(%2402, %2403) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2405 = "mix.prim.rsqrt"(%2404) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2406 = "mix.prim.mul"(%2398, %2405) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2407 = "mix.prim.mul"(%2399, %2406) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2408 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2409 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2410 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2411 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2412 = "mix.prim.transpose"(%2407) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2413 = "mix.prim.transpose"(%2408) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2414 = "mix.prim.reshape"(%2412) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2415 = "mix.prim.matmul"(%2414, %2413) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2416 = "mix.prim.reshape"(%2415) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2417 = "mix.prim.reshape"(%2416) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2418 = "mix.prim.transpose"(%2409) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2419 = "mix.prim.reshape"(%2412) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2420 = "mix.prim.matmul"(%2419, %2418) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2421 = "mix.prim.reshape"(%2420) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2422 = "mix.prim.reshape"(%2421) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2423 = "mix.prim.slice"(%2422) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2424 = "mix.prim.slice"(%2422) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2425 = "mix.prim.reshape"(%2417) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2426 = "mix.prim.reshape"(%2423) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2427 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2428 = "mix.prim.convert"(%2427) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2429 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2430 = "mix.prim.div"(%2428, %2429) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2431 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2432 = "mix.prim.pow"(%2431, %2430) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2433 = "mix.prim.reciprocal"(%2432) : (tensor<80xf16>) -> tensor<80xf16>
    %2434 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2435 = "mix.prim.mul"(%2434, %2433) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2436 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2437 = "mix.prim.unsqueeze"(%2436) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2438 = "mix.prim.permute"(%2437) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2439 = "mix.prim.unsqueeze"(%2435) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2440 = "mix.prim.permute"(%2439) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2441 = "mix.prim.mul"(%2438, %2440) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2442 = "mix.prim.concat"(%2441, %2441) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2443 = "mix.prim.cos"(%2442) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2444 = "mix.prim.slice"(%2443) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2445 = "mix.prim.unsqueeze"(%2444) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2446 = "mix.prim.slice"(%2445) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2447 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2448 = "mix.prim.mul"(%2446, %2447) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2449 = "mix.prim.sin"(%2442) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2450 = "mix.prim.slice"(%2449) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2451 = "mix.prim.unsqueeze"(%2450) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2452 = "mix.prim.slice"(%2451) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2453 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2454 = "mix.prim.mul"(%2452, %2453) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2455 = "mix.prim.slice"(%2448) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2456 = "mix.prim.slice"(%2454) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2457 = "mix.prim.mul"(%2425, %2455) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2458 = "mix.prim.slice"(%2425) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2459 = "mix.prim.slice"(%2425) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2460 = "mix.prim.neg"(%2459) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2461 = "mix.prim.concat"(%2460, %2458) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2462 = "mix.prim.mul"(%2461, %2456) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2463 = "mix.prim.add"(%2457, %2462) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2464 = "mix.prim.mul"(%2426, %2455) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2465 = "mix.prim.slice"(%2426) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2466 = "mix.prim.slice"(%2426) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2467 = "mix.prim.neg"(%2466) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2468 = "mix.prim.concat"(%2467, %2465) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2469 = "mix.prim.mul"(%2468, %2456) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2470 = "mix.prim.add"(%2464, %2469) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2471 = "mix.prim.reshape"(%2463) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2472 = "mix.prim.reshape"(%2470) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2473 = "mix.prim.reshape"(%2471) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2474 = "mix.prim.reshape"(%2472) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2475 = "mix.prim.transpose"(%2473) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2476 = "mix.prim.transpose"(%2474) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2477 = "mix.prim.transpose"(%2476) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2478 = "mix.prim.unsqueeze"(%2475) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2479 = "mix.prim.permute"(%2478) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2480 = "mix.prim.unsqueeze"(%2477) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2481 = "mix.prim.permute"(%2480) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2482 = "mix.prim.permute"(%2479) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2483 = "mix.prim.reshape"(%2482) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2484 = "mix.prim.permute"(%2481) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2485 = "mix.prim.reshape"(%2484) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2486 = "mix.prim.batch_matmul"(%2483, %2485) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2487 = "mix.prim.reshape"(%2486) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2488 = "mix.prim.permute"(%2487) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2489 = "mix.prim.reshape"(%2488) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2490 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2491 = "mix.prim.mul"(%2489, %2490) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2492 = "mix.prim.reshape"(%2491) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2493 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2494 = "mix.comp.masked_fill"(%2492, %4, %2493) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2495 = "mix.comp.softmax"(%2494) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2496 = "mix.prim.reshape"(%2495) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2497 = "mix.prim.reshape"(%2424) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2498 = "mix.prim.transpose"(%2497) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2499 = "mix.prim.batch_matmul"(%2496, %2498) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2500 = "mix.prim.reshape"(%2499) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2501 = "mix.prim.permute"(%2500) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2502 = "mix.prim.reshape"(%2501) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2503 = "mix.prim.reshape"(%2502) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2504 = "mix.prim.transpose"(%2410) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2505 = "mix.prim.matmul"(%2503, %2504) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2506 = "mix.prim.add"(%2505, %2411) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2507 = "mix.prim.reshape"(%2506) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2508 = "mix.prim.mul"(%2398, %2507) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2509 = "mix.comp.weight"() <{param_loc = "transformer.h.18.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2510 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2511 = "mix.prim.pow"(%2508, %2510) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2512 = "mix.comp.mean"(%2511) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2513 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2514 = "mix.prim.add"(%2512, %2513) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2515 = "mix.prim.rsqrt"(%2514) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2516 = "mix.prim.mul"(%2508, %2515) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2517 = "mix.prim.mul"(%2509, %2516) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2518 = "mix.comp.weight"() <{param_loc = "transformer.h.18.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2519 = "mix.prim.reshape"(%2518) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2520 = "mix.prim.batch_matmul"(%2517, %2519) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2521 = "mix.comp.silu"(%2520) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2522 = "mix.comp.weight"() <{param_loc = "transformer.h.18.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2523 = "mix.prim.reshape"(%2522) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2524 = "mix.prim.batch_matmul"(%2517, %2523) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2525 = "mix.prim.mul"(%2521, %2524) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2526 = "mix.comp.weight"() <{param_loc = "transformer.h.18.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %2527 = "mix.prim.reshape"(%2526) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %2528 = "mix.prim.batch_matmul"(%2525, %2527) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %2529 = "mix.comp.weight"() <{param_loc = "transformer.h.18.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %2530 = "mix.prim.add"(%2528, %2529) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %2531 = "mix.prim.add"(%2530, %2508) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2532 = "mix.comp.weight"() <{param_loc = "transformer.h.19.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2533 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2534 = "mix.prim.pow"(%2531, %2533) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2535 = "mix.comp.mean"(%2534) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2536 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2537 = "mix.prim.add"(%2535, %2536) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2538 = "mix.prim.rsqrt"(%2537) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2539 = "mix.prim.mul"(%2531, %2538) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2540 = "mix.prim.mul"(%2532, %2539) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2541 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2542 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2543 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2544 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2545 = "mix.prim.transpose"(%2540) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2546 = "mix.prim.transpose"(%2541) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2547 = "mix.prim.reshape"(%2545) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2548 = "mix.prim.matmul"(%2547, %2546) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2549 = "mix.prim.reshape"(%2548) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2550 = "mix.prim.reshape"(%2549) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2551 = "mix.prim.transpose"(%2542) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2552 = "mix.prim.reshape"(%2545) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2553 = "mix.prim.matmul"(%2552, %2551) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2554 = "mix.prim.reshape"(%2553) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2555 = "mix.prim.reshape"(%2554) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2556 = "mix.prim.slice"(%2555) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2557 = "mix.prim.slice"(%2555) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2558 = "mix.prim.reshape"(%2550) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2559 = "mix.prim.reshape"(%2556) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2560 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2561 = "mix.prim.convert"(%2560) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2562 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2563 = "mix.prim.div"(%2561, %2562) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2564 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2565 = "mix.prim.pow"(%2564, %2563) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2566 = "mix.prim.reciprocal"(%2565) : (tensor<80xf16>) -> tensor<80xf16>
    %2567 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2568 = "mix.prim.mul"(%2567, %2566) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2569 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2570 = "mix.prim.unsqueeze"(%2569) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2571 = "mix.prim.permute"(%2570) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2572 = "mix.prim.unsqueeze"(%2568) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2573 = "mix.prim.permute"(%2572) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2574 = "mix.prim.mul"(%2571, %2573) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2575 = "mix.prim.concat"(%2574, %2574) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2576 = "mix.prim.cos"(%2575) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2577 = "mix.prim.slice"(%2576) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2578 = "mix.prim.unsqueeze"(%2577) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2579 = "mix.prim.slice"(%2578) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2580 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2581 = "mix.prim.mul"(%2579, %2580) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2582 = "mix.prim.sin"(%2575) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2583 = "mix.prim.slice"(%2582) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2584 = "mix.prim.unsqueeze"(%2583) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2585 = "mix.prim.slice"(%2584) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2586 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2587 = "mix.prim.mul"(%2585, %2586) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2588 = "mix.prim.slice"(%2581) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2589 = "mix.prim.slice"(%2587) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2590 = "mix.prim.mul"(%2558, %2588) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2591 = "mix.prim.slice"(%2558) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2592 = "mix.prim.slice"(%2558) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2593 = "mix.prim.neg"(%2592) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2594 = "mix.prim.concat"(%2593, %2591) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2595 = "mix.prim.mul"(%2594, %2589) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2596 = "mix.prim.add"(%2590, %2595) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2597 = "mix.prim.mul"(%2559, %2588) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2598 = "mix.prim.slice"(%2559) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2599 = "mix.prim.slice"(%2559) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2600 = "mix.prim.neg"(%2599) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2601 = "mix.prim.concat"(%2600, %2598) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2602 = "mix.prim.mul"(%2601, %2589) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2603 = "mix.prim.add"(%2597, %2602) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2604 = "mix.prim.reshape"(%2596) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2605 = "mix.prim.reshape"(%2603) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2606 = "mix.prim.reshape"(%2604) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2607 = "mix.prim.reshape"(%2605) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2608 = "mix.prim.transpose"(%2606) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2609 = "mix.prim.transpose"(%2607) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2610 = "mix.prim.transpose"(%2609) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2611 = "mix.prim.unsqueeze"(%2608) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2612 = "mix.prim.permute"(%2611) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2613 = "mix.prim.unsqueeze"(%2610) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2614 = "mix.prim.permute"(%2613) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2615 = "mix.prim.permute"(%2612) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2616 = "mix.prim.reshape"(%2615) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2617 = "mix.prim.permute"(%2614) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2618 = "mix.prim.reshape"(%2617) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2619 = "mix.prim.batch_matmul"(%2616, %2618) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2620 = "mix.prim.reshape"(%2619) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2621 = "mix.prim.permute"(%2620) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2622 = "mix.prim.reshape"(%2621) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2623 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2624 = "mix.prim.mul"(%2622, %2623) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2625 = "mix.prim.reshape"(%2624) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2626 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2627 = "mix.comp.masked_fill"(%2625, %4, %2626) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2628 = "mix.comp.softmax"(%2627) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2629 = "mix.prim.reshape"(%2628) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2630 = "mix.prim.reshape"(%2557) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2631 = "mix.prim.transpose"(%2630) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2632 = "mix.prim.batch_matmul"(%2629, %2631) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2633 = "mix.prim.reshape"(%2632) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2634 = "mix.prim.permute"(%2633) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2635 = "mix.prim.reshape"(%2634) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2636 = "mix.prim.reshape"(%2635) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2637 = "mix.prim.transpose"(%2543) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2638 = "mix.prim.matmul"(%2636, %2637) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2639 = "mix.prim.add"(%2638, %2544) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2640 = "mix.prim.reshape"(%2639) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2641 = "mix.prim.mul"(%2531, %2640) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2642 = "mix.comp.weight"() <{param_loc = "transformer.h.19.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2643 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2644 = "mix.prim.pow"(%2641, %2643) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2645 = "mix.comp.mean"(%2644) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2646 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2647 = "mix.prim.add"(%2645, %2646) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2648 = "mix.prim.rsqrt"(%2647) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2649 = "mix.prim.mul"(%2641, %2648) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2650 = "mix.prim.mul"(%2642, %2649) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2651 = "mix.comp.weight"() <{param_loc = "transformer.h.19.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2652 = "mix.prim.reshape"(%2651) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2653 = "mix.prim.batch_matmul"(%2650, %2652) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2654 = "mix.comp.silu"(%2653) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2655 = "mix.comp.weight"() <{param_loc = "transformer.h.19.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2656 = "mix.prim.reshape"(%2655) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2657 = "mix.prim.batch_matmul"(%2650, %2656) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2658 = "mix.prim.mul"(%2654, %2657) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2659 = "mix.comp.weight"() <{param_loc = "transformer.h.19.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %2660 = "mix.prim.reshape"(%2659) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %2661 = "mix.prim.batch_matmul"(%2658, %2660) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %2662 = "mix.comp.weight"() <{param_loc = "transformer.h.19.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %2663 = "mix.prim.add"(%2661, %2662) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %2664 = "mix.prim.add"(%2663, %2641) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2665 = "mix.comp.weight"() <{param_loc = "transformer.h.20.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2666 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2667 = "mix.prim.pow"(%2664, %2666) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2668 = "mix.comp.mean"(%2667) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2669 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2670 = "mix.prim.add"(%2668, %2669) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2671 = "mix.prim.rsqrt"(%2670) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2672 = "mix.prim.mul"(%2664, %2671) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2673 = "mix.prim.mul"(%2665, %2672) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2674 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2675 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2676 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2677 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2678 = "mix.prim.transpose"(%2673) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2679 = "mix.prim.transpose"(%2674) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2680 = "mix.prim.reshape"(%2678) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2681 = "mix.prim.matmul"(%2680, %2679) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2682 = "mix.prim.reshape"(%2681) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2683 = "mix.prim.reshape"(%2682) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2684 = "mix.prim.transpose"(%2675) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2685 = "mix.prim.reshape"(%2678) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2686 = "mix.prim.matmul"(%2685, %2684) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2687 = "mix.prim.reshape"(%2686) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2688 = "mix.prim.reshape"(%2687) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2689 = "mix.prim.slice"(%2688) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2690 = "mix.prim.slice"(%2688) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2691 = "mix.prim.reshape"(%2683) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2692 = "mix.prim.reshape"(%2689) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2693 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2694 = "mix.prim.convert"(%2693) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2695 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2696 = "mix.prim.div"(%2694, %2695) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2697 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2698 = "mix.prim.pow"(%2697, %2696) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2699 = "mix.prim.reciprocal"(%2698) : (tensor<80xf16>) -> tensor<80xf16>
    %2700 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2701 = "mix.prim.mul"(%2700, %2699) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2702 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2703 = "mix.prim.unsqueeze"(%2702) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2704 = "mix.prim.permute"(%2703) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2705 = "mix.prim.unsqueeze"(%2701) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2706 = "mix.prim.permute"(%2705) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2707 = "mix.prim.mul"(%2704, %2706) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2708 = "mix.prim.concat"(%2707, %2707) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2709 = "mix.prim.cos"(%2708) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2710 = "mix.prim.slice"(%2709) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2711 = "mix.prim.unsqueeze"(%2710) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2712 = "mix.prim.slice"(%2711) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2713 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2714 = "mix.prim.mul"(%2712, %2713) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2715 = "mix.prim.sin"(%2708) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2716 = "mix.prim.slice"(%2715) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2717 = "mix.prim.unsqueeze"(%2716) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2718 = "mix.prim.slice"(%2717) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2719 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2720 = "mix.prim.mul"(%2718, %2719) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2721 = "mix.prim.slice"(%2714) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2722 = "mix.prim.slice"(%2720) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2723 = "mix.prim.mul"(%2691, %2721) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2724 = "mix.prim.slice"(%2691) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2725 = "mix.prim.slice"(%2691) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2726 = "mix.prim.neg"(%2725) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2727 = "mix.prim.concat"(%2726, %2724) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2728 = "mix.prim.mul"(%2727, %2722) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2729 = "mix.prim.add"(%2723, %2728) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2730 = "mix.prim.mul"(%2692, %2721) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2731 = "mix.prim.slice"(%2692) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2732 = "mix.prim.slice"(%2692) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2733 = "mix.prim.neg"(%2732) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2734 = "mix.prim.concat"(%2733, %2731) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2735 = "mix.prim.mul"(%2734, %2722) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2736 = "mix.prim.add"(%2730, %2735) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2737 = "mix.prim.reshape"(%2729) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2738 = "mix.prim.reshape"(%2736) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2739 = "mix.prim.reshape"(%2737) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2740 = "mix.prim.reshape"(%2738) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2741 = "mix.prim.transpose"(%2739) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2742 = "mix.prim.transpose"(%2740) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2743 = "mix.prim.transpose"(%2742) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2744 = "mix.prim.unsqueeze"(%2741) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2745 = "mix.prim.permute"(%2744) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2746 = "mix.prim.unsqueeze"(%2743) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2747 = "mix.prim.permute"(%2746) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2748 = "mix.prim.permute"(%2745) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2749 = "mix.prim.reshape"(%2748) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2750 = "mix.prim.permute"(%2747) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2751 = "mix.prim.reshape"(%2750) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2752 = "mix.prim.batch_matmul"(%2749, %2751) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2753 = "mix.prim.reshape"(%2752) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2754 = "mix.prim.permute"(%2753) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2755 = "mix.prim.reshape"(%2754) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2756 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2757 = "mix.prim.mul"(%2755, %2756) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2758 = "mix.prim.reshape"(%2757) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2759 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2760 = "mix.comp.masked_fill"(%2758, %4, %2759) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2761 = "mix.comp.softmax"(%2760) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2762 = "mix.prim.reshape"(%2761) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2763 = "mix.prim.reshape"(%2690) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2764 = "mix.prim.transpose"(%2763) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2765 = "mix.prim.batch_matmul"(%2762, %2764) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2766 = "mix.prim.reshape"(%2765) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2767 = "mix.prim.permute"(%2766) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2768 = "mix.prim.reshape"(%2767) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2769 = "mix.prim.reshape"(%2768) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2770 = "mix.prim.transpose"(%2676) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2771 = "mix.prim.matmul"(%2769, %2770) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2772 = "mix.prim.add"(%2771, %2677) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2773 = "mix.prim.reshape"(%2772) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2774 = "mix.prim.mul"(%2664, %2773) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2775 = "mix.comp.weight"() <{param_loc = "transformer.h.20.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2776 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2777 = "mix.prim.pow"(%2774, %2776) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2778 = "mix.comp.mean"(%2777) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2779 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2780 = "mix.prim.add"(%2778, %2779) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2781 = "mix.prim.rsqrt"(%2780) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2782 = "mix.prim.mul"(%2774, %2781) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2783 = "mix.prim.mul"(%2775, %2782) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2784 = "mix.comp.weight"() <{param_loc = "transformer.h.20.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2785 = "mix.prim.reshape"(%2784) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2786 = "mix.prim.batch_matmul"(%2783, %2785) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2787 = "mix.comp.silu"(%2786) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2788 = "mix.comp.weight"() <{param_loc = "transformer.h.20.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2789 = "mix.prim.reshape"(%2788) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2790 = "mix.prim.batch_matmul"(%2783, %2789) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2791 = "mix.prim.mul"(%2787, %2790) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2792 = "mix.comp.weight"() <{param_loc = "transformer.h.20.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %2793 = "mix.prim.reshape"(%2792) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %2794 = "mix.prim.batch_matmul"(%2791, %2793) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %2795 = "mix.comp.weight"() <{param_loc = "transformer.h.20.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %2796 = "mix.prim.add"(%2794, %2795) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %2797 = "mix.prim.add"(%2796, %2774) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2798 = "mix.comp.weight"() <{param_loc = "transformer.h.21.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2799 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2800 = "mix.prim.pow"(%2797, %2799) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2801 = "mix.comp.mean"(%2800) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2802 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2803 = "mix.prim.add"(%2801, %2802) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2804 = "mix.prim.rsqrt"(%2803) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2805 = "mix.prim.mul"(%2797, %2804) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2806 = "mix.prim.mul"(%2798, %2805) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2807 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2808 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2809 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2810 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2811 = "mix.prim.transpose"(%2806) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2812 = "mix.prim.transpose"(%2807) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2813 = "mix.prim.reshape"(%2811) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2814 = "mix.prim.matmul"(%2813, %2812) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2815 = "mix.prim.reshape"(%2814) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2816 = "mix.prim.reshape"(%2815) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2817 = "mix.prim.transpose"(%2808) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2818 = "mix.prim.reshape"(%2811) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2819 = "mix.prim.matmul"(%2818, %2817) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2820 = "mix.prim.reshape"(%2819) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2821 = "mix.prim.reshape"(%2820) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2822 = "mix.prim.slice"(%2821) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2823 = "mix.prim.slice"(%2821) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2824 = "mix.prim.reshape"(%2816) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2825 = "mix.prim.reshape"(%2822) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2826 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2827 = "mix.prim.convert"(%2826) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2828 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2829 = "mix.prim.div"(%2827, %2828) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2830 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2831 = "mix.prim.pow"(%2830, %2829) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2832 = "mix.prim.reciprocal"(%2831) : (tensor<80xf16>) -> tensor<80xf16>
    %2833 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2834 = "mix.prim.mul"(%2833, %2832) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2835 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2836 = "mix.prim.unsqueeze"(%2835) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2837 = "mix.prim.permute"(%2836) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2838 = "mix.prim.unsqueeze"(%2834) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2839 = "mix.prim.permute"(%2838) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2840 = "mix.prim.mul"(%2837, %2839) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2841 = "mix.prim.concat"(%2840, %2840) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2842 = "mix.prim.cos"(%2841) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2843 = "mix.prim.slice"(%2842) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2844 = "mix.prim.unsqueeze"(%2843) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2845 = "mix.prim.slice"(%2844) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2846 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2847 = "mix.prim.mul"(%2845, %2846) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2848 = "mix.prim.sin"(%2841) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2849 = "mix.prim.slice"(%2848) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2850 = "mix.prim.unsqueeze"(%2849) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2851 = "mix.prim.slice"(%2850) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2852 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2853 = "mix.prim.mul"(%2851, %2852) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2854 = "mix.prim.slice"(%2847) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2855 = "mix.prim.slice"(%2853) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2856 = "mix.prim.mul"(%2824, %2854) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2857 = "mix.prim.slice"(%2824) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2858 = "mix.prim.slice"(%2824) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2859 = "mix.prim.neg"(%2858) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2860 = "mix.prim.concat"(%2859, %2857) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2861 = "mix.prim.mul"(%2860, %2855) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2862 = "mix.prim.add"(%2856, %2861) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2863 = "mix.prim.mul"(%2825, %2854) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2864 = "mix.prim.slice"(%2825) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2865 = "mix.prim.slice"(%2825) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2866 = "mix.prim.neg"(%2865) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2867 = "mix.prim.concat"(%2866, %2864) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2868 = "mix.prim.mul"(%2867, %2855) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2869 = "mix.prim.add"(%2863, %2868) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2870 = "mix.prim.reshape"(%2862) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2871 = "mix.prim.reshape"(%2869) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2872 = "mix.prim.reshape"(%2870) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2873 = "mix.prim.reshape"(%2871) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2874 = "mix.prim.transpose"(%2872) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2875 = "mix.prim.transpose"(%2873) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2876 = "mix.prim.transpose"(%2875) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2877 = "mix.prim.unsqueeze"(%2874) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2878 = "mix.prim.permute"(%2877) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2879 = "mix.prim.unsqueeze"(%2876) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2880 = "mix.prim.permute"(%2879) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2881 = "mix.prim.permute"(%2878) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2882 = "mix.prim.reshape"(%2881) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2883 = "mix.prim.permute"(%2880) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2884 = "mix.prim.reshape"(%2883) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2885 = "mix.prim.batch_matmul"(%2882, %2884) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2886 = "mix.prim.reshape"(%2885) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2887 = "mix.prim.permute"(%2886) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2888 = "mix.prim.reshape"(%2887) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2889 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2890 = "mix.prim.mul"(%2888, %2889) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2891 = "mix.prim.reshape"(%2890) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2892 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2893 = "mix.comp.masked_fill"(%2891, %4, %2892) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2894 = "mix.comp.softmax"(%2893) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2895 = "mix.prim.reshape"(%2894) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2896 = "mix.prim.reshape"(%2823) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2897 = "mix.prim.transpose"(%2896) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2898 = "mix.prim.batch_matmul"(%2895, %2897) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2899 = "mix.prim.reshape"(%2898) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2900 = "mix.prim.permute"(%2899) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2901 = "mix.prim.reshape"(%2900) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2902 = "mix.prim.reshape"(%2901) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2903 = "mix.prim.transpose"(%2809) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2904 = "mix.prim.matmul"(%2902, %2903) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2905 = "mix.prim.add"(%2904, %2810) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2906 = "mix.prim.reshape"(%2905) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2907 = "mix.prim.mul"(%2797, %2906) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2908 = "mix.comp.weight"() <{param_loc = "transformer.h.21.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2909 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2910 = "mix.prim.pow"(%2907, %2909) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2911 = "mix.comp.mean"(%2910) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2912 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2913 = "mix.prim.add"(%2911, %2912) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2914 = "mix.prim.rsqrt"(%2913) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2915 = "mix.prim.mul"(%2907, %2914) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2916 = "mix.prim.mul"(%2908, %2915) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2917 = "mix.comp.weight"() <{param_loc = "transformer.h.21.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2918 = "mix.prim.reshape"(%2917) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2919 = "mix.prim.batch_matmul"(%2916, %2918) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2920 = "mix.comp.silu"(%2919) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2921 = "mix.comp.weight"() <{param_loc = "transformer.h.21.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %2922 = "mix.prim.reshape"(%2921) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %2923 = "mix.prim.batch_matmul"(%2916, %2922) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %2924 = "mix.prim.mul"(%2920, %2923) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2925 = "mix.comp.weight"() <{param_loc = "transformer.h.21.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %2926 = "mix.prim.reshape"(%2925) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %2927 = "mix.prim.batch_matmul"(%2924, %2926) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %2928 = "mix.comp.weight"() <{param_loc = "transformer.h.21.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %2929 = "mix.prim.add"(%2927, %2928) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %2930 = "mix.prim.add"(%2929, %2907) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2931 = "mix.comp.weight"() <{param_loc = "transformer.h.22.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2932 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2933 = "mix.prim.pow"(%2930, %2932) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2934 = "mix.comp.mean"(%2933) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2935 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2936 = "mix.prim.add"(%2934, %2935) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2937 = "mix.prim.rsqrt"(%2936) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2938 = "mix.prim.mul"(%2930, %2937) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2939 = "mix.prim.mul"(%2931, %2938) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2940 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2941 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2942 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2943 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2944 = "mix.prim.transpose"(%2939) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2945 = "mix.prim.transpose"(%2940) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2946 = "mix.prim.reshape"(%2944) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2947 = "mix.prim.matmul"(%2946, %2945) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2948 = "mix.prim.reshape"(%2947) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2949 = "mix.prim.reshape"(%2948) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2950 = "mix.prim.transpose"(%2941) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2951 = "mix.prim.reshape"(%2944) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2952 = "mix.prim.matmul"(%2951, %2950) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2953 = "mix.prim.reshape"(%2952) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2954 = "mix.prim.reshape"(%2953) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2955 = "mix.prim.slice"(%2954) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2956 = "mix.prim.slice"(%2954) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2957 = "mix.prim.reshape"(%2949) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2958 = "mix.prim.reshape"(%2955) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2959 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2960 = "mix.prim.convert"(%2959) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2961 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2962 = "mix.prim.div"(%2960, %2961) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2963 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2964 = "mix.prim.pow"(%2963, %2962) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2965 = "mix.prim.reciprocal"(%2964) : (tensor<80xf16>) -> tensor<80xf16>
    %2966 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2967 = "mix.prim.mul"(%2966, %2965) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2968 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2969 = "mix.prim.unsqueeze"(%2968) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2970 = "mix.prim.permute"(%2969) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2971 = "mix.prim.unsqueeze"(%2967) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2972 = "mix.prim.permute"(%2971) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2973 = "mix.prim.mul"(%2970, %2972) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2974 = "mix.prim.concat"(%2973, %2973) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2975 = "mix.prim.cos"(%2974) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2976 = "mix.prim.slice"(%2975) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2977 = "mix.prim.unsqueeze"(%2976) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2978 = "mix.prim.slice"(%2977) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2979 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2980 = "mix.prim.mul"(%2978, %2979) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2981 = "mix.prim.sin"(%2974) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2982 = "mix.prim.slice"(%2981) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2983 = "mix.prim.unsqueeze"(%2982) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2984 = "mix.prim.slice"(%2983) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2985 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2986 = "mix.prim.mul"(%2984, %2985) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2987 = "mix.prim.slice"(%2980) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2988 = "mix.prim.slice"(%2986) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2989 = "mix.prim.mul"(%2957, %2987) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2990 = "mix.prim.slice"(%2957) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2991 = "mix.prim.slice"(%2957) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2992 = "mix.prim.neg"(%2991) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2993 = "mix.prim.concat"(%2992, %2990) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2994 = "mix.prim.mul"(%2993, %2988) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2995 = "mix.prim.add"(%2989, %2994) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2996 = "mix.prim.mul"(%2958, %2987) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2997 = "mix.prim.slice"(%2958) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2998 = "mix.prim.slice"(%2958) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2999 = "mix.prim.neg"(%2998) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3000 = "mix.prim.concat"(%2999, %2997) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3001 = "mix.prim.mul"(%3000, %2988) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3002 = "mix.prim.add"(%2996, %3001) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3003 = "mix.prim.reshape"(%2995) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3004 = "mix.prim.reshape"(%3002) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3005 = "mix.prim.reshape"(%3003) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3006 = "mix.prim.reshape"(%3004) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3007 = "mix.prim.transpose"(%3005) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3008 = "mix.prim.transpose"(%3006) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3009 = "mix.prim.transpose"(%3008) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3010 = "mix.prim.unsqueeze"(%3007) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3011 = "mix.prim.permute"(%3010) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3012 = "mix.prim.unsqueeze"(%3009) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3013 = "mix.prim.permute"(%3012) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3014 = "mix.prim.permute"(%3011) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3015 = "mix.prim.reshape"(%3014) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3016 = "mix.prim.permute"(%3013) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3017 = "mix.prim.reshape"(%3016) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3018 = "mix.prim.batch_matmul"(%3015, %3017) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3019 = "mix.prim.reshape"(%3018) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3020 = "mix.prim.permute"(%3019) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3021 = "mix.prim.reshape"(%3020) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3022 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3023 = "mix.prim.mul"(%3021, %3022) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3024 = "mix.prim.reshape"(%3023) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3025 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3026 = "mix.comp.masked_fill"(%3024, %4, %3025) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3027 = "mix.comp.softmax"(%3026) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3028 = "mix.prim.reshape"(%3027) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3029 = "mix.prim.reshape"(%2956) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3030 = "mix.prim.transpose"(%3029) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3031 = "mix.prim.batch_matmul"(%3028, %3030) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3032 = "mix.prim.reshape"(%3031) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3033 = "mix.prim.permute"(%3032) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3034 = "mix.prim.reshape"(%3033) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3035 = "mix.prim.reshape"(%3034) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3036 = "mix.prim.transpose"(%2942) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3037 = "mix.prim.matmul"(%3035, %3036) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3038 = "mix.prim.add"(%3037, %2943) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3039 = "mix.prim.reshape"(%3038) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3040 = "mix.prim.mul"(%2930, %3039) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3041 = "mix.comp.weight"() <{param_loc = "transformer.h.22.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3042 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3043 = "mix.prim.pow"(%3040, %3042) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3044 = "mix.comp.mean"(%3043) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3045 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3046 = "mix.prim.add"(%3044, %3045) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3047 = "mix.prim.rsqrt"(%3046) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3048 = "mix.prim.mul"(%3040, %3047) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3049 = "mix.prim.mul"(%3041, %3048) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3050 = "mix.comp.weight"() <{param_loc = "transformer.h.22.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3051 = "mix.prim.reshape"(%3050) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3052 = "mix.prim.batch_matmul"(%3049, %3051) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3053 = "mix.comp.silu"(%3052) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3054 = "mix.comp.weight"() <{param_loc = "transformer.h.22.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3055 = "mix.prim.reshape"(%3054) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3056 = "mix.prim.batch_matmul"(%3049, %3055) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3057 = "mix.prim.mul"(%3053, %3056) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3058 = "mix.comp.weight"() <{param_loc = "transformer.h.22.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %3059 = "mix.prim.reshape"(%3058) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %3060 = "mix.prim.batch_matmul"(%3057, %3059) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %3061 = "mix.comp.weight"() <{param_loc = "transformer.h.22.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %3062 = "mix.prim.add"(%3060, %3061) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %3063 = "mix.prim.add"(%3062, %3040) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3064 = "mix.comp.weight"() <{param_loc = "transformer.h.23.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3065 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3066 = "mix.prim.pow"(%3063, %3065) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3067 = "mix.comp.mean"(%3066) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3068 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3069 = "mix.prim.add"(%3067, %3068) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3070 = "mix.prim.rsqrt"(%3069) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3071 = "mix.prim.mul"(%3063, %3070) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3072 = "mix.prim.mul"(%3064, %3071) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3073 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3074 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3075 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3076 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3077 = "mix.prim.transpose"(%3072) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3078 = "mix.prim.transpose"(%3073) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3079 = "mix.prim.reshape"(%3077) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3080 = "mix.prim.matmul"(%3079, %3078) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3081 = "mix.prim.reshape"(%3080) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3082 = "mix.prim.reshape"(%3081) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3083 = "mix.prim.transpose"(%3074) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3084 = "mix.prim.reshape"(%3077) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3085 = "mix.prim.matmul"(%3084, %3083) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3086 = "mix.prim.reshape"(%3085) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3087 = "mix.prim.reshape"(%3086) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3088 = "mix.prim.slice"(%3087) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3089 = "mix.prim.slice"(%3087) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3090 = "mix.prim.reshape"(%3082) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3091 = "mix.prim.reshape"(%3088) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3092 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3093 = "mix.prim.convert"(%3092) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3094 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3095 = "mix.prim.div"(%3093, %3094) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3096 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3097 = "mix.prim.pow"(%3096, %3095) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3098 = "mix.prim.reciprocal"(%3097) : (tensor<80xf16>) -> tensor<80xf16>
    %3099 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3100 = "mix.prim.mul"(%3099, %3098) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3101 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3102 = "mix.prim.unsqueeze"(%3101) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3103 = "mix.prim.permute"(%3102) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3104 = "mix.prim.unsqueeze"(%3100) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3105 = "mix.prim.permute"(%3104) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3106 = "mix.prim.mul"(%3103, %3105) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3107 = "mix.prim.concat"(%3106, %3106) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3108 = "mix.prim.cos"(%3107) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3109 = "mix.prim.slice"(%3108) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3110 = "mix.prim.unsqueeze"(%3109) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3111 = "mix.prim.slice"(%3110) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3112 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3113 = "mix.prim.mul"(%3111, %3112) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3114 = "mix.prim.sin"(%3107) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3115 = "mix.prim.slice"(%3114) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3116 = "mix.prim.unsqueeze"(%3115) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3117 = "mix.prim.slice"(%3116) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3118 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3119 = "mix.prim.mul"(%3117, %3118) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3120 = "mix.prim.slice"(%3113) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3121 = "mix.prim.slice"(%3119) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3122 = "mix.prim.mul"(%3090, %3120) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3123 = "mix.prim.slice"(%3090) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3124 = "mix.prim.slice"(%3090) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3125 = "mix.prim.neg"(%3124) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3126 = "mix.prim.concat"(%3125, %3123) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3127 = "mix.prim.mul"(%3126, %3121) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3128 = "mix.prim.add"(%3122, %3127) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3129 = "mix.prim.mul"(%3091, %3120) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3130 = "mix.prim.slice"(%3091) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3131 = "mix.prim.slice"(%3091) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3132 = "mix.prim.neg"(%3131) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3133 = "mix.prim.concat"(%3132, %3130) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3134 = "mix.prim.mul"(%3133, %3121) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3135 = "mix.prim.add"(%3129, %3134) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3136 = "mix.prim.reshape"(%3128) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3137 = "mix.prim.reshape"(%3135) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3138 = "mix.prim.reshape"(%3136) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3139 = "mix.prim.reshape"(%3137) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3140 = "mix.prim.transpose"(%3138) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3141 = "mix.prim.transpose"(%3139) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3142 = "mix.prim.transpose"(%3141) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3143 = "mix.prim.unsqueeze"(%3140) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3144 = "mix.prim.permute"(%3143) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3145 = "mix.prim.unsqueeze"(%3142) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3146 = "mix.prim.permute"(%3145) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3147 = "mix.prim.permute"(%3144) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3148 = "mix.prim.reshape"(%3147) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3149 = "mix.prim.permute"(%3146) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3150 = "mix.prim.reshape"(%3149) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3151 = "mix.prim.batch_matmul"(%3148, %3150) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3152 = "mix.prim.reshape"(%3151) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3153 = "mix.prim.permute"(%3152) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3154 = "mix.prim.reshape"(%3153) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3155 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3156 = "mix.prim.mul"(%3154, %3155) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3157 = "mix.prim.reshape"(%3156) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3158 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3159 = "mix.comp.masked_fill"(%3157, %4, %3158) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3160 = "mix.comp.softmax"(%3159) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3161 = "mix.prim.reshape"(%3160) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3162 = "mix.prim.reshape"(%3089) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3163 = "mix.prim.transpose"(%3162) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3164 = "mix.prim.batch_matmul"(%3161, %3163) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3165 = "mix.prim.reshape"(%3164) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3166 = "mix.prim.permute"(%3165) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3167 = "mix.prim.reshape"(%3166) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3168 = "mix.prim.reshape"(%3167) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3169 = "mix.prim.transpose"(%3075) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3170 = "mix.prim.matmul"(%3168, %3169) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3171 = "mix.prim.add"(%3170, %3076) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3172 = "mix.prim.reshape"(%3171) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3173 = "mix.prim.mul"(%3063, %3172) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3174 = "mix.comp.weight"() <{param_loc = "transformer.h.23.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3175 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3176 = "mix.prim.pow"(%3173, %3175) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3177 = "mix.comp.mean"(%3176) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3178 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3179 = "mix.prim.add"(%3177, %3178) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3180 = "mix.prim.rsqrt"(%3179) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3181 = "mix.prim.mul"(%3173, %3180) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3182 = "mix.prim.mul"(%3174, %3181) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3183 = "mix.comp.weight"() <{param_loc = "transformer.h.23.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3184 = "mix.prim.reshape"(%3183) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3185 = "mix.prim.batch_matmul"(%3182, %3184) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3186 = "mix.comp.silu"(%3185) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3187 = "mix.comp.weight"() <{param_loc = "transformer.h.23.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3188 = "mix.prim.reshape"(%3187) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3189 = "mix.prim.batch_matmul"(%3182, %3188) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3190 = "mix.prim.mul"(%3186, %3189) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3191 = "mix.comp.weight"() <{param_loc = "transformer.h.23.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %3192 = "mix.prim.reshape"(%3191) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %3193 = "mix.prim.batch_matmul"(%3190, %3192) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %3194 = "mix.comp.weight"() <{param_loc = "transformer.h.23.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %3195 = "mix.prim.add"(%3193, %3194) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %3196 = "mix.prim.add"(%3195, %3173) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3197 = "mix.comp.weight"() <{param_loc = "transformer.h.24.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3198 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3199 = "mix.prim.pow"(%3196, %3198) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3200 = "mix.comp.mean"(%3199) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3201 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3202 = "mix.prim.add"(%3200, %3201) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3203 = "mix.prim.rsqrt"(%3202) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3204 = "mix.prim.mul"(%3196, %3203) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3205 = "mix.prim.mul"(%3197, %3204) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3206 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3207 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3208 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3209 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3210 = "mix.prim.transpose"(%3205) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3211 = "mix.prim.transpose"(%3206) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3212 = "mix.prim.reshape"(%3210) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3213 = "mix.prim.matmul"(%3212, %3211) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3214 = "mix.prim.reshape"(%3213) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3215 = "mix.prim.reshape"(%3214) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3216 = "mix.prim.transpose"(%3207) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3217 = "mix.prim.reshape"(%3210) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3218 = "mix.prim.matmul"(%3217, %3216) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3219 = "mix.prim.reshape"(%3218) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3220 = "mix.prim.reshape"(%3219) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3221 = "mix.prim.slice"(%3220) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3222 = "mix.prim.slice"(%3220) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3223 = "mix.prim.reshape"(%3215) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3224 = "mix.prim.reshape"(%3221) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3225 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3226 = "mix.prim.convert"(%3225) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3227 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3228 = "mix.prim.div"(%3226, %3227) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3229 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3230 = "mix.prim.pow"(%3229, %3228) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3231 = "mix.prim.reciprocal"(%3230) : (tensor<80xf16>) -> tensor<80xf16>
    %3232 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3233 = "mix.prim.mul"(%3232, %3231) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3234 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3235 = "mix.prim.unsqueeze"(%3234) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3236 = "mix.prim.permute"(%3235) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3237 = "mix.prim.unsqueeze"(%3233) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3238 = "mix.prim.permute"(%3237) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3239 = "mix.prim.mul"(%3236, %3238) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3240 = "mix.prim.concat"(%3239, %3239) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3241 = "mix.prim.cos"(%3240) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3242 = "mix.prim.slice"(%3241) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3243 = "mix.prim.unsqueeze"(%3242) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3244 = "mix.prim.slice"(%3243) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3245 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3246 = "mix.prim.mul"(%3244, %3245) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3247 = "mix.prim.sin"(%3240) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3248 = "mix.prim.slice"(%3247) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3249 = "mix.prim.unsqueeze"(%3248) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3250 = "mix.prim.slice"(%3249) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3251 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3252 = "mix.prim.mul"(%3250, %3251) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3253 = "mix.prim.slice"(%3246) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3254 = "mix.prim.slice"(%3252) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3255 = "mix.prim.mul"(%3223, %3253) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3256 = "mix.prim.slice"(%3223) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3257 = "mix.prim.slice"(%3223) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3258 = "mix.prim.neg"(%3257) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3259 = "mix.prim.concat"(%3258, %3256) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3260 = "mix.prim.mul"(%3259, %3254) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3261 = "mix.prim.add"(%3255, %3260) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3262 = "mix.prim.mul"(%3224, %3253) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3263 = "mix.prim.slice"(%3224) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3264 = "mix.prim.slice"(%3224) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3265 = "mix.prim.neg"(%3264) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3266 = "mix.prim.concat"(%3265, %3263) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3267 = "mix.prim.mul"(%3266, %3254) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3268 = "mix.prim.add"(%3262, %3267) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3269 = "mix.prim.reshape"(%3261) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3270 = "mix.prim.reshape"(%3268) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3271 = "mix.prim.reshape"(%3269) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3272 = "mix.prim.reshape"(%3270) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3273 = "mix.prim.transpose"(%3271) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3274 = "mix.prim.transpose"(%3272) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3275 = "mix.prim.transpose"(%3274) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3276 = "mix.prim.unsqueeze"(%3273) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3277 = "mix.prim.permute"(%3276) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3278 = "mix.prim.unsqueeze"(%3275) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3279 = "mix.prim.permute"(%3278) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3280 = "mix.prim.permute"(%3277) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3281 = "mix.prim.reshape"(%3280) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3282 = "mix.prim.permute"(%3279) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3283 = "mix.prim.reshape"(%3282) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3284 = "mix.prim.batch_matmul"(%3281, %3283) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3285 = "mix.prim.reshape"(%3284) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3286 = "mix.prim.permute"(%3285) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3287 = "mix.prim.reshape"(%3286) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3288 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3289 = "mix.prim.mul"(%3287, %3288) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3290 = "mix.prim.reshape"(%3289) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3291 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3292 = "mix.comp.masked_fill"(%3290, %4, %3291) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3293 = "mix.comp.softmax"(%3292) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3294 = "mix.prim.reshape"(%3293) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3295 = "mix.prim.reshape"(%3222) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3296 = "mix.prim.transpose"(%3295) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3297 = "mix.prim.batch_matmul"(%3294, %3296) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3298 = "mix.prim.reshape"(%3297) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3299 = "mix.prim.permute"(%3298) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3300 = "mix.prim.reshape"(%3299) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3301 = "mix.prim.reshape"(%3300) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3302 = "mix.prim.transpose"(%3208) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3303 = "mix.prim.matmul"(%3301, %3302) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3304 = "mix.prim.add"(%3303, %3209) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3305 = "mix.prim.reshape"(%3304) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3306 = "mix.prim.mul"(%3196, %3305) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3307 = "mix.comp.weight"() <{param_loc = "transformer.h.24.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3308 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3309 = "mix.prim.pow"(%3306, %3308) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3310 = "mix.comp.mean"(%3309) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3311 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3312 = "mix.prim.add"(%3310, %3311) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3313 = "mix.prim.rsqrt"(%3312) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3314 = "mix.prim.mul"(%3306, %3313) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3315 = "mix.prim.mul"(%3307, %3314) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3316 = "mix.comp.weight"() <{param_loc = "transformer.h.24.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3317 = "mix.prim.reshape"(%3316) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3318 = "mix.prim.batch_matmul"(%3315, %3317) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3319 = "mix.comp.silu"(%3318) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3320 = "mix.comp.weight"() <{param_loc = "transformer.h.24.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3321 = "mix.prim.reshape"(%3320) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3322 = "mix.prim.batch_matmul"(%3315, %3321) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3323 = "mix.prim.mul"(%3319, %3322) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3324 = "mix.comp.weight"() <{param_loc = "transformer.h.24.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %3325 = "mix.prim.reshape"(%3324) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %3326 = "mix.prim.batch_matmul"(%3323, %3325) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %3327 = "mix.comp.weight"() <{param_loc = "transformer.h.24.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %3328 = "mix.prim.add"(%3326, %3327) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %3329 = "mix.prim.add"(%3328, %3306) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3330 = "mix.comp.weight"() <{param_loc = "transformer.h.25.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3331 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3332 = "mix.prim.pow"(%3329, %3331) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3333 = "mix.comp.mean"(%3332) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3334 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3335 = "mix.prim.add"(%3333, %3334) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3336 = "mix.prim.rsqrt"(%3335) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3337 = "mix.prim.mul"(%3329, %3336) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3338 = "mix.prim.mul"(%3330, %3337) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3339 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3340 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3341 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3342 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3343 = "mix.prim.transpose"(%3338) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3344 = "mix.prim.transpose"(%3339) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3345 = "mix.prim.reshape"(%3343) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3346 = "mix.prim.matmul"(%3345, %3344) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3347 = "mix.prim.reshape"(%3346) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3348 = "mix.prim.reshape"(%3347) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3349 = "mix.prim.transpose"(%3340) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3350 = "mix.prim.reshape"(%3343) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3351 = "mix.prim.matmul"(%3350, %3349) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3352 = "mix.prim.reshape"(%3351) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3353 = "mix.prim.reshape"(%3352) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3354 = "mix.prim.slice"(%3353) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3355 = "mix.prim.slice"(%3353) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3356 = "mix.prim.reshape"(%3348) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3357 = "mix.prim.reshape"(%3354) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3358 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3359 = "mix.prim.convert"(%3358) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3360 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3361 = "mix.prim.div"(%3359, %3360) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3362 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3363 = "mix.prim.pow"(%3362, %3361) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3364 = "mix.prim.reciprocal"(%3363) : (tensor<80xf16>) -> tensor<80xf16>
    %3365 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3366 = "mix.prim.mul"(%3365, %3364) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3367 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3368 = "mix.prim.unsqueeze"(%3367) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3369 = "mix.prim.permute"(%3368) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3370 = "mix.prim.unsqueeze"(%3366) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3371 = "mix.prim.permute"(%3370) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3372 = "mix.prim.mul"(%3369, %3371) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3373 = "mix.prim.concat"(%3372, %3372) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3374 = "mix.prim.cos"(%3373) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3375 = "mix.prim.slice"(%3374) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3376 = "mix.prim.unsqueeze"(%3375) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3377 = "mix.prim.slice"(%3376) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3378 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3379 = "mix.prim.mul"(%3377, %3378) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3380 = "mix.prim.sin"(%3373) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3381 = "mix.prim.slice"(%3380) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3382 = "mix.prim.unsqueeze"(%3381) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3383 = "mix.prim.slice"(%3382) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3384 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3385 = "mix.prim.mul"(%3383, %3384) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3386 = "mix.prim.slice"(%3379) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3387 = "mix.prim.slice"(%3385) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3388 = "mix.prim.mul"(%3356, %3386) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3389 = "mix.prim.slice"(%3356) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3390 = "mix.prim.slice"(%3356) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3391 = "mix.prim.neg"(%3390) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3392 = "mix.prim.concat"(%3391, %3389) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3393 = "mix.prim.mul"(%3392, %3387) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3394 = "mix.prim.add"(%3388, %3393) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3395 = "mix.prim.mul"(%3357, %3386) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3396 = "mix.prim.slice"(%3357) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3397 = "mix.prim.slice"(%3357) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3398 = "mix.prim.neg"(%3397) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3399 = "mix.prim.concat"(%3398, %3396) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3400 = "mix.prim.mul"(%3399, %3387) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3401 = "mix.prim.add"(%3395, %3400) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3402 = "mix.prim.reshape"(%3394) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3403 = "mix.prim.reshape"(%3401) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3404 = "mix.prim.reshape"(%3402) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3405 = "mix.prim.reshape"(%3403) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3406 = "mix.prim.transpose"(%3404) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3407 = "mix.prim.transpose"(%3405) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3408 = "mix.prim.transpose"(%3407) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3409 = "mix.prim.unsqueeze"(%3406) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3410 = "mix.prim.permute"(%3409) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3411 = "mix.prim.unsqueeze"(%3408) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3412 = "mix.prim.permute"(%3411) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3413 = "mix.prim.permute"(%3410) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3414 = "mix.prim.reshape"(%3413) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3415 = "mix.prim.permute"(%3412) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3416 = "mix.prim.reshape"(%3415) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3417 = "mix.prim.batch_matmul"(%3414, %3416) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3418 = "mix.prim.reshape"(%3417) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3419 = "mix.prim.permute"(%3418) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3420 = "mix.prim.reshape"(%3419) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3421 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3422 = "mix.prim.mul"(%3420, %3421) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3423 = "mix.prim.reshape"(%3422) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3424 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3425 = "mix.comp.masked_fill"(%3423, %4, %3424) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3426 = "mix.comp.softmax"(%3425) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3427 = "mix.prim.reshape"(%3426) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3428 = "mix.prim.reshape"(%3355) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3429 = "mix.prim.transpose"(%3428) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3430 = "mix.prim.batch_matmul"(%3427, %3429) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3431 = "mix.prim.reshape"(%3430) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3432 = "mix.prim.permute"(%3431) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3433 = "mix.prim.reshape"(%3432) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3434 = "mix.prim.reshape"(%3433) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3435 = "mix.prim.transpose"(%3341) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3436 = "mix.prim.matmul"(%3434, %3435) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3437 = "mix.prim.add"(%3436, %3342) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3438 = "mix.prim.reshape"(%3437) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3439 = "mix.prim.mul"(%3329, %3438) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3440 = "mix.comp.weight"() <{param_loc = "transformer.h.25.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3441 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3442 = "mix.prim.pow"(%3439, %3441) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3443 = "mix.comp.mean"(%3442) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3444 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3445 = "mix.prim.add"(%3443, %3444) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3446 = "mix.prim.rsqrt"(%3445) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3447 = "mix.prim.mul"(%3439, %3446) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3448 = "mix.prim.mul"(%3440, %3447) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3449 = "mix.comp.weight"() <{param_loc = "transformer.h.25.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3450 = "mix.prim.reshape"(%3449) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3451 = "mix.prim.batch_matmul"(%3448, %3450) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3452 = "mix.comp.silu"(%3451) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3453 = "mix.comp.weight"() <{param_loc = "transformer.h.25.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3454 = "mix.prim.reshape"(%3453) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3455 = "mix.prim.batch_matmul"(%3448, %3454) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3456 = "mix.prim.mul"(%3452, %3455) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3457 = "mix.comp.weight"() <{param_loc = "transformer.h.25.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %3458 = "mix.prim.reshape"(%3457) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %3459 = "mix.prim.batch_matmul"(%3456, %3458) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %3460 = "mix.comp.weight"() <{param_loc = "transformer.h.25.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %3461 = "mix.prim.add"(%3459, %3460) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %3462 = "mix.prim.add"(%3461, %3439) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3463 = "mix.comp.weight"() <{param_loc = "transformer.h.26.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3464 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3465 = "mix.prim.pow"(%3462, %3464) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3466 = "mix.comp.mean"(%3465) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3467 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3468 = "mix.prim.add"(%3466, %3467) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3469 = "mix.prim.rsqrt"(%3468) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3470 = "mix.prim.mul"(%3462, %3469) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3471 = "mix.prim.mul"(%3463, %3470) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3472 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3473 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3474 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3475 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3476 = "mix.prim.transpose"(%3471) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3477 = "mix.prim.transpose"(%3472) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3478 = "mix.prim.reshape"(%3476) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3479 = "mix.prim.matmul"(%3478, %3477) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3480 = "mix.prim.reshape"(%3479) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3481 = "mix.prim.reshape"(%3480) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3482 = "mix.prim.transpose"(%3473) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3483 = "mix.prim.reshape"(%3476) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3484 = "mix.prim.matmul"(%3483, %3482) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3485 = "mix.prim.reshape"(%3484) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3486 = "mix.prim.reshape"(%3485) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3487 = "mix.prim.slice"(%3486) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3488 = "mix.prim.slice"(%3486) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3489 = "mix.prim.reshape"(%3481) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3490 = "mix.prim.reshape"(%3487) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3491 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3492 = "mix.prim.convert"(%3491) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3493 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3494 = "mix.prim.div"(%3492, %3493) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3495 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3496 = "mix.prim.pow"(%3495, %3494) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3497 = "mix.prim.reciprocal"(%3496) : (tensor<80xf16>) -> tensor<80xf16>
    %3498 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3499 = "mix.prim.mul"(%3498, %3497) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3500 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3501 = "mix.prim.unsqueeze"(%3500) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3502 = "mix.prim.permute"(%3501) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3503 = "mix.prim.unsqueeze"(%3499) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3504 = "mix.prim.permute"(%3503) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3505 = "mix.prim.mul"(%3502, %3504) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3506 = "mix.prim.concat"(%3505, %3505) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3507 = "mix.prim.cos"(%3506) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3508 = "mix.prim.slice"(%3507) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3509 = "mix.prim.unsqueeze"(%3508) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3510 = "mix.prim.slice"(%3509) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3511 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3512 = "mix.prim.mul"(%3510, %3511) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3513 = "mix.prim.sin"(%3506) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3514 = "mix.prim.slice"(%3513) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3515 = "mix.prim.unsqueeze"(%3514) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3516 = "mix.prim.slice"(%3515) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3517 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3518 = "mix.prim.mul"(%3516, %3517) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3519 = "mix.prim.slice"(%3512) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3520 = "mix.prim.slice"(%3518) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3521 = "mix.prim.mul"(%3489, %3519) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3522 = "mix.prim.slice"(%3489) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3523 = "mix.prim.slice"(%3489) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3524 = "mix.prim.neg"(%3523) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3525 = "mix.prim.concat"(%3524, %3522) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3526 = "mix.prim.mul"(%3525, %3520) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3527 = "mix.prim.add"(%3521, %3526) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3528 = "mix.prim.mul"(%3490, %3519) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3529 = "mix.prim.slice"(%3490) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3530 = "mix.prim.slice"(%3490) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3531 = "mix.prim.neg"(%3530) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3532 = "mix.prim.concat"(%3531, %3529) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3533 = "mix.prim.mul"(%3532, %3520) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3534 = "mix.prim.add"(%3528, %3533) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3535 = "mix.prim.reshape"(%3527) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3536 = "mix.prim.reshape"(%3534) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3537 = "mix.prim.reshape"(%3535) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3538 = "mix.prim.reshape"(%3536) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3539 = "mix.prim.transpose"(%3537) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3540 = "mix.prim.transpose"(%3538) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3541 = "mix.prim.transpose"(%3540) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3542 = "mix.prim.unsqueeze"(%3539) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3543 = "mix.prim.permute"(%3542) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3544 = "mix.prim.unsqueeze"(%3541) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3545 = "mix.prim.permute"(%3544) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3546 = "mix.prim.permute"(%3543) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3547 = "mix.prim.reshape"(%3546) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3548 = "mix.prim.permute"(%3545) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3549 = "mix.prim.reshape"(%3548) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3550 = "mix.prim.batch_matmul"(%3547, %3549) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3551 = "mix.prim.reshape"(%3550) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3552 = "mix.prim.permute"(%3551) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3553 = "mix.prim.reshape"(%3552) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3554 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3555 = "mix.prim.mul"(%3553, %3554) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3556 = "mix.prim.reshape"(%3555) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3557 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3558 = "mix.comp.masked_fill"(%3556, %4, %3557) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3559 = "mix.comp.softmax"(%3558) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3560 = "mix.prim.reshape"(%3559) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3561 = "mix.prim.reshape"(%3488) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3562 = "mix.prim.transpose"(%3561) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3563 = "mix.prim.batch_matmul"(%3560, %3562) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3564 = "mix.prim.reshape"(%3563) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3565 = "mix.prim.permute"(%3564) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3566 = "mix.prim.reshape"(%3565) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3567 = "mix.prim.reshape"(%3566) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3568 = "mix.prim.transpose"(%3474) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3569 = "mix.prim.matmul"(%3567, %3568) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3570 = "mix.prim.add"(%3569, %3475) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3571 = "mix.prim.reshape"(%3570) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3572 = "mix.prim.mul"(%3462, %3571) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3573 = "mix.comp.weight"() <{param_loc = "transformer.h.26.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3574 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3575 = "mix.prim.pow"(%3572, %3574) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3576 = "mix.comp.mean"(%3575) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3577 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3578 = "mix.prim.add"(%3576, %3577) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3579 = "mix.prim.rsqrt"(%3578) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3580 = "mix.prim.mul"(%3572, %3579) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3581 = "mix.prim.mul"(%3573, %3580) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3582 = "mix.comp.weight"() <{param_loc = "transformer.h.26.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3583 = "mix.prim.reshape"(%3582) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3584 = "mix.prim.batch_matmul"(%3581, %3583) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3585 = "mix.comp.silu"(%3584) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3586 = "mix.comp.weight"() <{param_loc = "transformer.h.26.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3587 = "mix.prim.reshape"(%3586) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3588 = "mix.prim.batch_matmul"(%3581, %3587) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3589 = "mix.prim.mul"(%3585, %3588) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3590 = "mix.comp.weight"() <{param_loc = "transformer.h.26.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %3591 = "mix.prim.reshape"(%3590) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %3592 = "mix.prim.batch_matmul"(%3589, %3591) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %3593 = "mix.comp.weight"() <{param_loc = "transformer.h.26.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %3594 = "mix.prim.add"(%3592, %3593) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %3595 = "mix.prim.add"(%3594, %3572) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3596 = "mix.comp.weight"() <{param_loc = "transformer.h.27.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3597 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3598 = "mix.prim.pow"(%3595, %3597) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3599 = "mix.comp.mean"(%3598) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3600 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3601 = "mix.prim.add"(%3599, %3600) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3602 = "mix.prim.rsqrt"(%3601) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3603 = "mix.prim.mul"(%3595, %3602) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3604 = "mix.prim.mul"(%3596, %3603) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3605 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3606 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3607 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3608 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3609 = "mix.prim.transpose"(%3604) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3610 = "mix.prim.transpose"(%3605) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3611 = "mix.prim.reshape"(%3609) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3612 = "mix.prim.matmul"(%3611, %3610) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3613 = "mix.prim.reshape"(%3612) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3614 = "mix.prim.reshape"(%3613) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3615 = "mix.prim.transpose"(%3606) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3616 = "mix.prim.reshape"(%3609) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3617 = "mix.prim.matmul"(%3616, %3615) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3618 = "mix.prim.reshape"(%3617) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3619 = "mix.prim.reshape"(%3618) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3620 = "mix.prim.slice"(%3619) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3621 = "mix.prim.slice"(%3619) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3622 = "mix.prim.reshape"(%3614) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3623 = "mix.prim.reshape"(%3620) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3624 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3625 = "mix.prim.convert"(%3624) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3626 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3627 = "mix.prim.div"(%3625, %3626) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3628 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3629 = "mix.prim.pow"(%3628, %3627) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3630 = "mix.prim.reciprocal"(%3629) : (tensor<80xf16>) -> tensor<80xf16>
    %3631 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3632 = "mix.prim.mul"(%3631, %3630) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3633 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3634 = "mix.prim.unsqueeze"(%3633) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3635 = "mix.prim.permute"(%3634) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3636 = "mix.prim.unsqueeze"(%3632) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3637 = "mix.prim.permute"(%3636) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3638 = "mix.prim.mul"(%3635, %3637) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3639 = "mix.prim.concat"(%3638, %3638) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3640 = "mix.prim.cos"(%3639) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3641 = "mix.prim.slice"(%3640) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3642 = "mix.prim.unsqueeze"(%3641) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3643 = "mix.prim.slice"(%3642) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3644 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3645 = "mix.prim.mul"(%3643, %3644) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3646 = "mix.prim.sin"(%3639) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3647 = "mix.prim.slice"(%3646) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3648 = "mix.prim.unsqueeze"(%3647) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3649 = "mix.prim.slice"(%3648) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3650 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3651 = "mix.prim.mul"(%3649, %3650) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3652 = "mix.prim.slice"(%3645) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3653 = "mix.prim.slice"(%3651) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3654 = "mix.prim.mul"(%3622, %3652) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3655 = "mix.prim.slice"(%3622) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3656 = "mix.prim.slice"(%3622) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3657 = "mix.prim.neg"(%3656) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3658 = "mix.prim.concat"(%3657, %3655) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3659 = "mix.prim.mul"(%3658, %3653) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3660 = "mix.prim.add"(%3654, %3659) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3661 = "mix.prim.mul"(%3623, %3652) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3662 = "mix.prim.slice"(%3623) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3663 = "mix.prim.slice"(%3623) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3664 = "mix.prim.neg"(%3663) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3665 = "mix.prim.concat"(%3664, %3662) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3666 = "mix.prim.mul"(%3665, %3653) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3667 = "mix.prim.add"(%3661, %3666) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3668 = "mix.prim.reshape"(%3660) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3669 = "mix.prim.reshape"(%3667) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3670 = "mix.prim.reshape"(%3668) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3671 = "mix.prim.reshape"(%3669) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3672 = "mix.prim.transpose"(%3670) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3673 = "mix.prim.transpose"(%3671) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3674 = "mix.prim.transpose"(%3673) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3675 = "mix.prim.unsqueeze"(%3672) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3676 = "mix.prim.permute"(%3675) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3677 = "mix.prim.unsqueeze"(%3674) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3678 = "mix.prim.permute"(%3677) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3679 = "mix.prim.permute"(%3676) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3680 = "mix.prim.reshape"(%3679) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3681 = "mix.prim.permute"(%3678) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3682 = "mix.prim.reshape"(%3681) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3683 = "mix.prim.batch_matmul"(%3680, %3682) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3684 = "mix.prim.reshape"(%3683) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3685 = "mix.prim.permute"(%3684) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3686 = "mix.prim.reshape"(%3685) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3687 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3688 = "mix.prim.mul"(%3686, %3687) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3689 = "mix.prim.reshape"(%3688) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3690 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3691 = "mix.comp.masked_fill"(%3689, %4, %3690) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3692 = "mix.comp.softmax"(%3691) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3693 = "mix.prim.reshape"(%3692) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3694 = "mix.prim.reshape"(%3621) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3695 = "mix.prim.transpose"(%3694) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3696 = "mix.prim.batch_matmul"(%3693, %3695) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3697 = "mix.prim.reshape"(%3696) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3698 = "mix.prim.permute"(%3697) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3699 = "mix.prim.reshape"(%3698) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3700 = "mix.prim.reshape"(%3699) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3701 = "mix.prim.transpose"(%3607) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3702 = "mix.prim.matmul"(%3700, %3701) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3703 = "mix.prim.add"(%3702, %3608) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3704 = "mix.prim.reshape"(%3703) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3705 = "mix.prim.mul"(%3595, %3704) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3706 = "mix.comp.weight"() <{param_loc = "transformer.h.27.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3707 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3708 = "mix.prim.pow"(%3705, %3707) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3709 = "mix.comp.mean"(%3708) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3710 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3711 = "mix.prim.add"(%3709, %3710) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3712 = "mix.prim.rsqrt"(%3711) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3713 = "mix.prim.mul"(%3705, %3712) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3714 = "mix.prim.mul"(%3706, %3713) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3715 = "mix.comp.weight"() <{param_loc = "transformer.h.27.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3716 = "mix.prim.reshape"(%3715) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3717 = "mix.prim.batch_matmul"(%3714, %3716) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3718 = "mix.comp.silu"(%3717) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3719 = "mix.comp.weight"() <{param_loc = "transformer.h.27.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3720 = "mix.prim.reshape"(%3719) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3721 = "mix.prim.batch_matmul"(%3714, %3720) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3722 = "mix.prim.mul"(%3718, %3721) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3723 = "mix.comp.weight"() <{param_loc = "transformer.h.27.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %3724 = "mix.prim.reshape"(%3723) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %3725 = "mix.prim.batch_matmul"(%3722, %3724) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %3726 = "mix.comp.weight"() <{param_loc = "transformer.h.27.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %3727 = "mix.prim.add"(%3725, %3726) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %3728 = "mix.prim.add"(%3727, %3705) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3729 = "mix.comp.weight"() <{param_loc = "transformer.h.28.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3730 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3731 = "mix.prim.pow"(%3728, %3730) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3732 = "mix.comp.mean"(%3731) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3733 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3734 = "mix.prim.add"(%3732, %3733) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3735 = "mix.prim.rsqrt"(%3734) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3736 = "mix.prim.mul"(%3728, %3735) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3737 = "mix.prim.mul"(%3729, %3736) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3738 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3739 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3740 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3741 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3742 = "mix.prim.transpose"(%3737) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3743 = "mix.prim.transpose"(%3738) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3744 = "mix.prim.reshape"(%3742) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3745 = "mix.prim.matmul"(%3744, %3743) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3746 = "mix.prim.reshape"(%3745) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3747 = "mix.prim.reshape"(%3746) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3748 = "mix.prim.transpose"(%3739) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3749 = "mix.prim.reshape"(%3742) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3750 = "mix.prim.matmul"(%3749, %3748) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3751 = "mix.prim.reshape"(%3750) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3752 = "mix.prim.reshape"(%3751) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3753 = "mix.prim.slice"(%3752) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3754 = "mix.prim.slice"(%3752) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3755 = "mix.prim.reshape"(%3747) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3756 = "mix.prim.reshape"(%3753) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3757 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3758 = "mix.prim.convert"(%3757) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3759 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3760 = "mix.prim.div"(%3758, %3759) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3761 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3762 = "mix.prim.pow"(%3761, %3760) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3763 = "mix.prim.reciprocal"(%3762) : (tensor<80xf16>) -> tensor<80xf16>
    %3764 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3765 = "mix.prim.mul"(%3764, %3763) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3766 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3767 = "mix.prim.unsqueeze"(%3766) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3768 = "mix.prim.permute"(%3767) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3769 = "mix.prim.unsqueeze"(%3765) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3770 = "mix.prim.permute"(%3769) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3771 = "mix.prim.mul"(%3768, %3770) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3772 = "mix.prim.concat"(%3771, %3771) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3773 = "mix.prim.cos"(%3772) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3774 = "mix.prim.slice"(%3773) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3775 = "mix.prim.unsqueeze"(%3774) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3776 = "mix.prim.slice"(%3775) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3777 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3778 = "mix.prim.mul"(%3776, %3777) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3779 = "mix.prim.sin"(%3772) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3780 = "mix.prim.slice"(%3779) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3781 = "mix.prim.unsqueeze"(%3780) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3782 = "mix.prim.slice"(%3781) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3783 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3784 = "mix.prim.mul"(%3782, %3783) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3785 = "mix.prim.slice"(%3778) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3786 = "mix.prim.slice"(%3784) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3787 = "mix.prim.mul"(%3755, %3785) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3788 = "mix.prim.slice"(%3755) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3789 = "mix.prim.slice"(%3755) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3790 = "mix.prim.neg"(%3789) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3791 = "mix.prim.concat"(%3790, %3788) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3792 = "mix.prim.mul"(%3791, %3786) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3793 = "mix.prim.add"(%3787, %3792) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3794 = "mix.prim.mul"(%3756, %3785) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3795 = "mix.prim.slice"(%3756) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3796 = "mix.prim.slice"(%3756) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3797 = "mix.prim.neg"(%3796) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3798 = "mix.prim.concat"(%3797, %3795) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3799 = "mix.prim.mul"(%3798, %3786) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3800 = "mix.prim.add"(%3794, %3799) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3801 = "mix.prim.reshape"(%3793) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3802 = "mix.prim.reshape"(%3800) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3803 = "mix.prim.reshape"(%3801) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3804 = "mix.prim.reshape"(%3802) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3805 = "mix.prim.transpose"(%3803) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3806 = "mix.prim.transpose"(%3804) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3807 = "mix.prim.transpose"(%3806) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3808 = "mix.prim.unsqueeze"(%3805) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3809 = "mix.prim.permute"(%3808) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3810 = "mix.prim.unsqueeze"(%3807) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3811 = "mix.prim.permute"(%3810) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3812 = "mix.prim.permute"(%3809) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3813 = "mix.prim.reshape"(%3812) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3814 = "mix.prim.permute"(%3811) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3815 = "mix.prim.reshape"(%3814) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3816 = "mix.prim.batch_matmul"(%3813, %3815) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3817 = "mix.prim.reshape"(%3816) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3818 = "mix.prim.permute"(%3817) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3819 = "mix.prim.reshape"(%3818) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3820 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3821 = "mix.prim.mul"(%3819, %3820) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3822 = "mix.prim.reshape"(%3821) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3823 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3824 = "mix.comp.masked_fill"(%3822, %4, %3823) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3825 = "mix.comp.softmax"(%3824) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3826 = "mix.prim.reshape"(%3825) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3827 = "mix.prim.reshape"(%3754) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3828 = "mix.prim.transpose"(%3827) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3829 = "mix.prim.batch_matmul"(%3826, %3828) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3830 = "mix.prim.reshape"(%3829) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3831 = "mix.prim.permute"(%3830) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3832 = "mix.prim.reshape"(%3831) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3833 = "mix.prim.reshape"(%3832) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3834 = "mix.prim.transpose"(%3740) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3835 = "mix.prim.matmul"(%3833, %3834) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3836 = "mix.prim.add"(%3835, %3741) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3837 = "mix.prim.reshape"(%3836) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3838 = "mix.prim.mul"(%3728, %3837) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3839 = "mix.comp.weight"() <{param_loc = "transformer.h.28.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3840 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3841 = "mix.prim.pow"(%3838, %3840) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3842 = "mix.comp.mean"(%3841) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3843 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3844 = "mix.prim.add"(%3842, %3843) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3845 = "mix.prim.rsqrt"(%3844) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3846 = "mix.prim.mul"(%3838, %3845) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3847 = "mix.prim.mul"(%3839, %3846) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3848 = "mix.comp.weight"() <{param_loc = "transformer.h.28.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3849 = "mix.prim.reshape"(%3848) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3850 = "mix.prim.batch_matmul"(%3847, %3849) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3851 = "mix.comp.silu"(%3850) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3852 = "mix.comp.weight"() <{param_loc = "transformer.h.28.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3853 = "mix.prim.reshape"(%3852) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3854 = "mix.prim.batch_matmul"(%3847, %3853) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3855 = "mix.prim.mul"(%3851, %3854) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3856 = "mix.comp.weight"() <{param_loc = "transformer.h.28.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %3857 = "mix.prim.reshape"(%3856) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %3858 = "mix.prim.batch_matmul"(%3855, %3857) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %3859 = "mix.comp.weight"() <{param_loc = "transformer.h.28.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %3860 = "mix.prim.add"(%3858, %3859) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %3861 = "mix.prim.add"(%3860, %3838) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3862 = "mix.comp.weight"() <{param_loc = "transformer.h.29.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3863 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3864 = "mix.prim.pow"(%3861, %3863) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3865 = "mix.comp.mean"(%3864) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3866 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3867 = "mix.prim.add"(%3865, %3866) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3868 = "mix.prim.rsqrt"(%3867) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3869 = "mix.prim.mul"(%3861, %3868) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3870 = "mix.prim.mul"(%3862, %3869) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3871 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3872 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3873 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3874 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3875 = "mix.prim.transpose"(%3870) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3876 = "mix.prim.transpose"(%3871) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3877 = "mix.prim.reshape"(%3875) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3878 = "mix.prim.matmul"(%3877, %3876) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3879 = "mix.prim.reshape"(%3878) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3880 = "mix.prim.reshape"(%3879) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3881 = "mix.prim.transpose"(%3872) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3882 = "mix.prim.reshape"(%3875) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3883 = "mix.prim.matmul"(%3882, %3881) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3884 = "mix.prim.reshape"(%3883) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3885 = "mix.prim.reshape"(%3884) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3886 = "mix.prim.slice"(%3885) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3887 = "mix.prim.slice"(%3885) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3888 = "mix.prim.reshape"(%3880) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3889 = "mix.prim.reshape"(%3886) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3890 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3891 = "mix.prim.convert"(%3890) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3892 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3893 = "mix.prim.div"(%3891, %3892) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3894 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3895 = "mix.prim.pow"(%3894, %3893) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3896 = "mix.prim.reciprocal"(%3895) : (tensor<80xf16>) -> tensor<80xf16>
    %3897 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3898 = "mix.prim.mul"(%3897, %3896) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3899 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3900 = "mix.prim.unsqueeze"(%3899) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3901 = "mix.prim.permute"(%3900) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3902 = "mix.prim.unsqueeze"(%3898) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3903 = "mix.prim.permute"(%3902) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3904 = "mix.prim.mul"(%3901, %3903) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3905 = "mix.prim.concat"(%3904, %3904) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3906 = "mix.prim.cos"(%3905) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3907 = "mix.prim.slice"(%3906) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3908 = "mix.prim.unsqueeze"(%3907) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3909 = "mix.prim.slice"(%3908) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3910 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3911 = "mix.prim.mul"(%3909, %3910) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3912 = "mix.prim.sin"(%3905) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3913 = "mix.prim.slice"(%3912) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3914 = "mix.prim.unsqueeze"(%3913) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3915 = "mix.prim.slice"(%3914) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3916 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3917 = "mix.prim.mul"(%3915, %3916) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3918 = "mix.prim.slice"(%3911) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3919 = "mix.prim.slice"(%3917) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3920 = "mix.prim.mul"(%3888, %3918) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3921 = "mix.prim.slice"(%3888) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3922 = "mix.prim.slice"(%3888) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3923 = "mix.prim.neg"(%3922) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3924 = "mix.prim.concat"(%3923, %3921) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3925 = "mix.prim.mul"(%3924, %3919) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3926 = "mix.prim.add"(%3920, %3925) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3927 = "mix.prim.mul"(%3889, %3918) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3928 = "mix.prim.slice"(%3889) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3929 = "mix.prim.slice"(%3889) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3930 = "mix.prim.neg"(%3929) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3931 = "mix.prim.concat"(%3930, %3928) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3932 = "mix.prim.mul"(%3931, %3919) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3933 = "mix.prim.add"(%3927, %3932) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3934 = "mix.prim.reshape"(%3926) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3935 = "mix.prim.reshape"(%3933) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3936 = "mix.prim.reshape"(%3934) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3937 = "mix.prim.reshape"(%3935) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3938 = "mix.prim.transpose"(%3936) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3939 = "mix.prim.transpose"(%3937) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3940 = "mix.prim.transpose"(%3939) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3941 = "mix.prim.unsqueeze"(%3938) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3942 = "mix.prim.permute"(%3941) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3943 = "mix.prim.unsqueeze"(%3940) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3944 = "mix.prim.permute"(%3943) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3945 = "mix.prim.permute"(%3942) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3946 = "mix.prim.reshape"(%3945) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3947 = "mix.prim.permute"(%3944) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3948 = "mix.prim.reshape"(%3947) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3949 = "mix.prim.batch_matmul"(%3946, %3948) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3950 = "mix.prim.reshape"(%3949) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3951 = "mix.prim.permute"(%3950) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3952 = "mix.prim.reshape"(%3951) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3953 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3954 = "mix.prim.mul"(%3952, %3953) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3955 = "mix.prim.reshape"(%3954) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3956 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3957 = "mix.comp.masked_fill"(%3955, %4, %3956) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3958 = "mix.comp.softmax"(%3957) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3959 = "mix.prim.reshape"(%3958) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3960 = "mix.prim.reshape"(%3887) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3961 = "mix.prim.transpose"(%3960) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3962 = "mix.prim.batch_matmul"(%3959, %3961) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3963 = "mix.prim.reshape"(%3962) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3964 = "mix.prim.permute"(%3963) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3965 = "mix.prim.reshape"(%3964) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3966 = "mix.prim.reshape"(%3965) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3967 = "mix.prim.transpose"(%3873) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3968 = "mix.prim.matmul"(%3966, %3967) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3969 = "mix.prim.add"(%3968, %3874) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3970 = "mix.prim.reshape"(%3969) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3971 = "mix.prim.mul"(%3861, %3970) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3972 = "mix.comp.weight"() <{param_loc = "transformer.h.29.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3973 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3974 = "mix.prim.pow"(%3971, %3973) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3975 = "mix.comp.mean"(%3974) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3976 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3977 = "mix.prim.add"(%3975, %3976) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3978 = "mix.prim.rsqrt"(%3977) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3979 = "mix.prim.mul"(%3971, %3978) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3980 = "mix.prim.mul"(%3972, %3979) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3981 = "mix.comp.weight"() <{param_loc = "transformer.h.29.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3982 = "mix.prim.reshape"(%3981) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3983 = "mix.prim.batch_matmul"(%3980, %3982) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3984 = "mix.comp.silu"(%3983) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3985 = "mix.comp.weight"() <{param_loc = "transformer.h.29.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %3986 = "mix.prim.reshape"(%3985) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %3987 = "mix.prim.batch_matmul"(%3980, %3986) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %3988 = "mix.prim.mul"(%3984, %3987) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3989 = "mix.comp.weight"() <{param_loc = "transformer.h.29.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %3990 = "mix.prim.reshape"(%3989) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %3991 = "mix.prim.batch_matmul"(%3988, %3990) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %3992 = "mix.comp.weight"() <{param_loc = "transformer.h.29.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %3993 = "mix.prim.add"(%3991, %3992) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %3994 = "mix.prim.add"(%3993, %3971) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3995 = "mix.comp.weight"() <{param_loc = "transformer.h.30.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3996 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3997 = "mix.prim.pow"(%3994, %3996) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3998 = "mix.comp.mean"(%3997) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3999 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4000 = "mix.prim.add"(%3998, %3999) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4001 = "mix.prim.rsqrt"(%4000) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4002 = "mix.prim.mul"(%3994, %4001) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4003 = "mix.prim.mul"(%3995, %4002) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4004 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4005 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4006 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4007 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4008 = "mix.prim.transpose"(%4003) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4009 = "mix.prim.transpose"(%4004) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4010 = "mix.prim.reshape"(%4008) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4011 = "mix.prim.matmul"(%4010, %4009) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4012 = "mix.prim.reshape"(%4011) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4013 = "mix.prim.reshape"(%4012) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4014 = "mix.prim.transpose"(%4005) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4015 = "mix.prim.reshape"(%4008) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4016 = "mix.prim.matmul"(%4015, %4014) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4017 = "mix.prim.reshape"(%4016) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4018 = "mix.prim.reshape"(%4017) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4019 = "mix.prim.slice"(%4018) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4020 = "mix.prim.slice"(%4018) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4021 = "mix.prim.reshape"(%4013) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4022 = "mix.prim.reshape"(%4019) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4023 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4024 = "mix.prim.convert"(%4023) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4025 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4026 = "mix.prim.div"(%4024, %4025) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4027 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4028 = "mix.prim.pow"(%4027, %4026) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4029 = "mix.prim.reciprocal"(%4028) : (tensor<80xf16>) -> tensor<80xf16>
    %4030 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4031 = "mix.prim.mul"(%4030, %4029) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4032 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4033 = "mix.prim.unsqueeze"(%4032) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4034 = "mix.prim.permute"(%4033) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4035 = "mix.prim.unsqueeze"(%4031) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4036 = "mix.prim.permute"(%4035) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4037 = "mix.prim.mul"(%4034, %4036) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4038 = "mix.prim.concat"(%4037, %4037) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4039 = "mix.prim.cos"(%4038) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4040 = "mix.prim.slice"(%4039) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4041 = "mix.prim.unsqueeze"(%4040) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4042 = "mix.prim.slice"(%4041) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4043 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4044 = "mix.prim.mul"(%4042, %4043) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4045 = "mix.prim.sin"(%4038) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4046 = "mix.prim.slice"(%4045) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4047 = "mix.prim.unsqueeze"(%4046) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4048 = "mix.prim.slice"(%4047) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4049 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4050 = "mix.prim.mul"(%4048, %4049) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4051 = "mix.prim.slice"(%4044) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4052 = "mix.prim.slice"(%4050) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4053 = "mix.prim.mul"(%4021, %4051) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4054 = "mix.prim.slice"(%4021) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4055 = "mix.prim.slice"(%4021) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4056 = "mix.prim.neg"(%4055) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4057 = "mix.prim.concat"(%4056, %4054) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4058 = "mix.prim.mul"(%4057, %4052) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4059 = "mix.prim.add"(%4053, %4058) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4060 = "mix.prim.mul"(%4022, %4051) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4061 = "mix.prim.slice"(%4022) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4062 = "mix.prim.slice"(%4022) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4063 = "mix.prim.neg"(%4062) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4064 = "mix.prim.concat"(%4063, %4061) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4065 = "mix.prim.mul"(%4064, %4052) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4066 = "mix.prim.add"(%4060, %4065) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4067 = "mix.prim.reshape"(%4059) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4068 = "mix.prim.reshape"(%4066) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4069 = "mix.prim.reshape"(%4067) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4070 = "mix.prim.reshape"(%4068) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4071 = "mix.prim.transpose"(%4069) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4072 = "mix.prim.transpose"(%4070) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4073 = "mix.prim.transpose"(%4072) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4074 = "mix.prim.unsqueeze"(%4071) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4075 = "mix.prim.permute"(%4074) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4076 = "mix.prim.unsqueeze"(%4073) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4077 = "mix.prim.permute"(%4076) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4078 = "mix.prim.permute"(%4075) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4079 = "mix.prim.reshape"(%4078) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4080 = "mix.prim.permute"(%4077) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4081 = "mix.prim.reshape"(%4080) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4082 = "mix.prim.batch_matmul"(%4079, %4081) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4083 = "mix.prim.reshape"(%4082) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4084 = "mix.prim.permute"(%4083) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4085 = "mix.prim.reshape"(%4084) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4086 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4087 = "mix.prim.mul"(%4085, %4086) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4088 = "mix.prim.reshape"(%4087) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4089 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4090 = "mix.comp.masked_fill"(%4088, %4, %4089) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4091 = "mix.comp.softmax"(%4090) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4092 = "mix.prim.reshape"(%4091) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4093 = "mix.prim.reshape"(%4020) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4094 = "mix.prim.transpose"(%4093) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4095 = "mix.prim.batch_matmul"(%4092, %4094) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4096 = "mix.prim.reshape"(%4095) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4097 = "mix.prim.permute"(%4096) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4098 = "mix.prim.reshape"(%4097) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4099 = "mix.prim.reshape"(%4098) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4100 = "mix.prim.transpose"(%4006) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4101 = "mix.prim.matmul"(%4099, %4100) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4102 = "mix.prim.add"(%4101, %4007) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4103 = "mix.prim.reshape"(%4102) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4104 = "mix.prim.mul"(%3994, %4103) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4105 = "mix.comp.weight"() <{param_loc = "transformer.h.30.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4106 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4107 = "mix.prim.pow"(%4104, %4106) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4108 = "mix.comp.mean"(%4107) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4109 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4110 = "mix.prim.add"(%4108, %4109) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4111 = "mix.prim.rsqrt"(%4110) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4112 = "mix.prim.mul"(%4104, %4111) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4113 = "mix.prim.mul"(%4105, %4112) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4114 = "mix.comp.weight"() <{param_loc = "transformer.h.30.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4115 = "mix.prim.reshape"(%4114) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4116 = "mix.prim.batch_matmul"(%4113, %4115) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4117 = "mix.comp.silu"(%4116) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4118 = "mix.comp.weight"() <{param_loc = "transformer.h.30.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4119 = "mix.prim.reshape"(%4118) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4120 = "mix.prim.batch_matmul"(%4113, %4119) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4121 = "mix.prim.mul"(%4117, %4120) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4122 = "mix.comp.weight"() <{param_loc = "transformer.h.30.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %4123 = "mix.prim.reshape"(%4122) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %4124 = "mix.prim.batch_matmul"(%4121, %4123) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %4125 = "mix.comp.weight"() <{param_loc = "transformer.h.30.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %4126 = "mix.prim.add"(%4124, %4125) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %4127 = "mix.prim.add"(%4126, %4104) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4128 = "mix.comp.weight"() <{param_loc = "transformer.h.31.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4129 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4130 = "mix.prim.pow"(%4127, %4129) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4131 = "mix.comp.mean"(%4130) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4132 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4133 = "mix.prim.add"(%4131, %4132) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4134 = "mix.prim.rsqrt"(%4133) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4135 = "mix.prim.mul"(%4127, %4134) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4136 = "mix.prim.mul"(%4128, %4135) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4137 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4138 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4139 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4140 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4141 = "mix.prim.transpose"(%4136) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4142 = "mix.prim.transpose"(%4137) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4143 = "mix.prim.reshape"(%4141) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4144 = "mix.prim.matmul"(%4143, %4142) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4145 = "mix.prim.reshape"(%4144) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4146 = "mix.prim.reshape"(%4145) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4147 = "mix.prim.transpose"(%4138) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4148 = "mix.prim.reshape"(%4141) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4149 = "mix.prim.matmul"(%4148, %4147) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4150 = "mix.prim.reshape"(%4149) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4151 = "mix.prim.reshape"(%4150) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4152 = "mix.prim.slice"(%4151) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4153 = "mix.prim.slice"(%4151) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4154 = "mix.prim.reshape"(%4146) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4155 = "mix.prim.reshape"(%4152) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4156 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4157 = "mix.prim.convert"(%4156) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4158 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4159 = "mix.prim.div"(%4157, %4158) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4160 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4161 = "mix.prim.pow"(%4160, %4159) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4162 = "mix.prim.reciprocal"(%4161) : (tensor<80xf16>) -> tensor<80xf16>
    %4163 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4164 = "mix.prim.mul"(%4163, %4162) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4165 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4166 = "mix.prim.unsqueeze"(%4165) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4167 = "mix.prim.permute"(%4166) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4168 = "mix.prim.unsqueeze"(%4164) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4169 = "mix.prim.permute"(%4168) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4170 = "mix.prim.mul"(%4167, %4169) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4171 = "mix.prim.concat"(%4170, %4170) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4172 = "mix.prim.cos"(%4171) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4173 = "mix.prim.slice"(%4172) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4174 = "mix.prim.unsqueeze"(%4173) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4175 = "mix.prim.slice"(%4174) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4176 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4177 = "mix.prim.mul"(%4175, %4176) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4178 = "mix.prim.sin"(%4171) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4179 = "mix.prim.slice"(%4178) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4180 = "mix.prim.unsqueeze"(%4179) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4181 = "mix.prim.slice"(%4180) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4182 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4183 = "mix.prim.mul"(%4181, %4182) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4184 = "mix.prim.slice"(%4177) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4185 = "mix.prim.slice"(%4183) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4186 = "mix.prim.mul"(%4154, %4184) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4187 = "mix.prim.slice"(%4154) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4188 = "mix.prim.slice"(%4154) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4189 = "mix.prim.neg"(%4188) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4190 = "mix.prim.concat"(%4189, %4187) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4191 = "mix.prim.mul"(%4190, %4185) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4192 = "mix.prim.add"(%4186, %4191) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4193 = "mix.prim.mul"(%4155, %4184) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4194 = "mix.prim.slice"(%4155) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4195 = "mix.prim.slice"(%4155) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4196 = "mix.prim.neg"(%4195) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4197 = "mix.prim.concat"(%4196, %4194) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4198 = "mix.prim.mul"(%4197, %4185) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4199 = "mix.prim.add"(%4193, %4198) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4200 = "mix.prim.reshape"(%4192) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4201 = "mix.prim.reshape"(%4199) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4202 = "mix.prim.reshape"(%4200) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4203 = "mix.prim.reshape"(%4201) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4204 = "mix.prim.transpose"(%4202) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4205 = "mix.prim.transpose"(%4203) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4206 = "mix.prim.transpose"(%4205) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4207 = "mix.prim.unsqueeze"(%4204) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4208 = "mix.prim.permute"(%4207) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4209 = "mix.prim.unsqueeze"(%4206) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4210 = "mix.prim.permute"(%4209) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4211 = "mix.prim.permute"(%4208) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4212 = "mix.prim.reshape"(%4211) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4213 = "mix.prim.permute"(%4210) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4214 = "mix.prim.reshape"(%4213) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4215 = "mix.prim.batch_matmul"(%4212, %4214) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4216 = "mix.prim.reshape"(%4215) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4217 = "mix.prim.permute"(%4216) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4218 = "mix.prim.reshape"(%4217) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4219 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4220 = "mix.prim.mul"(%4218, %4219) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4221 = "mix.prim.reshape"(%4220) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4222 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4223 = "mix.comp.masked_fill"(%4221, %4, %4222) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4224 = "mix.comp.softmax"(%4223) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4225 = "mix.prim.reshape"(%4224) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4226 = "mix.prim.reshape"(%4153) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4227 = "mix.prim.transpose"(%4226) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4228 = "mix.prim.batch_matmul"(%4225, %4227) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4229 = "mix.prim.reshape"(%4228) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4230 = "mix.prim.permute"(%4229) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4231 = "mix.prim.reshape"(%4230) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4232 = "mix.prim.reshape"(%4231) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4233 = "mix.prim.transpose"(%4139) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4234 = "mix.prim.matmul"(%4232, %4233) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4235 = "mix.prim.add"(%4234, %4140) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4236 = "mix.prim.reshape"(%4235) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4237 = "mix.prim.mul"(%4127, %4236) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4238 = "mix.comp.weight"() <{param_loc = "transformer.h.31.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4239 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4240 = "mix.prim.pow"(%4237, %4239) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4241 = "mix.comp.mean"(%4240) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4242 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4243 = "mix.prim.add"(%4241, %4242) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4244 = "mix.prim.rsqrt"(%4243) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4245 = "mix.prim.mul"(%4237, %4244) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4246 = "mix.prim.mul"(%4238, %4245) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4247 = "mix.comp.weight"() <{param_loc = "transformer.h.31.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4248 = "mix.prim.reshape"(%4247) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4249 = "mix.prim.batch_matmul"(%4246, %4248) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4250 = "mix.comp.silu"(%4249) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4251 = "mix.comp.weight"() <{param_loc = "transformer.h.31.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4252 = "mix.prim.reshape"(%4251) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4253 = "mix.prim.batch_matmul"(%4246, %4252) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4254 = "mix.prim.mul"(%4250, %4253) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4255 = "mix.comp.weight"() <{param_loc = "transformer.h.31.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %4256 = "mix.prim.reshape"(%4255) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %4257 = "mix.prim.batch_matmul"(%4254, %4256) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %4258 = "mix.comp.weight"() <{param_loc = "transformer.h.31.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %4259 = "mix.prim.add"(%4257, %4258) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %4260 = "mix.prim.add"(%4259, %4237) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4261 = "mix.comp.weight"() <{param_loc = "transformer.h.32.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4262 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4263 = "mix.prim.pow"(%4260, %4262) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4264 = "mix.comp.mean"(%4263) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4265 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4266 = "mix.prim.add"(%4264, %4265) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4267 = "mix.prim.rsqrt"(%4266) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4268 = "mix.prim.mul"(%4260, %4267) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4269 = "mix.prim.mul"(%4261, %4268) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4270 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4271 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4272 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4273 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4274 = "mix.prim.transpose"(%4269) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4275 = "mix.prim.transpose"(%4270) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4276 = "mix.prim.reshape"(%4274) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4277 = "mix.prim.matmul"(%4276, %4275) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4278 = "mix.prim.reshape"(%4277) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4279 = "mix.prim.reshape"(%4278) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4280 = "mix.prim.transpose"(%4271) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4281 = "mix.prim.reshape"(%4274) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4282 = "mix.prim.matmul"(%4281, %4280) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4283 = "mix.prim.reshape"(%4282) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4284 = "mix.prim.reshape"(%4283) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4285 = "mix.prim.slice"(%4284) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4286 = "mix.prim.slice"(%4284) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4287 = "mix.prim.reshape"(%4279) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4288 = "mix.prim.reshape"(%4285) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4289 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4290 = "mix.prim.convert"(%4289) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4291 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4292 = "mix.prim.div"(%4290, %4291) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4293 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4294 = "mix.prim.pow"(%4293, %4292) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4295 = "mix.prim.reciprocal"(%4294) : (tensor<80xf16>) -> tensor<80xf16>
    %4296 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4297 = "mix.prim.mul"(%4296, %4295) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4298 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4299 = "mix.prim.unsqueeze"(%4298) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4300 = "mix.prim.permute"(%4299) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4301 = "mix.prim.unsqueeze"(%4297) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4302 = "mix.prim.permute"(%4301) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4303 = "mix.prim.mul"(%4300, %4302) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4304 = "mix.prim.concat"(%4303, %4303) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4305 = "mix.prim.cos"(%4304) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4306 = "mix.prim.slice"(%4305) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4307 = "mix.prim.unsqueeze"(%4306) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4308 = "mix.prim.slice"(%4307) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4309 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4310 = "mix.prim.mul"(%4308, %4309) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4311 = "mix.prim.sin"(%4304) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4312 = "mix.prim.slice"(%4311) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4313 = "mix.prim.unsqueeze"(%4312) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4314 = "mix.prim.slice"(%4313) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4315 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4316 = "mix.prim.mul"(%4314, %4315) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4317 = "mix.prim.slice"(%4310) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4318 = "mix.prim.slice"(%4316) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4319 = "mix.prim.mul"(%4287, %4317) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4320 = "mix.prim.slice"(%4287) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4321 = "mix.prim.slice"(%4287) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4322 = "mix.prim.neg"(%4321) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4323 = "mix.prim.concat"(%4322, %4320) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4324 = "mix.prim.mul"(%4323, %4318) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4325 = "mix.prim.add"(%4319, %4324) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4326 = "mix.prim.mul"(%4288, %4317) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4327 = "mix.prim.slice"(%4288) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4328 = "mix.prim.slice"(%4288) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4329 = "mix.prim.neg"(%4328) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4330 = "mix.prim.concat"(%4329, %4327) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4331 = "mix.prim.mul"(%4330, %4318) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4332 = "mix.prim.add"(%4326, %4331) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4333 = "mix.prim.reshape"(%4325) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4334 = "mix.prim.reshape"(%4332) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4335 = "mix.prim.reshape"(%4333) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4336 = "mix.prim.reshape"(%4334) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4337 = "mix.prim.transpose"(%4335) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4338 = "mix.prim.transpose"(%4336) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4339 = "mix.prim.transpose"(%4338) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4340 = "mix.prim.unsqueeze"(%4337) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4341 = "mix.prim.permute"(%4340) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4342 = "mix.prim.unsqueeze"(%4339) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4343 = "mix.prim.permute"(%4342) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4344 = "mix.prim.permute"(%4341) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4345 = "mix.prim.reshape"(%4344) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4346 = "mix.prim.permute"(%4343) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4347 = "mix.prim.reshape"(%4346) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4348 = "mix.prim.batch_matmul"(%4345, %4347) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4349 = "mix.prim.reshape"(%4348) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4350 = "mix.prim.permute"(%4349) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4351 = "mix.prim.reshape"(%4350) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4352 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4353 = "mix.prim.mul"(%4351, %4352) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4354 = "mix.prim.reshape"(%4353) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4355 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4356 = "mix.comp.masked_fill"(%4354, %4, %4355) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4357 = "mix.comp.softmax"(%4356) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4358 = "mix.prim.reshape"(%4357) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4359 = "mix.prim.reshape"(%4286) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4360 = "mix.prim.transpose"(%4359) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4361 = "mix.prim.batch_matmul"(%4358, %4360) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4362 = "mix.prim.reshape"(%4361) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4363 = "mix.prim.permute"(%4362) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4364 = "mix.prim.reshape"(%4363) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4365 = "mix.prim.reshape"(%4364) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4366 = "mix.prim.transpose"(%4272) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4367 = "mix.prim.matmul"(%4365, %4366) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4368 = "mix.prim.add"(%4367, %4273) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4369 = "mix.prim.reshape"(%4368) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4370 = "mix.prim.mul"(%4260, %4369) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4371 = "mix.comp.weight"() <{param_loc = "transformer.h.32.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4372 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4373 = "mix.prim.pow"(%4370, %4372) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4374 = "mix.comp.mean"(%4373) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4375 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4376 = "mix.prim.add"(%4374, %4375) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4377 = "mix.prim.rsqrt"(%4376) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4378 = "mix.prim.mul"(%4370, %4377) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4379 = "mix.prim.mul"(%4371, %4378) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4380 = "mix.comp.weight"() <{param_loc = "transformer.h.32.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4381 = "mix.prim.reshape"(%4380) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4382 = "mix.prim.batch_matmul"(%4379, %4381) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4383 = "mix.comp.silu"(%4382) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4384 = "mix.comp.weight"() <{param_loc = "transformer.h.32.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4385 = "mix.prim.reshape"(%4384) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4386 = "mix.prim.batch_matmul"(%4379, %4385) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4387 = "mix.prim.mul"(%4383, %4386) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4388 = "mix.comp.weight"() <{param_loc = "transformer.h.32.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %4389 = "mix.prim.reshape"(%4388) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %4390 = "mix.prim.batch_matmul"(%4387, %4389) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %4391 = "mix.comp.weight"() <{param_loc = "transformer.h.32.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %4392 = "mix.prim.add"(%4390, %4391) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %4393 = "mix.prim.add"(%4392, %4370) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4394 = "mix.comp.weight"() <{param_loc = "transformer.h.33.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4395 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4396 = "mix.prim.pow"(%4393, %4395) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4397 = "mix.comp.mean"(%4396) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4398 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4399 = "mix.prim.add"(%4397, %4398) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4400 = "mix.prim.rsqrt"(%4399) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4401 = "mix.prim.mul"(%4393, %4400) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4402 = "mix.prim.mul"(%4394, %4401) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4403 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4404 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4405 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4406 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4407 = "mix.prim.transpose"(%4402) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4408 = "mix.prim.transpose"(%4403) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4409 = "mix.prim.reshape"(%4407) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4410 = "mix.prim.matmul"(%4409, %4408) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4411 = "mix.prim.reshape"(%4410) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4412 = "mix.prim.reshape"(%4411) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4413 = "mix.prim.transpose"(%4404) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4414 = "mix.prim.reshape"(%4407) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4415 = "mix.prim.matmul"(%4414, %4413) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4416 = "mix.prim.reshape"(%4415) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4417 = "mix.prim.reshape"(%4416) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4418 = "mix.prim.slice"(%4417) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4419 = "mix.prim.slice"(%4417) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4420 = "mix.prim.reshape"(%4412) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4421 = "mix.prim.reshape"(%4418) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4422 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4423 = "mix.prim.convert"(%4422) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4424 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4425 = "mix.prim.div"(%4423, %4424) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4426 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4427 = "mix.prim.pow"(%4426, %4425) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4428 = "mix.prim.reciprocal"(%4427) : (tensor<80xf16>) -> tensor<80xf16>
    %4429 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4430 = "mix.prim.mul"(%4429, %4428) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4431 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4432 = "mix.prim.unsqueeze"(%4431) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4433 = "mix.prim.permute"(%4432) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4434 = "mix.prim.unsqueeze"(%4430) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4435 = "mix.prim.permute"(%4434) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4436 = "mix.prim.mul"(%4433, %4435) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4437 = "mix.prim.concat"(%4436, %4436) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4438 = "mix.prim.cos"(%4437) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4439 = "mix.prim.slice"(%4438) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4440 = "mix.prim.unsqueeze"(%4439) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4441 = "mix.prim.slice"(%4440) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4442 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4443 = "mix.prim.mul"(%4441, %4442) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4444 = "mix.prim.sin"(%4437) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4445 = "mix.prim.slice"(%4444) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4446 = "mix.prim.unsqueeze"(%4445) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4447 = "mix.prim.slice"(%4446) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4448 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4449 = "mix.prim.mul"(%4447, %4448) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4450 = "mix.prim.slice"(%4443) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4451 = "mix.prim.slice"(%4449) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4452 = "mix.prim.mul"(%4420, %4450) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4453 = "mix.prim.slice"(%4420) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4454 = "mix.prim.slice"(%4420) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4455 = "mix.prim.neg"(%4454) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4456 = "mix.prim.concat"(%4455, %4453) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4457 = "mix.prim.mul"(%4456, %4451) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4458 = "mix.prim.add"(%4452, %4457) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4459 = "mix.prim.mul"(%4421, %4450) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4460 = "mix.prim.slice"(%4421) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4461 = "mix.prim.slice"(%4421) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4462 = "mix.prim.neg"(%4461) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4463 = "mix.prim.concat"(%4462, %4460) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4464 = "mix.prim.mul"(%4463, %4451) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4465 = "mix.prim.add"(%4459, %4464) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4466 = "mix.prim.reshape"(%4458) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4467 = "mix.prim.reshape"(%4465) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4468 = "mix.prim.reshape"(%4466) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4469 = "mix.prim.reshape"(%4467) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4470 = "mix.prim.transpose"(%4468) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4471 = "mix.prim.transpose"(%4469) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4472 = "mix.prim.transpose"(%4471) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4473 = "mix.prim.unsqueeze"(%4470) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4474 = "mix.prim.permute"(%4473) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4475 = "mix.prim.unsqueeze"(%4472) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4476 = "mix.prim.permute"(%4475) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4477 = "mix.prim.permute"(%4474) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4478 = "mix.prim.reshape"(%4477) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4479 = "mix.prim.permute"(%4476) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4480 = "mix.prim.reshape"(%4479) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4481 = "mix.prim.batch_matmul"(%4478, %4480) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4482 = "mix.prim.reshape"(%4481) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4483 = "mix.prim.permute"(%4482) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4484 = "mix.prim.reshape"(%4483) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4485 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4486 = "mix.prim.mul"(%4484, %4485) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4487 = "mix.prim.reshape"(%4486) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4488 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4489 = "mix.comp.masked_fill"(%4487, %4, %4488) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4490 = "mix.comp.softmax"(%4489) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4491 = "mix.prim.reshape"(%4490) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4492 = "mix.prim.reshape"(%4419) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4493 = "mix.prim.transpose"(%4492) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4494 = "mix.prim.batch_matmul"(%4491, %4493) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4495 = "mix.prim.reshape"(%4494) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4496 = "mix.prim.permute"(%4495) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4497 = "mix.prim.reshape"(%4496) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4498 = "mix.prim.reshape"(%4497) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4499 = "mix.prim.transpose"(%4405) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4500 = "mix.prim.matmul"(%4498, %4499) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4501 = "mix.prim.add"(%4500, %4406) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4502 = "mix.prim.reshape"(%4501) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4503 = "mix.prim.mul"(%4393, %4502) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4504 = "mix.comp.weight"() <{param_loc = "transformer.h.33.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4505 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4506 = "mix.prim.pow"(%4503, %4505) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4507 = "mix.comp.mean"(%4506) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4508 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4509 = "mix.prim.add"(%4507, %4508) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4510 = "mix.prim.rsqrt"(%4509) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4511 = "mix.prim.mul"(%4503, %4510) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4512 = "mix.prim.mul"(%4504, %4511) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4513 = "mix.comp.weight"() <{param_loc = "transformer.h.33.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4514 = "mix.prim.reshape"(%4513) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4515 = "mix.prim.batch_matmul"(%4512, %4514) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4516 = "mix.comp.silu"(%4515) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4517 = "mix.comp.weight"() <{param_loc = "transformer.h.33.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4518 = "mix.prim.reshape"(%4517) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4519 = "mix.prim.batch_matmul"(%4512, %4518) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4520 = "mix.prim.mul"(%4516, %4519) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4521 = "mix.comp.weight"() <{param_loc = "transformer.h.33.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %4522 = "mix.prim.reshape"(%4521) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %4523 = "mix.prim.batch_matmul"(%4520, %4522) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %4524 = "mix.comp.weight"() <{param_loc = "transformer.h.33.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %4525 = "mix.prim.add"(%4523, %4524) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %4526 = "mix.prim.add"(%4525, %4503) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4527 = "mix.comp.weight"() <{param_loc = "transformer.h.34.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4528 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4529 = "mix.prim.pow"(%4526, %4528) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4530 = "mix.comp.mean"(%4529) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4531 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4532 = "mix.prim.add"(%4530, %4531) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4533 = "mix.prim.rsqrt"(%4532) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4534 = "mix.prim.mul"(%4526, %4533) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4535 = "mix.prim.mul"(%4527, %4534) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4536 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4537 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4538 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4539 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4540 = "mix.prim.transpose"(%4535) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4541 = "mix.prim.transpose"(%4536) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4542 = "mix.prim.reshape"(%4540) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4543 = "mix.prim.matmul"(%4542, %4541) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4544 = "mix.prim.reshape"(%4543) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4545 = "mix.prim.reshape"(%4544) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4546 = "mix.prim.transpose"(%4537) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4547 = "mix.prim.reshape"(%4540) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4548 = "mix.prim.matmul"(%4547, %4546) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4549 = "mix.prim.reshape"(%4548) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4550 = "mix.prim.reshape"(%4549) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4551 = "mix.prim.slice"(%4550) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4552 = "mix.prim.slice"(%4550) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4553 = "mix.prim.reshape"(%4545) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4554 = "mix.prim.reshape"(%4551) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4555 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4556 = "mix.prim.convert"(%4555) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4557 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4558 = "mix.prim.div"(%4556, %4557) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4559 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4560 = "mix.prim.pow"(%4559, %4558) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4561 = "mix.prim.reciprocal"(%4560) : (tensor<80xf16>) -> tensor<80xf16>
    %4562 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4563 = "mix.prim.mul"(%4562, %4561) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4564 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4565 = "mix.prim.unsqueeze"(%4564) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4566 = "mix.prim.permute"(%4565) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4567 = "mix.prim.unsqueeze"(%4563) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4568 = "mix.prim.permute"(%4567) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4569 = "mix.prim.mul"(%4566, %4568) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4570 = "mix.prim.concat"(%4569, %4569) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4571 = "mix.prim.cos"(%4570) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4572 = "mix.prim.slice"(%4571) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4573 = "mix.prim.unsqueeze"(%4572) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4574 = "mix.prim.slice"(%4573) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4575 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4576 = "mix.prim.mul"(%4574, %4575) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4577 = "mix.prim.sin"(%4570) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4578 = "mix.prim.slice"(%4577) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4579 = "mix.prim.unsqueeze"(%4578) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4580 = "mix.prim.slice"(%4579) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4581 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4582 = "mix.prim.mul"(%4580, %4581) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4583 = "mix.prim.slice"(%4576) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4584 = "mix.prim.slice"(%4582) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4585 = "mix.prim.mul"(%4553, %4583) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4586 = "mix.prim.slice"(%4553) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4587 = "mix.prim.slice"(%4553) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4588 = "mix.prim.neg"(%4587) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4589 = "mix.prim.concat"(%4588, %4586) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4590 = "mix.prim.mul"(%4589, %4584) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4591 = "mix.prim.add"(%4585, %4590) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4592 = "mix.prim.mul"(%4554, %4583) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4593 = "mix.prim.slice"(%4554) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4594 = "mix.prim.slice"(%4554) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4595 = "mix.prim.neg"(%4594) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4596 = "mix.prim.concat"(%4595, %4593) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4597 = "mix.prim.mul"(%4596, %4584) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4598 = "mix.prim.add"(%4592, %4597) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4599 = "mix.prim.reshape"(%4591) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4600 = "mix.prim.reshape"(%4598) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4601 = "mix.prim.reshape"(%4599) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4602 = "mix.prim.reshape"(%4600) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4603 = "mix.prim.transpose"(%4601) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4604 = "mix.prim.transpose"(%4602) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4605 = "mix.prim.transpose"(%4604) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4606 = "mix.prim.unsqueeze"(%4603) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4607 = "mix.prim.permute"(%4606) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4608 = "mix.prim.unsqueeze"(%4605) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4609 = "mix.prim.permute"(%4608) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4610 = "mix.prim.permute"(%4607) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4611 = "mix.prim.reshape"(%4610) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4612 = "mix.prim.permute"(%4609) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4613 = "mix.prim.reshape"(%4612) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4614 = "mix.prim.batch_matmul"(%4611, %4613) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4615 = "mix.prim.reshape"(%4614) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4616 = "mix.prim.permute"(%4615) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4617 = "mix.prim.reshape"(%4616) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4618 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4619 = "mix.prim.mul"(%4617, %4618) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4620 = "mix.prim.reshape"(%4619) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4621 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4622 = "mix.comp.masked_fill"(%4620, %4, %4621) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4623 = "mix.comp.softmax"(%4622) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4624 = "mix.prim.reshape"(%4623) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4625 = "mix.prim.reshape"(%4552) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4626 = "mix.prim.transpose"(%4625) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4627 = "mix.prim.batch_matmul"(%4624, %4626) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4628 = "mix.prim.reshape"(%4627) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4629 = "mix.prim.permute"(%4628) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4630 = "mix.prim.reshape"(%4629) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4631 = "mix.prim.reshape"(%4630) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4632 = "mix.prim.transpose"(%4538) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4633 = "mix.prim.matmul"(%4631, %4632) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4634 = "mix.prim.add"(%4633, %4539) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4635 = "mix.prim.reshape"(%4634) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4636 = "mix.prim.mul"(%4526, %4635) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4637 = "mix.comp.weight"() <{param_loc = "transformer.h.34.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4638 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4639 = "mix.prim.pow"(%4636, %4638) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4640 = "mix.comp.mean"(%4639) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4641 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4642 = "mix.prim.add"(%4640, %4641) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4643 = "mix.prim.rsqrt"(%4642) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4644 = "mix.prim.mul"(%4636, %4643) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4645 = "mix.prim.mul"(%4637, %4644) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4646 = "mix.comp.weight"() <{param_loc = "transformer.h.34.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4647 = "mix.prim.reshape"(%4646) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4648 = "mix.prim.batch_matmul"(%4645, %4647) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4649 = "mix.comp.silu"(%4648) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4650 = "mix.comp.weight"() <{param_loc = "transformer.h.34.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4651 = "mix.prim.reshape"(%4650) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4652 = "mix.prim.batch_matmul"(%4645, %4651) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4653 = "mix.prim.mul"(%4649, %4652) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4654 = "mix.comp.weight"() <{param_loc = "transformer.h.34.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %4655 = "mix.prim.reshape"(%4654) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %4656 = "mix.prim.batch_matmul"(%4653, %4655) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %4657 = "mix.comp.weight"() <{param_loc = "transformer.h.34.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %4658 = "mix.prim.add"(%4656, %4657) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %4659 = "mix.prim.add"(%4658, %4636) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4660 = "mix.comp.weight"() <{param_loc = "transformer.h.35.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4661 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4662 = "mix.prim.pow"(%4659, %4661) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4663 = "mix.comp.mean"(%4662) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4664 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4665 = "mix.prim.add"(%4663, %4664) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4666 = "mix.prim.rsqrt"(%4665) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4667 = "mix.prim.mul"(%4659, %4666) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4668 = "mix.prim.mul"(%4660, %4667) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4669 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4670 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4671 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4672 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4673 = "mix.prim.transpose"(%4668) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4674 = "mix.prim.transpose"(%4669) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4675 = "mix.prim.reshape"(%4673) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4676 = "mix.prim.matmul"(%4675, %4674) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4677 = "mix.prim.reshape"(%4676) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4678 = "mix.prim.reshape"(%4677) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4679 = "mix.prim.transpose"(%4670) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4680 = "mix.prim.reshape"(%4673) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4681 = "mix.prim.matmul"(%4680, %4679) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4682 = "mix.prim.reshape"(%4681) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4683 = "mix.prim.reshape"(%4682) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4684 = "mix.prim.slice"(%4683) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4685 = "mix.prim.slice"(%4683) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4686 = "mix.prim.reshape"(%4678) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4687 = "mix.prim.reshape"(%4684) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4688 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4689 = "mix.prim.convert"(%4688) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4690 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4691 = "mix.prim.div"(%4689, %4690) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4692 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4693 = "mix.prim.pow"(%4692, %4691) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4694 = "mix.prim.reciprocal"(%4693) : (tensor<80xf16>) -> tensor<80xf16>
    %4695 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4696 = "mix.prim.mul"(%4695, %4694) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4697 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4698 = "mix.prim.unsqueeze"(%4697) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4699 = "mix.prim.permute"(%4698) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4700 = "mix.prim.unsqueeze"(%4696) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4701 = "mix.prim.permute"(%4700) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4702 = "mix.prim.mul"(%4699, %4701) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4703 = "mix.prim.concat"(%4702, %4702) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4704 = "mix.prim.cos"(%4703) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4705 = "mix.prim.slice"(%4704) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4706 = "mix.prim.unsqueeze"(%4705) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4707 = "mix.prim.slice"(%4706) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4708 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4709 = "mix.prim.mul"(%4707, %4708) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4710 = "mix.prim.sin"(%4703) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4711 = "mix.prim.slice"(%4710) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4712 = "mix.prim.unsqueeze"(%4711) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4713 = "mix.prim.slice"(%4712) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4714 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4715 = "mix.prim.mul"(%4713, %4714) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4716 = "mix.prim.slice"(%4709) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4717 = "mix.prim.slice"(%4715) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4718 = "mix.prim.mul"(%4686, %4716) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4719 = "mix.prim.slice"(%4686) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4720 = "mix.prim.slice"(%4686) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4721 = "mix.prim.neg"(%4720) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4722 = "mix.prim.concat"(%4721, %4719) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4723 = "mix.prim.mul"(%4722, %4717) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4724 = "mix.prim.add"(%4718, %4723) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4725 = "mix.prim.mul"(%4687, %4716) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4726 = "mix.prim.slice"(%4687) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4727 = "mix.prim.slice"(%4687) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4728 = "mix.prim.neg"(%4727) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4729 = "mix.prim.concat"(%4728, %4726) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4730 = "mix.prim.mul"(%4729, %4717) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4731 = "mix.prim.add"(%4725, %4730) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4732 = "mix.prim.reshape"(%4724) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4733 = "mix.prim.reshape"(%4731) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4734 = "mix.prim.reshape"(%4732) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4735 = "mix.prim.reshape"(%4733) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4736 = "mix.prim.transpose"(%4734) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4737 = "mix.prim.transpose"(%4735) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4738 = "mix.prim.transpose"(%4737) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4739 = "mix.prim.unsqueeze"(%4736) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4740 = "mix.prim.permute"(%4739) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4741 = "mix.prim.unsqueeze"(%4738) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4742 = "mix.prim.permute"(%4741) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4743 = "mix.prim.permute"(%4740) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4744 = "mix.prim.reshape"(%4743) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4745 = "mix.prim.permute"(%4742) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4746 = "mix.prim.reshape"(%4745) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4747 = "mix.prim.batch_matmul"(%4744, %4746) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4748 = "mix.prim.reshape"(%4747) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4749 = "mix.prim.permute"(%4748) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4750 = "mix.prim.reshape"(%4749) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4751 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4752 = "mix.prim.mul"(%4750, %4751) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4753 = "mix.prim.reshape"(%4752) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4754 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4755 = "mix.comp.masked_fill"(%4753, %4, %4754) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4756 = "mix.comp.softmax"(%4755) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4757 = "mix.prim.reshape"(%4756) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4758 = "mix.prim.reshape"(%4685) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4759 = "mix.prim.transpose"(%4758) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4760 = "mix.prim.batch_matmul"(%4757, %4759) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4761 = "mix.prim.reshape"(%4760) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4762 = "mix.prim.permute"(%4761) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4763 = "mix.prim.reshape"(%4762) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4764 = "mix.prim.reshape"(%4763) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4765 = "mix.prim.transpose"(%4671) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4766 = "mix.prim.matmul"(%4764, %4765) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4767 = "mix.prim.add"(%4766, %4672) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4768 = "mix.prim.reshape"(%4767) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4769 = "mix.prim.mul"(%4659, %4768) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4770 = "mix.comp.weight"() <{param_loc = "transformer.h.35.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4771 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4772 = "mix.prim.pow"(%4769, %4771) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4773 = "mix.comp.mean"(%4772) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4774 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4775 = "mix.prim.add"(%4773, %4774) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4776 = "mix.prim.rsqrt"(%4775) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4777 = "mix.prim.mul"(%4769, %4776) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4778 = "mix.prim.mul"(%4770, %4777) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4779 = "mix.comp.weight"() <{param_loc = "transformer.h.35.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4780 = "mix.prim.reshape"(%4779) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4781 = "mix.prim.batch_matmul"(%4778, %4780) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4782 = "mix.comp.silu"(%4781) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4783 = "mix.comp.weight"() <{param_loc = "transformer.h.35.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4784 = "mix.prim.reshape"(%4783) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4785 = "mix.prim.batch_matmul"(%4778, %4784) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4786 = "mix.prim.mul"(%4782, %4785) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4787 = "mix.comp.weight"() <{param_loc = "transformer.h.35.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %4788 = "mix.prim.reshape"(%4787) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %4789 = "mix.prim.batch_matmul"(%4786, %4788) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %4790 = "mix.comp.weight"() <{param_loc = "transformer.h.35.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %4791 = "mix.prim.add"(%4789, %4790) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %4792 = "mix.prim.add"(%4791, %4769) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4793 = "mix.comp.weight"() <{param_loc = "transformer.h.36.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4794 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4795 = "mix.prim.pow"(%4792, %4794) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4796 = "mix.comp.mean"(%4795) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4797 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4798 = "mix.prim.add"(%4796, %4797) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4799 = "mix.prim.rsqrt"(%4798) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4800 = "mix.prim.mul"(%4792, %4799) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4801 = "mix.prim.mul"(%4793, %4800) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4802 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4803 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4804 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4805 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4806 = "mix.prim.transpose"(%4801) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4807 = "mix.prim.transpose"(%4802) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4808 = "mix.prim.reshape"(%4806) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4809 = "mix.prim.matmul"(%4808, %4807) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4810 = "mix.prim.reshape"(%4809) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4811 = "mix.prim.reshape"(%4810) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4812 = "mix.prim.transpose"(%4803) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4813 = "mix.prim.reshape"(%4806) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4814 = "mix.prim.matmul"(%4813, %4812) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4815 = "mix.prim.reshape"(%4814) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4816 = "mix.prim.reshape"(%4815) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4817 = "mix.prim.slice"(%4816) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4818 = "mix.prim.slice"(%4816) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4819 = "mix.prim.reshape"(%4811) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4820 = "mix.prim.reshape"(%4817) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4821 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4822 = "mix.prim.convert"(%4821) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4823 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4824 = "mix.prim.div"(%4822, %4823) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4825 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4826 = "mix.prim.pow"(%4825, %4824) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4827 = "mix.prim.reciprocal"(%4826) : (tensor<80xf16>) -> tensor<80xf16>
    %4828 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4829 = "mix.prim.mul"(%4828, %4827) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4830 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4831 = "mix.prim.unsqueeze"(%4830) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4832 = "mix.prim.permute"(%4831) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4833 = "mix.prim.unsqueeze"(%4829) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4834 = "mix.prim.permute"(%4833) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4835 = "mix.prim.mul"(%4832, %4834) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4836 = "mix.prim.concat"(%4835, %4835) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4837 = "mix.prim.cos"(%4836) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4838 = "mix.prim.slice"(%4837) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4839 = "mix.prim.unsqueeze"(%4838) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4840 = "mix.prim.slice"(%4839) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4841 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4842 = "mix.prim.mul"(%4840, %4841) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4843 = "mix.prim.sin"(%4836) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4844 = "mix.prim.slice"(%4843) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4845 = "mix.prim.unsqueeze"(%4844) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4846 = "mix.prim.slice"(%4845) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4847 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4848 = "mix.prim.mul"(%4846, %4847) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4849 = "mix.prim.slice"(%4842) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4850 = "mix.prim.slice"(%4848) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4851 = "mix.prim.mul"(%4819, %4849) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4852 = "mix.prim.slice"(%4819) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4853 = "mix.prim.slice"(%4819) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4854 = "mix.prim.neg"(%4853) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4855 = "mix.prim.concat"(%4854, %4852) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4856 = "mix.prim.mul"(%4855, %4850) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4857 = "mix.prim.add"(%4851, %4856) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4858 = "mix.prim.mul"(%4820, %4849) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4859 = "mix.prim.slice"(%4820) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4860 = "mix.prim.slice"(%4820) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4861 = "mix.prim.neg"(%4860) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4862 = "mix.prim.concat"(%4861, %4859) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4863 = "mix.prim.mul"(%4862, %4850) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4864 = "mix.prim.add"(%4858, %4863) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4865 = "mix.prim.reshape"(%4857) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4866 = "mix.prim.reshape"(%4864) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4867 = "mix.prim.reshape"(%4865) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4868 = "mix.prim.reshape"(%4866) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4869 = "mix.prim.transpose"(%4867) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4870 = "mix.prim.transpose"(%4868) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4871 = "mix.prim.transpose"(%4870) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4872 = "mix.prim.unsqueeze"(%4869) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4873 = "mix.prim.permute"(%4872) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4874 = "mix.prim.unsqueeze"(%4871) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4875 = "mix.prim.permute"(%4874) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4876 = "mix.prim.permute"(%4873) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4877 = "mix.prim.reshape"(%4876) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4878 = "mix.prim.permute"(%4875) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4879 = "mix.prim.reshape"(%4878) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4880 = "mix.prim.batch_matmul"(%4877, %4879) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4881 = "mix.prim.reshape"(%4880) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4882 = "mix.prim.permute"(%4881) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4883 = "mix.prim.reshape"(%4882) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4884 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4885 = "mix.prim.mul"(%4883, %4884) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4886 = "mix.prim.reshape"(%4885) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4887 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4888 = "mix.comp.masked_fill"(%4886, %4, %4887) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4889 = "mix.comp.softmax"(%4888) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4890 = "mix.prim.reshape"(%4889) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4891 = "mix.prim.reshape"(%4818) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4892 = "mix.prim.transpose"(%4891) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4893 = "mix.prim.batch_matmul"(%4890, %4892) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4894 = "mix.prim.reshape"(%4893) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4895 = "mix.prim.permute"(%4894) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4896 = "mix.prim.reshape"(%4895) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4897 = "mix.prim.reshape"(%4896) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4898 = "mix.prim.transpose"(%4804) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4899 = "mix.prim.matmul"(%4897, %4898) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4900 = "mix.prim.add"(%4899, %4805) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4901 = "mix.prim.reshape"(%4900) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4902 = "mix.prim.mul"(%4792, %4901) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4903 = "mix.comp.weight"() <{param_loc = "transformer.h.36.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4904 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4905 = "mix.prim.pow"(%4902, %4904) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4906 = "mix.comp.mean"(%4905) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4907 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4908 = "mix.prim.add"(%4906, %4907) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4909 = "mix.prim.rsqrt"(%4908) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4910 = "mix.prim.mul"(%4902, %4909) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4911 = "mix.prim.mul"(%4903, %4910) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4912 = "mix.comp.weight"() <{param_loc = "transformer.h.36.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4913 = "mix.prim.reshape"(%4912) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4914 = "mix.prim.batch_matmul"(%4911, %4913) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4915 = "mix.comp.silu"(%4914) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4916 = "mix.comp.weight"() <{param_loc = "transformer.h.36.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %4917 = "mix.prim.reshape"(%4916) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %4918 = "mix.prim.batch_matmul"(%4911, %4917) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %4919 = "mix.prim.mul"(%4915, %4918) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4920 = "mix.comp.weight"() <{param_loc = "transformer.h.36.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %4921 = "mix.prim.reshape"(%4920) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %4922 = "mix.prim.batch_matmul"(%4919, %4921) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %4923 = "mix.comp.weight"() <{param_loc = "transformer.h.36.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %4924 = "mix.prim.add"(%4922, %4923) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %4925 = "mix.prim.add"(%4924, %4902) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4926 = "mix.comp.weight"() <{param_loc = "transformer.h.37.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4927 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4928 = "mix.prim.pow"(%4925, %4927) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4929 = "mix.comp.mean"(%4928) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4930 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4931 = "mix.prim.add"(%4929, %4930) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4932 = "mix.prim.rsqrt"(%4931) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4933 = "mix.prim.mul"(%4925, %4932) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4934 = "mix.prim.mul"(%4926, %4933) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4935 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4936 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4937 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4938 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4939 = "mix.prim.transpose"(%4934) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4940 = "mix.prim.transpose"(%4935) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4941 = "mix.prim.reshape"(%4939) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4942 = "mix.prim.matmul"(%4941, %4940) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4943 = "mix.prim.reshape"(%4942) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4944 = "mix.prim.reshape"(%4943) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4945 = "mix.prim.transpose"(%4936) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4946 = "mix.prim.reshape"(%4939) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4947 = "mix.prim.matmul"(%4946, %4945) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4948 = "mix.prim.reshape"(%4947) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4949 = "mix.prim.reshape"(%4948) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4950 = "mix.prim.slice"(%4949) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4951 = "mix.prim.slice"(%4949) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4952 = "mix.prim.reshape"(%4944) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4953 = "mix.prim.reshape"(%4950) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4954 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4955 = "mix.prim.convert"(%4954) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4956 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4957 = "mix.prim.div"(%4955, %4956) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4958 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4959 = "mix.prim.pow"(%4958, %4957) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4960 = "mix.prim.reciprocal"(%4959) : (tensor<80xf16>) -> tensor<80xf16>
    %4961 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4962 = "mix.prim.mul"(%4961, %4960) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4963 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4964 = "mix.prim.unsqueeze"(%4963) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4965 = "mix.prim.permute"(%4964) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4966 = "mix.prim.unsqueeze"(%4962) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4967 = "mix.prim.permute"(%4966) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4968 = "mix.prim.mul"(%4965, %4967) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4969 = "mix.prim.concat"(%4968, %4968) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4970 = "mix.prim.cos"(%4969) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4971 = "mix.prim.slice"(%4970) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4972 = "mix.prim.unsqueeze"(%4971) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4973 = "mix.prim.slice"(%4972) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4974 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4975 = "mix.prim.mul"(%4973, %4974) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4976 = "mix.prim.sin"(%4969) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4977 = "mix.prim.slice"(%4976) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4978 = "mix.prim.unsqueeze"(%4977) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4979 = "mix.prim.slice"(%4978) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4980 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4981 = "mix.prim.mul"(%4979, %4980) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4982 = "mix.prim.slice"(%4975) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4983 = "mix.prim.slice"(%4981) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4984 = "mix.prim.mul"(%4952, %4982) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4985 = "mix.prim.slice"(%4952) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4986 = "mix.prim.slice"(%4952) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4987 = "mix.prim.neg"(%4986) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4988 = "mix.prim.concat"(%4987, %4985) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4989 = "mix.prim.mul"(%4988, %4983) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4990 = "mix.prim.add"(%4984, %4989) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4991 = "mix.prim.mul"(%4953, %4982) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4992 = "mix.prim.slice"(%4953) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4993 = "mix.prim.slice"(%4953) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4994 = "mix.prim.neg"(%4993) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4995 = "mix.prim.concat"(%4994, %4992) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4996 = "mix.prim.mul"(%4995, %4983) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4997 = "mix.prim.add"(%4991, %4996) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4998 = "mix.prim.reshape"(%4990) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4999 = "mix.prim.reshape"(%4997) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %5000 = "mix.prim.reshape"(%4998) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %5001 = "mix.prim.reshape"(%4999) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %5002 = "mix.prim.transpose"(%5000) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %5003 = "mix.prim.transpose"(%5001) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %5004 = "mix.prim.transpose"(%5003) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %5005 = "mix.prim.unsqueeze"(%5002) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %5006 = "mix.prim.permute"(%5005) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %5007 = "mix.prim.unsqueeze"(%5004) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %5008 = "mix.prim.permute"(%5007) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %5009 = "mix.prim.permute"(%5006) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %5010 = "mix.prim.reshape"(%5009) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %5011 = "mix.prim.permute"(%5008) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %5012 = "mix.prim.reshape"(%5011) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %5013 = "mix.prim.batch_matmul"(%5010, %5012) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %5014 = "mix.prim.reshape"(%5013) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %5015 = "mix.prim.permute"(%5014) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %5016 = "mix.prim.reshape"(%5015) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %5017 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %5018 = "mix.prim.mul"(%5016, %5017) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %5019 = "mix.prim.reshape"(%5018) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %5020 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %5021 = "mix.comp.masked_fill"(%5019, %4, %5020) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %5022 = "mix.comp.softmax"(%5021) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %5023 = "mix.prim.reshape"(%5022) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %5024 = "mix.prim.reshape"(%4951) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %5025 = "mix.prim.transpose"(%5024) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %5026 = "mix.prim.batch_matmul"(%5023, %5025) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %5027 = "mix.prim.reshape"(%5026) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %5028 = "mix.prim.permute"(%5027) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %5029 = "mix.prim.reshape"(%5028) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %5030 = "mix.prim.reshape"(%5029) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %5031 = "mix.prim.transpose"(%4937) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %5032 = "mix.prim.matmul"(%5030, %5031) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %5033 = "mix.prim.add"(%5032, %4938) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %5034 = "mix.prim.reshape"(%5033) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %5035 = "mix.prim.mul"(%4925, %5034) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %5036 = "mix.comp.weight"() <{param_loc = "transformer.h.37.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %5037 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %5038 = "mix.prim.pow"(%5035, %5037) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %5039 = "mix.comp.mean"(%5038) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %5040 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %5041 = "mix.prim.add"(%5039, %5040) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %5042 = "mix.prim.rsqrt"(%5041) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %5043 = "mix.prim.mul"(%5035, %5042) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %5044 = "mix.prim.mul"(%5036, %5043) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %5045 = "mix.comp.weight"() <{param_loc = "transformer.h.37.mlp.gate_proj.weight"}> : () -> tensor<5120x12288xf16>
    %5046 = "mix.prim.reshape"(%5045) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %5047 = "mix.prim.batch_matmul"(%5044, %5046) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %5048 = "mix.comp.silu"(%5047) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %5049 = "mix.comp.weight"() <{param_loc = "transformer.h.37.mlp.up_proj.weight"}> : () -> tensor<5120x12288xf16>
    %5050 = "mix.prim.reshape"(%5049) <{shape = [1, 5120, 12288]}> : (tensor<5120x12288xf16>) -> tensor<1x5120x12288xf16>
    %5051 = "mix.prim.batch_matmul"(%5044, %5050) : (tensor<1x40x5120xf16>, tensor<1x5120x12288xf16>) -> tensor<1x40x12288xf16>
    %5052 = "mix.prim.mul"(%5048, %5051) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %5053 = "mix.comp.weight"() <{param_loc = "transformer.h.37.mlp.down_proj.weight"}> : () -> tensor<12288x5120xf16>
    %5054 = "mix.prim.reshape"(%5053) <{shape = [1, 12288, 5120]}> : (tensor<12288x5120xf16>) -> tensor<1x12288x5120xf16>
    %5055 = "mix.prim.batch_matmul"(%5052, %5054) : (tensor<1x40x12288xf16>, tensor<1x12288x5120xf16>) -> tensor<1x40x5120xf16>
    %5056 = "mix.comp.weight"() <{param_loc = "transformer.h.37.mlp.down_proj.bias"}> : () -> tensor<5120xf16>
    %5057 = "mix.prim.add"(%5055, %5056) : (tensor<1x40x5120xf16>, tensor<5120xf16>) -> tensor<1x40x5120xf16>
    %5058 = "mix.prim.add"(%5057, %5035) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %5059 = "mix.comp.weight"() <{param_loc = "transformer.ln_f.weight"}> : () -> tensor<5120xf16>
    %5060 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %5061 = "mix.prim.pow"(%5058, %5060) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %5062 = "mix.comp.mean"(%5061) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %5063 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %5064 = "mix.prim.add"(%5062, %5063) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %5065 = "mix.prim.rsqrt"(%5064) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %5066 = "mix.prim.mul"(%5058, %5065) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %5067 = "mix.prim.mul"(%5059, %5066) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %5068 = "mix.comp.weight"() <{param_loc = "lm_head.weight.weight"}> : () -> tensor<5120x120000xf16>
    %5069 = "mix.prim.reshape"(%5068) <{shape = [1, 5120, 120000]}> : (tensor<5120x120000xf16>) -> tensor<1x5120x120000xf16>
    %5070 = "mix.prim.batch_matmul"(%5067, %5069) : (tensor<1x40x5120xf16>, tensor<1x5120x120000xf16>) -> tensor<1x40x120000xf16>
    return %5070 : tensor<1x40x120000xf16>
  }
}
