module {
  func.func private @printMemrefF16(tensor<*xf16>)
  func.func private @Telechat(%arg0: tensor<1x40xi32>) -> tensor<1x40x120000xf16> {
    %0 = "mix.module.embedding"(%arg0) <{dtype = f16, embedding_dim = 5120 : i32, num_embeddings = 120000 : i32, params_loc = "transformer.word_embeddings"}> : (tensor<1x40xi32>) -> tensor<1x40x5120xf16>
    %1 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]> : tensor<40xi16>}> : () -> tensor<40xi16>
    %2 = "mix.prim.slice"(%1) <{dim = 0 : i64, end = 9223372036854775807 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40xi16>) -> tensor<40xi16>
    %3 = "mix.prim.unsqueeze"(%2) <{axis = 1 : i32}> : (tensor<40xi16>) -> tensor<40x1xi16>
    %4 = "mix.prim.unsqueeze"(%1) <{axis = 0 : i32}> : (tensor<40xi16>) -> tensor<1x40xi16>
    %5 = "mix.prim.slice"(%4) <{dim = 1 : i64, end = 9223372036854775807 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<1x40xi16>) -> tensor<1x40xi16>
    %6 = "mix.prim.lt"(%3, %5) : (tensor<40x1xi16>, tensor<1x40xi16>) -> tensor<40x40xi1>
    %7 = "mix.prim.constant"() <{value = dense<true> : tensor<40x0xi1>}> : () -> tensor<40x0xi1>
    %8 = "mix.prim.constant"() <{value = dense<true> : tensor<0x40xi1>}> : () -> tensor<0x40xi1>
    %9 = "mix.prim.constant"() <{value = dense<true> : tensor<0x0xi1>}> : () -> tensor<0x0xi1>
    %10 = "mix.prim.concat"(%6, %7) <{axis = 1 : i64}> : (tensor<40x40xi1>, tensor<40x0xi1>) -> tensor<40x40xi1>
    %11 = "mix.prim.concat"(%8, %9) <{axis = 1 : i64}> : (tensor<0x40xi1>, tensor<0x0xi1>) -> tensor<0x40xi1>
    %12 = "mix.prim.concat"(%10, %11) <{axis = 0 : i64}> : (tensor<40x40xi1>, tensor<0x40xi1>) -> tensor<40x40xi1>
    %13 = "mix.prim.unsqueeze"(%12) <{axis = 0 : i32}> : (tensor<40x40xi1>) -> tensor<1x40x40xi1>
    %14 = "mix.prim.unsqueeze"(%13) <{axis = 0 : i32}> : (tensor<1x40x40xi1>) -> tensor<1x1x40x40xi1>
    %15 = "mix.comp.weight"() <{param_loc = "transformer.h.0.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %16 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %17 = "mix.prim.pow"(%0, %16) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %18 = "mix.comp.mean"(%17) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %19 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %20 = "mix.prim.add"(%18, %19) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %21 = "mix.prim.rsqrt"(%20) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %22 = "mix.prim.mul"(%0, %21) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %23 = "mix.prim.mul"(%15, %22) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %24 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %25 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %26 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %27 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %28 = "mix.prim.transpose"(%23) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %29 = "mix.prim.transpose"(%24) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %30 = "mix.prim.reshape"(%28) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %31 = "mix.prim.matmul"(%30, %29) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %32 = "mix.prim.reshape"(%31) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %33 = "mix.prim.reshape"(%32) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %34 = "mix.prim.transpose"(%25) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %35 = "mix.prim.reshape"(%28) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %36 = "mix.prim.matmul"(%35, %34) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %37 = "mix.prim.reshape"(%36) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %38 = "mix.prim.reshape"(%37) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %39 = "mix.prim.slice"(%38) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %40 = "mix.prim.slice"(%38) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %41 = "mix.prim.reshape"(%33) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %42 = "mix.prim.reshape"(%39) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %43 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %44 = "mix.prim.convert"(%43) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %45 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %46 = "mix.prim.div"(%44, %45) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %47 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %48 = "mix.prim.pow"(%47, %46) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %49 = "mix.prim.reciprocal"(%48) : (tensor<80xf16>) -> tensor<80xf16>
    %50 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %51 = "mix.prim.mul"(%50, %49) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %52 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %53 = "mix.prim.unsqueeze"(%52) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %54 = "mix.prim.permute"(%53) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %55 = "mix.prim.unsqueeze"(%51) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %56 = "mix.prim.permute"(%55) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %57 = "mix.prim.mul"(%54, %56) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %58 = "mix.prim.concat"(%57, %57) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %59 = "mix.prim.cos"(%58) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %60 = "mix.prim.slice"(%59) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %61 = "mix.prim.unsqueeze"(%60) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %62 = "mix.prim.slice"(%61) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %63 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %64 = "mix.prim.mul"(%62, %63) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %65 = "mix.prim.sin"(%58) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %66 = "mix.prim.slice"(%65) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %67 = "mix.prim.unsqueeze"(%66) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %68 = "mix.prim.slice"(%67) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %69 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %70 = "mix.prim.mul"(%68, %69) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %71 = "mix.prim.slice"(%64) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %72 = "mix.prim.slice"(%70) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %73 = "mix.prim.mul"(%41, %71) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %74 = "mix.prim.slice"(%41) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %75 = "mix.prim.slice"(%41) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %76 = "mix.prim.neg"(%75) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %77 = "mix.prim.concat"(%76, %74) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %78 = "mix.prim.mul"(%77, %72) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %79 = "mix.prim.add"(%73, %78) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %80 = "mix.prim.mul"(%42, %71) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %81 = "mix.prim.slice"(%42) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %82 = "mix.prim.slice"(%42) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %83 = "mix.prim.neg"(%82) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %84 = "mix.prim.concat"(%83, %81) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %85 = "mix.prim.mul"(%84, %72) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %86 = "mix.prim.add"(%80, %85) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %87 = "mix.prim.reshape"(%79) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %88 = "mix.prim.reshape"(%86) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %89 = "mix.prim.reshape"(%87) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %90 = "mix.prim.reshape"(%88) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %91 = "mix.prim.transpose"(%89) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %92 = "mix.prim.transpose"(%90) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %93 = "mix.prim.transpose"(%92) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %94 = "mix.prim.unsqueeze"(%91) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %95 = "mix.prim.permute"(%94) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %96 = "mix.prim.unsqueeze"(%93) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %97 = "mix.prim.permute"(%96) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %98 = "mix.prim.permute"(%95) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %99 = "mix.prim.reshape"(%98) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %100 = "mix.prim.permute"(%97) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %101 = "mix.prim.reshape"(%100) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %102 = "mix.prim.batch_matmul"(%99, %101) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %103 = "mix.prim.reshape"(%102) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %104 = "mix.prim.permute"(%103) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %105 = "mix.prim.reshape"(%104) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %106 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %107 = "mix.prim.mul"(%105, %106) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %108 = "mix.prim.reshape"(%107) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %109 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %110 = "mix.comp.masked_fill"(%108, %14, %109) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %111 = "mix.comp.softmax"(%110) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %112 = "mix.prim.reshape"(%111) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %113 = "mix.prim.reshape"(%40) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %114 = "mix.prim.transpose"(%113) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %115 = "mix.prim.batch_matmul"(%112, %114) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %116 = "mix.prim.reshape"(%115) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %117 = "mix.prim.permute"(%116) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %118 = "mix.prim.reshape"(%117) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %119 = "mix.prim.reshape"(%118) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %120 = "mix.prim.transpose"(%26) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %121 = "mix.prim.matmul"(%119, %120) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %122 = "mix.prim.add"(%121, %27) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %123 = "mix.prim.reshape"(%122) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %124 = "mix.prim.mul"(%0, %123) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %125 = "mix.comp.weight"() <{param_loc = "transformer.h.0.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %126 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %127 = "mix.prim.pow"(%124, %126) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %128 = "mix.comp.mean"(%127) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %129 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %130 = "mix.prim.add"(%128, %129) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %131 = "mix.prim.rsqrt"(%130) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %132 = "mix.prim.mul"(%124, %131) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %133 = "mix.prim.mul"(%125, %132) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %134 = "mix.module.linear"(%133) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.0.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %135 = "mix.comp.silu"(%134) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %136 = "mix.module.linear"(%133) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.0.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %137 = "mix.prim.mul"(%135, %136) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %138 = "mix.module.linear"(%137) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.0.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %139 = "mix.prim.add"(%138, %124) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %140 = "mix.comp.weight"() <{param_loc = "transformer.h.1.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %141 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %142 = "mix.prim.pow"(%139, %141) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %143 = "mix.comp.mean"(%142) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %144 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %145 = "mix.prim.add"(%143, %144) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %146 = "mix.prim.rsqrt"(%145) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %147 = "mix.prim.mul"(%139, %146) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %148 = "mix.prim.mul"(%140, %147) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %149 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %150 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %151 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %152 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %153 = "mix.prim.transpose"(%148) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %154 = "mix.prim.transpose"(%149) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %155 = "mix.prim.reshape"(%153) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %156 = "mix.prim.matmul"(%155, %154) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %157 = "mix.prim.reshape"(%156) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %158 = "mix.prim.reshape"(%157) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %159 = "mix.prim.transpose"(%150) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %160 = "mix.prim.reshape"(%153) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %161 = "mix.prim.matmul"(%160, %159) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %162 = "mix.prim.reshape"(%161) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %163 = "mix.prim.reshape"(%162) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %164 = "mix.prim.slice"(%163) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %165 = "mix.prim.slice"(%163) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %166 = "mix.prim.reshape"(%158) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %167 = "mix.prim.reshape"(%164) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %168 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %169 = "mix.prim.convert"(%168) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %170 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %171 = "mix.prim.div"(%169, %170) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %172 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %173 = "mix.prim.pow"(%172, %171) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %174 = "mix.prim.reciprocal"(%173) : (tensor<80xf16>) -> tensor<80xf16>
    %175 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %176 = "mix.prim.mul"(%175, %174) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %177 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %178 = "mix.prim.unsqueeze"(%177) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %179 = "mix.prim.permute"(%178) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %180 = "mix.prim.unsqueeze"(%176) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %181 = "mix.prim.permute"(%180) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %182 = "mix.prim.mul"(%179, %181) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %183 = "mix.prim.concat"(%182, %182) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %184 = "mix.prim.cos"(%183) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %185 = "mix.prim.slice"(%184) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %186 = "mix.prim.unsqueeze"(%185) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %187 = "mix.prim.slice"(%186) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %188 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %189 = "mix.prim.mul"(%187, %188) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %190 = "mix.prim.sin"(%183) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %191 = "mix.prim.slice"(%190) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %192 = "mix.prim.unsqueeze"(%191) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %193 = "mix.prim.slice"(%192) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %194 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %195 = "mix.prim.mul"(%193, %194) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %196 = "mix.prim.slice"(%189) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %197 = "mix.prim.slice"(%195) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %198 = "mix.prim.mul"(%166, %196) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %199 = "mix.prim.slice"(%166) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %200 = "mix.prim.slice"(%166) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %201 = "mix.prim.neg"(%200) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %202 = "mix.prim.concat"(%201, %199) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %203 = "mix.prim.mul"(%202, %197) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %204 = "mix.prim.add"(%198, %203) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %205 = "mix.prim.mul"(%167, %196) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %206 = "mix.prim.slice"(%167) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %207 = "mix.prim.slice"(%167) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %208 = "mix.prim.neg"(%207) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %209 = "mix.prim.concat"(%208, %206) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %210 = "mix.prim.mul"(%209, %197) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %211 = "mix.prim.add"(%205, %210) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %212 = "mix.prim.reshape"(%204) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %213 = "mix.prim.reshape"(%211) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %214 = "mix.prim.reshape"(%212) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %215 = "mix.prim.reshape"(%213) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %216 = "mix.prim.transpose"(%214) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %217 = "mix.prim.transpose"(%215) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %218 = "mix.prim.transpose"(%217) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %219 = "mix.prim.unsqueeze"(%216) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %220 = "mix.prim.permute"(%219) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %221 = "mix.prim.unsqueeze"(%218) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %222 = "mix.prim.permute"(%221) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %223 = "mix.prim.permute"(%220) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %224 = "mix.prim.reshape"(%223) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %225 = "mix.prim.permute"(%222) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %226 = "mix.prim.reshape"(%225) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %227 = "mix.prim.batch_matmul"(%224, %226) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %228 = "mix.prim.reshape"(%227) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %229 = "mix.prim.permute"(%228) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %230 = "mix.prim.reshape"(%229) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %231 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %232 = "mix.prim.mul"(%230, %231) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %233 = "mix.prim.reshape"(%232) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %234 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %235 = "mix.comp.masked_fill"(%233, %14, %234) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %236 = "mix.comp.softmax"(%235) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %237 = "mix.prim.reshape"(%236) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %238 = "mix.prim.reshape"(%165) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %239 = "mix.prim.transpose"(%238) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %240 = "mix.prim.batch_matmul"(%237, %239) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %241 = "mix.prim.reshape"(%240) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %242 = "mix.prim.permute"(%241) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %243 = "mix.prim.reshape"(%242) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %244 = "mix.prim.reshape"(%243) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %245 = "mix.prim.transpose"(%151) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %246 = "mix.prim.matmul"(%244, %245) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %247 = "mix.prim.add"(%246, %152) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %248 = "mix.prim.reshape"(%247) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %249 = "mix.prim.mul"(%139, %248) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %250 = "mix.comp.weight"() <{param_loc = "transformer.h.1.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %251 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %252 = "mix.prim.pow"(%249, %251) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %253 = "mix.comp.mean"(%252) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %254 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %255 = "mix.prim.add"(%253, %254) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %256 = "mix.prim.rsqrt"(%255) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %257 = "mix.prim.mul"(%249, %256) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %258 = "mix.prim.mul"(%250, %257) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %259 = "mix.module.linear"(%258) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.1.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %260 = "mix.comp.silu"(%259) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %261 = "mix.module.linear"(%258) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.1.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %262 = "mix.prim.mul"(%260, %261) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %263 = "mix.module.linear"(%262) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.1.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %264 = "mix.prim.add"(%263, %249) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %265 = "mix.comp.weight"() <{param_loc = "transformer.h.2.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %266 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %267 = "mix.prim.pow"(%264, %266) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %268 = "mix.comp.mean"(%267) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %269 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %270 = "mix.prim.add"(%268, %269) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %271 = "mix.prim.rsqrt"(%270) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %272 = "mix.prim.mul"(%264, %271) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %273 = "mix.prim.mul"(%265, %272) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %274 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %275 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %276 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %277 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %278 = "mix.prim.transpose"(%273) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %279 = "mix.prim.transpose"(%274) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %280 = "mix.prim.reshape"(%278) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %281 = "mix.prim.matmul"(%280, %279) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %282 = "mix.prim.reshape"(%281) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %283 = "mix.prim.reshape"(%282) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %284 = "mix.prim.transpose"(%275) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %285 = "mix.prim.reshape"(%278) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %286 = "mix.prim.matmul"(%285, %284) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %287 = "mix.prim.reshape"(%286) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %288 = "mix.prim.reshape"(%287) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %289 = "mix.prim.slice"(%288) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %290 = "mix.prim.slice"(%288) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %291 = "mix.prim.reshape"(%283) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %292 = "mix.prim.reshape"(%289) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %293 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %294 = "mix.prim.convert"(%293) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %295 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %296 = "mix.prim.div"(%294, %295) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %297 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %298 = "mix.prim.pow"(%297, %296) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %299 = "mix.prim.reciprocal"(%298) : (tensor<80xf16>) -> tensor<80xf16>
    %300 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %301 = "mix.prim.mul"(%300, %299) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %302 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %303 = "mix.prim.unsqueeze"(%302) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %304 = "mix.prim.permute"(%303) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %305 = "mix.prim.unsqueeze"(%301) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %306 = "mix.prim.permute"(%305) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %307 = "mix.prim.mul"(%304, %306) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %308 = "mix.prim.concat"(%307, %307) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %309 = "mix.prim.cos"(%308) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %310 = "mix.prim.slice"(%309) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %311 = "mix.prim.unsqueeze"(%310) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %312 = "mix.prim.slice"(%311) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %313 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %314 = "mix.prim.mul"(%312, %313) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %315 = "mix.prim.sin"(%308) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %316 = "mix.prim.slice"(%315) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %317 = "mix.prim.unsqueeze"(%316) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %318 = "mix.prim.slice"(%317) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %319 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %320 = "mix.prim.mul"(%318, %319) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %321 = "mix.prim.slice"(%314) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %322 = "mix.prim.slice"(%320) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %323 = "mix.prim.mul"(%291, %321) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %324 = "mix.prim.slice"(%291) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %325 = "mix.prim.slice"(%291) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %326 = "mix.prim.neg"(%325) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %327 = "mix.prim.concat"(%326, %324) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %328 = "mix.prim.mul"(%327, %322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %329 = "mix.prim.add"(%323, %328) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %330 = "mix.prim.mul"(%292, %321) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %331 = "mix.prim.slice"(%292) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %332 = "mix.prim.slice"(%292) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %333 = "mix.prim.neg"(%332) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %334 = "mix.prim.concat"(%333, %331) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %335 = "mix.prim.mul"(%334, %322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %336 = "mix.prim.add"(%330, %335) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %337 = "mix.prim.reshape"(%329) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %338 = "mix.prim.reshape"(%336) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %339 = "mix.prim.reshape"(%337) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %340 = "mix.prim.reshape"(%338) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %341 = "mix.prim.transpose"(%339) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %342 = "mix.prim.transpose"(%340) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %343 = "mix.prim.transpose"(%342) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %344 = "mix.prim.unsqueeze"(%341) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %345 = "mix.prim.permute"(%344) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %346 = "mix.prim.unsqueeze"(%343) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %347 = "mix.prim.permute"(%346) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %348 = "mix.prim.permute"(%345) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %349 = "mix.prim.reshape"(%348) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %350 = "mix.prim.permute"(%347) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %351 = "mix.prim.reshape"(%350) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %352 = "mix.prim.batch_matmul"(%349, %351) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %353 = "mix.prim.reshape"(%352) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %354 = "mix.prim.permute"(%353) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %355 = "mix.prim.reshape"(%354) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %356 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %357 = "mix.prim.mul"(%355, %356) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %358 = "mix.prim.reshape"(%357) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %359 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %360 = "mix.comp.masked_fill"(%358, %14, %359) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %361 = "mix.comp.softmax"(%360) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %362 = "mix.prim.reshape"(%361) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %363 = "mix.prim.reshape"(%290) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %364 = "mix.prim.transpose"(%363) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %365 = "mix.prim.batch_matmul"(%362, %364) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %366 = "mix.prim.reshape"(%365) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %367 = "mix.prim.permute"(%366) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %368 = "mix.prim.reshape"(%367) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %369 = "mix.prim.reshape"(%368) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %370 = "mix.prim.transpose"(%276) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %371 = "mix.prim.matmul"(%369, %370) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %372 = "mix.prim.add"(%371, %277) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %373 = "mix.prim.reshape"(%372) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %374 = "mix.prim.mul"(%264, %373) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %375 = "mix.comp.weight"() <{param_loc = "transformer.h.2.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %376 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %377 = "mix.prim.pow"(%374, %376) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %378 = "mix.comp.mean"(%377) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %379 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %380 = "mix.prim.add"(%378, %379) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %381 = "mix.prim.rsqrt"(%380) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %382 = "mix.prim.mul"(%374, %381) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %383 = "mix.prim.mul"(%375, %382) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %384 = "mix.module.linear"(%383) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.2.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %385 = "mix.comp.silu"(%384) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %386 = "mix.module.linear"(%383) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.2.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %387 = "mix.prim.mul"(%385, %386) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %388 = "mix.module.linear"(%387) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.2.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %389 = "mix.prim.add"(%388, %374) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %390 = "mix.comp.weight"() <{param_loc = "transformer.h.3.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %391 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %392 = "mix.prim.pow"(%389, %391) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %393 = "mix.comp.mean"(%392) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %394 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %395 = "mix.prim.add"(%393, %394) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %396 = "mix.prim.rsqrt"(%395) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %397 = "mix.prim.mul"(%389, %396) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %398 = "mix.prim.mul"(%390, %397) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %399 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %400 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %401 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %402 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %403 = "mix.prim.transpose"(%398) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %404 = "mix.prim.transpose"(%399) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %405 = "mix.prim.reshape"(%403) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %406 = "mix.prim.matmul"(%405, %404) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %407 = "mix.prim.reshape"(%406) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %408 = "mix.prim.reshape"(%407) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %409 = "mix.prim.transpose"(%400) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %410 = "mix.prim.reshape"(%403) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %411 = "mix.prim.matmul"(%410, %409) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %412 = "mix.prim.reshape"(%411) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %413 = "mix.prim.reshape"(%412) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %414 = "mix.prim.slice"(%413) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %415 = "mix.prim.slice"(%413) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %416 = "mix.prim.reshape"(%408) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %417 = "mix.prim.reshape"(%414) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %418 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %419 = "mix.prim.convert"(%418) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %420 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %421 = "mix.prim.div"(%419, %420) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %422 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %423 = "mix.prim.pow"(%422, %421) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %424 = "mix.prim.reciprocal"(%423) : (tensor<80xf16>) -> tensor<80xf16>
    %425 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %426 = "mix.prim.mul"(%425, %424) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %427 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %428 = "mix.prim.unsqueeze"(%427) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %429 = "mix.prim.permute"(%428) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %430 = "mix.prim.unsqueeze"(%426) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %431 = "mix.prim.permute"(%430) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %432 = "mix.prim.mul"(%429, %431) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %433 = "mix.prim.concat"(%432, %432) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %434 = "mix.prim.cos"(%433) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %435 = "mix.prim.slice"(%434) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %436 = "mix.prim.unsqueeze"(%435) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %437 = "mix.prim.slice"(%436) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %438 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %439 = "mix.prim.mul"(%437, %438) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %440 = "mix.prim.sin"(%433) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %441 = "mix.prim.slice"(%440) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %442 = "mix.prim.unsqueeze"(%441) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %443 = "mix.prim.slice"(%442) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %444 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %445 = "mix.prim.mul"(%443, %444) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %446 = "mix.prim.slice"(%439) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %447 = "mix.prim.slice"(%445) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %448 = "mix.prim.mul"(%416, %446) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %449 = "mix.prim.slice"(%416) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %450 = "mix.prim.slice"(%416) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %451 = "mix.prim.neg"(%450) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %452 = "mix.prim.concat"(%451, %449) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %453 = "mix.prim.mul"(%452, %447) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %454 = "mix.prim.add"(%448, %453) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %455 = "mix.prim.mul"(%417, %446) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %456 = "mix.prim.slice"(%417) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %457 = "mix.prim.slice"(%417) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %458 = "mix.prim.neg"(%457) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %459 = "mix.prim.concat"(%458, %456) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %460 = "mix.prim.mul"(%459, %447) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %461 = "mix.prim.add"(%455, %460) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %462 = "mix.prim.reshape"(%454) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %463 = "mix.prim.reshape"(%461) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %464 = "mix.prim.reshape"(%462) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %465 = "mix.prim.reshape"(%463) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %466 = "mix.prim.transpose"(%464) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %467 = "mix.prim.transpose"(%465) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %468 = "mix.prim.transpose"(%467) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %469 = "mix.prim.unsqueeze"(%466) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %470 = "mix.prim.permute"(%469) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %471 = "mix.prim.unsqueeze"(%468) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %472 = "mix.prim.permute"(%471) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %473 = "mix.prim.permute"(%470) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %474 = "mix.prim.reshape"(%473) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %475 = "mix.prim.permute"(%472) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %476 = "mix.prim.reshape"(%475) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %477 = "mix.prim.batch_matmul"(%474, %476) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %478 = "mix.prim.reshape"(%477) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %479 = "mix.prim.permute"(%478) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %480 = "mix.prim.reshape"(%479) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %481 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %482 = "mix.prim.mul"(%480, %481) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %483 = "mix.prim.reshape"(%482) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %484 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %485 = "mix.comp.masked_fill"(%483, %14, %484) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %486 = "mix.comp.softmax"(%485) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %487 = "mix.prim.reshape"(%486) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %488 = "mix.prim.reshape"(%415) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %489 = "mix.prim.transpose"(%488) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %490 = "mix.prim.batch_matmul"(%487, %489) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %491 = "mix.prim.reshape"(%490) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %492 = "mix.prim.permute"(%491) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %493 = "mix.prim.reshape"(%492) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %494 = "mix.prim.reshape"(%493) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %495 = "mix.prim.transpose"(%401) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %496 = "mix.prim.matmul"(%494, %495) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %497 = "mix.prim.add"(%496, %402) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %498 = "mix.prim.reshape"(%497) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %499 = "mix.prim.mul"(%389, %498) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %500 = "mix.comp.weight"() <{param_loc = "transformer.h.3.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %501 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %502 = "mix.prim.pow"(%499, %501) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %503 = "mix.comp.mean"(%502) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %504 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %505 = "mix.prim.add"(%503, %504) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %506 = "mix.prim.rsqrt"(%505) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %507 = "mix.prim.mul"(%499, %506) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %508 = "mix.prim.mul"(%500, %507) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %509 = "mix.module.linear"(%508) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.3.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %510 = "mix.comp.silu"(%509) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %511 = "mix.module.linear"(%508) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.3.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %512 = "mix.prim.mul"(%510, %511) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %513 = "mix.module.linear"(%512) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.3.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %514 = "mix.prim.add"(%513, %499) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %515 = "mix.comp.weight"() <{param_loc = "transformer.h.4.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %516 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %517 = "mix.prim.pow"(%514, %516) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %518 = "mix.comp.mean"(%517) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %519 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %520 = "mix.prim.add"(%518, %519) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %521 = "mix.prim.rsqrt"(%520) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %522 = "mix.prim.mul"(%514, %521) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %523 = "mix.prim.mul"(%515, %522) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %524 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %525 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %526 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %527 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %528 = "mix.prim.transpose"(%523) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %529 = "mix.prim.transpose"(%524) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %530 = "mix.prim.reshape"(%528) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %531 = "mix.prim.matmul"(%530, %529) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %532 = "mix.prim.reshape"(%531) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %533 = "mix.prim.reshape"(%532) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %534 = "mix.prim.transpose"(%525) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %535 = "mix.prim.reshape"(%528) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %536 = "mix.prim.matmul"(%535, %534) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %537 = "mix.prim.reshape"(%536) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %538 = "mix.prim.reshape"(%537) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %539 = "mix.prim.slice"(%538) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %540 = "mix.prim.slice"(%538) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %541 = "mix.prim.reshape"(%533) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %542 = "mix.prim.reshape"(%539) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %543 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %544 = "mix.prim.convert"(%543) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %545 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %546 = "mix.prim.div"(%544, %545) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %547 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %548 = "mix.prim.pow"(%547, %546) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %549 = "mix.prim.reciprocal"(%548) : (tensor<80xf16>) -> tensor<80xf16>
    %550 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %551 = "mix.prim.mul"(%550, %549) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %552 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %553 = "mix.prim.unsqueeze"(%552) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %554 = "mix.prim.permute"(%553) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %555 = "mix.prim.unsqueeze"(%551) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %556 = "mix.prim.permute"(%555) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %557 = "mix.prim.mul"(%554, %556) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %558 = "mix.prim.concat"(%557, %557) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %559 = "mix.prim.cos"(%558) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %560 = "mix.prim.slice"(%559) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %561 = "mix.prim.unsqueeze"(%560) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %562 = "mix.prim.slice"(%561) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %563 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %564 = "mix.prim.mul"(%562, %563) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %565 = "mix.prim.sin"(%558) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %566 = "mix.prim.slice"(%565) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %567 = "mix.prim.unsqueeze"(%566) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %568 = "mix.prim.slice"(%567) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %569 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %570 = "mix.prim.mul"(%568, %569) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %571 = "mix.prim.slice"(%564) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %572 = "mix.prim.slice"(%570) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %573 = "mix.prim.mul"(%541, %571) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %574 = "mix.prim.slice"(%541) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %575 = "mix.prim.slice"(%541) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %576 = "mix.prim.neg"(%575) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %577 = "mix.prim.concat"(%576, %574) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %578 = "mix.prim.mul"(%577, %572) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %579 = "mix.prim.add"(%573, %578) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %580 = "mix.prim.mul"(%542, %571) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %581 = "mix.prim.slice"(%542) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %582 = "mix.prim.slice"(%542) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %583 = "mix.prim.neg"(%582) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %584 = "mix.prim.concat"(%583, %581) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %585 = "mix.prim.mul"(%584, %572) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %586 = "mix.prim.add"(%580, %585) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %587 = "mix.prim.reshape"(%579) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %588 = "mix.prim.reshape"(%586) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %589 = "mix.prim.reshape"(%587) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %590 = "mix.prim.reshape"(%588) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %591 = "mix.prim.transpose"(%589) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %592 = "mix.prim.transpose"(%590) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %593 = "mix.prim.transpose"(%592) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %594 = "mix.prim.unsqueeze"(%591) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %595 = "mix.prim.permute"(%594) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %596 = "mix.prim.unsqueeze"(%593) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %597 = "mix.prim.permute"(%596) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %598 = "mix.prim.permute"(%595) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %599 = "mix.prim.reshape"(%598) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %600 = "mix.prim.permute"(%597) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %601 = "mix.prim.reshape"(%600) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %602 = "mix.prim.batch_matmul"(%599, %601) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %603 = "mix.prim.reshape"(%602) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %604 = "mix.prim.permute"(%603) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %605 = "mix.prim.reshape"(%604) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %606 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %607 = "mix.prim.mul"(%605, %606) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %608 = "mix.prim.reshape"(%607) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %609 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %610 = "mix.comp.masked_fill"(%608, %14, %609) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %611 = "mix.comp.softmax"(%610) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %612 = "mix.prim.reshape"(%611) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %613 = "mix.prim.reshape"(%540) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %614 = "mix.prim.transpose"(%613) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %615 = "mix.prim.batch_matmul"(%612, %614) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %616 = "mix.prim.reshape"(%615) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %617 = "mix.prim.permute"(%616) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %618 = "mix.prim.reshape"(%617) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %619 = "mix.prim.reshape"(%618) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %620 = "mix.prim.transpose"(%526) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %621 = "mix.prim.matmul"(%619, %620) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %622 = "mix.prim.add"(%621, %527) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %623 = "mix.prim.reshape"(%622) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %624 = "mix.prim.mul"(%514, %623) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %625 = "mix.comp.weight"() <{param_loc = "transformer.h.4.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %626 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %627 = "mix.prim.pow"(%624, %626) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %628 = "mix.comp.mean"(%627) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %629 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %630 = "mix.prim.add"(%628, %629) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %631 = "mix.prim.rsqrt"(%630) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %632 = "mix.prim.mul"(%624, %631) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %633 = "mix.prim.mul"(%625, %632) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %634 = "mix.module.linear"(%633) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.4.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %635 = "mix.comp.silu"(%634) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %636 = "mix.module.linear"(%633) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.4.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %637 = "mix.prim.mul"(%635, %636) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %638 = "mix.module.linear"(%637) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.4.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %639 = "mix.prim.add"(%638, %624) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %640 = "mix.comp.weight"() <{param_loc = "transformer.h.5.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %641 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %642 = "mix.prim.pow"(%639, %641) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %643 = "mix.comp.mean"(%642) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %644 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %645 = "mix.prim.add"(%643, %644) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %646 = "mix.prim.rsqrt"(%645) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %647 = "mix.prim.mul"(%639, %646) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %648 = "mix.prim.mul"(%640, %647) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %649 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %650 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %651 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %652 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %653 = "mix.prim.transpose"(%648) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %654 = "mix.prim.transpose"(%649) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %655 = "mix.prim.reshape"(%653) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %656 = "mix.prim.matmul"(%655, %654) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %657 = "mix.prim.reshape"(%656) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %658 = "mix.prim.reshape"(%657) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %659 = "mix.prim.transpose"(%650) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %660 = "mix.prim.reshape"(%653) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %661 = "mix.prim.matmul"(%660, %659) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %662 = "mix.prim.reshape"(%661) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %663 = "mix.prim.reshape"(%662) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %664 = "mix.prim.slice"(%663) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %665 = "mix.prim.slice"(%663) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %666 = "mix.prim.reshape"(%658) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %667 = "mix.prim.reshape"(%664) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %668 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %669 = "mix.prim.convert"(%668) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %670 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %671 = "mix.prim.div"(%669, %670) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %672 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %673 = "mix.prim.pow"(%672, %671) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %674 = "mix.prim.reciprocal"(%673) : (tensor<80xf16>) -> tensor<80xf16>
    %675 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %676 = "mix.prim.mul"(%675, %674) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %677 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %678 = "mix.prim.unsqueeze"(%677) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %679 = "mix.prim.permute"(%678) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %680 = "mix.prim.unsqueeze"(%676) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %681 = "mix.prim.permute"(%680) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %682 = "mix.prim.mul"(%679, %681) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %683 = "mix.prim.concat"(%682, %682) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %684 = "mix.prim.cos"(%683) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %685 = "mix.prim.slice"(%684) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %686 = "mix.prim.unsqueeze"(%685) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %687 = "mix.prim.slice"(%686) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %688 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %689 = "mix.prim.mul"(%687, %688) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %690 = "mix.prim.sin"(%683) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %691 = "mix.prim.slice"(%690) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %692 = "mix.prim.unsqueeze"(%691) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %693 = "mix.prim.slice"(%692) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %694 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %695 = "mix.prim.mul"(%693, %694) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %696 = "mix.prim.slice"(%689) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %697 = "mix.prim.slice"(%695) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %698 = "mix.prim.mul"(%666, %696) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %699 = "mix.prim.slice"(%666) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %700 = "mix.prim.slice"(%666) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %701 = "mix.prim.neg"(%700) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %702 = "mix.prim.concat"(%701, %699) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %703 = "mix.prim.mul"(%702, %697) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %704 = "mix.prim.add"(%698, %703) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %705 = "mix.prim.mul"(%667, %696) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %706 = "mix.prim.slice"(%667) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %707 = "mix.prim.slice"(%667) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %708 = "mix.prim.neg"(%707) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %709 = "mix.prim.concat"(%708, %706) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %710 = "mix.prim.mul"(%709, %697) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %711 = "mix.prim.add"(%705, %710) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %712 = "mix.prim.reshape"(%704) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %713 = "mix.prim.reshape"(%711) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %714 = "mix.prim.reshape"(%712) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %715 = "mix.prim.reshape"(%713) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %716 = "mix.prim.transpose"(%714) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %717 = "mix.prim.transpose"(%715) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %718 = "mix.prim.transpose"(%717) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %719 = "mix.prim.unsqueeze"(%716) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %720 = "mix.prim.permute"(%719) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %721 = "mix.prim.unsqueeze"(%718) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %722 = "mix.prim.permute"(%721) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %723 = "mix.prim.permute"(%720) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %724 = "mix.prim.reshape"(%723) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %725 = "mix.prim.permute"(%722) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %726 = "mix.prim.reshape"(%725) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %727 = "mix.prim.batch_matmul"(%724, %726) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %728 = "mix.prim.reshape"(%727) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %729 = "mix.prim.permute"(%728) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %730 = "mix.prim.reshape"(%729) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %731 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %732 = "mix.prim.mul"(%730, %731) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %733 = "mix.prim.reshape"(%732) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %734 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %735 = "mix.comp.masked_fill"(%733, %14, %734) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %736 = "mix.comp.softmax"(%735) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %737 = "mix.prim.reshape"(%736) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %738 = "mix.prim.reshape"(%665) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %739 = "mix.prim.transpose"(%738) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %740 = "mix.prim.batch_matmul"(%737, %739) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %741 = "mix.prim.reshape"(%740) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %742 = "mix.prim.permute"(%741) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %743 = "mix.prim.reshape"(%742) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %744 = "mix.prim.reshape"(%743) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %745 = "mix.prim.transpose"(%651) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %746 = "mix.prim.matmul"(%744, %745) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %747 = "mix.prim.add"(%746, %652) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %748 = "mix.prim.reshape"(%747) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %749 = "mix.prim.mul"(%639, %748) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %750 = "mix.comp.weight"() <{param_loc = "transformer.h.5.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %751 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %752 = "mix.prim.pow"(%749, %751) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %753 = "mix.comp.mean"(%752) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %754 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %755 = "mix.prim.add"(%753, %754) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %756 = "mix.prim.rsqrt"(%755) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %757 = "mix.prim.mul"(%749, %756) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %758 = "mix.prim.mul"(%750, %757) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %759 = "mix.module.linear"(%758) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.5.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %760 = "mix.comp.silu"(%759) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %761 = "mix.module.linear"(%758) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.5.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %762 = "mix.prim.mul"(%760, %761) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %763 = "mix.module.linear"(%762) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.5.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %764 = "mix.prim.add"(%763, %749) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %765 = "mix.comp.weight"() <{param_loc = "transformer.h.6.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %766 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %767 = "mix.prim.pow"(%764, %766) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %768 = "mix.comp.mean"(%767) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %769 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %770 = "mix.prim.add"(%768, %769) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %771 = "mix.prim.rsqrt"(%770) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %772 = "mix.prim.mul"(%764, %771) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %773 = "mix.prim.mul"(%765, %772) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %774 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %775 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %776 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %777 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %778 = "mix.prim.transpose"(%773) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %779 = "mix.prim.transpose"(%774) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %780 = "mix.prim.reshape"(%778) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %781 = "mix.prim.matmul"(%780, %779) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %782 = "mix.prim.reshape"(%781) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %783 = "mix.prim.reshape"(%782) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %784 = "mix.prim.transpose"(%775) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %785 = "mix.prim.reshape"(%778) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %786 = "mix.prim.matmul"(%785, %784) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %787 = "mix.prim.reshape"(%786) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %788 = "mix.prim.reshape"(%787) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %789 = "mix.prim.slice"(%788) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %790 = "mix.prim.slice"(%788) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %791 = "mix.prim.reshape"(%783) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %792 = "mix.prim.reshape"(%789) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %793 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %794 = "mix.prim.convert"(%793) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %795 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %796 = "mix.prim.div"(%794, %795) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %797 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %798 = "mix.prim.pow"(%797, %796) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %799 = "mix.prim.reciprocal"(%798) : (tensor<80xf16>) -> tensor<80xf16>
    %800 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %801 = "mix.prim.mul"(%800, %799) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %802 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %803 = "mix.prim.unsqueeze"(%802) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %804 = "mix.prim.permute"(%803) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %805 = "mix.prim.unsqueeze"(%801) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %806 = "mix.prim.permute"(%805) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %807 = "mix.prim.mul"(%804, %806) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %808 = "mix.prim.concat"(%807, %807) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %809 = "mix.prim.cos"(%808) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %810 = "mix.prim.slice"(%809) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %811 = "mix.prim.unsqueeze"(%810) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %812 = "mix.prim.slice"(%811) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %813 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %814 = "mix.prim.mul"(%812, %813) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %815 = "mix.prim.sin"(%808) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %816 = "mix.prim.slice"(%815) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %817 = "mix.prim.unsqueeze"(%816) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %818 = "mix.prim.slice"(%817) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %819 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %820 = "mix.prim.mul"(%818, %819) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %821 = "mix.prim.slice"(%814) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %822 = "mix.prim.slice"(%820) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %823 = "mix.prim.mul"(%791, %821) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %824 = "mix.prim.slice"(%791) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %825 = "mix.prim.slice"(%791) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %826 = "mix.prim.neg"(%825) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %827 = "mix.prim.concat"(%826, %824) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %828 = "mix.prim.mul"(%827, %822) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %829 = "mix.prim.add"(%823, %828) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %830 = "mix.prim.mul"(%792, %821) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %831 = "mix.prim.slice"(%792) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %832 = "mix.prim.slice"(%792) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %833 = "mix.prim.neg"(%832) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %834 = "mix.prim.concat"(%833, %831) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %835 = "mix.prim.mul"(%834, %822) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %836 = "mix.prim.add"(%830, %835) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %837 = "mix.prim.reshape"(%829) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %838 = "mix.prim.reshape"(%836) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %839 = "mix.prim.reshape"(%837) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %840 = "mix.prim.reshape"(%838) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %841 = "mix.prim.transpose"(%839) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %842 = "mix.prim.transpose"(%840) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %843 = "mix.prim.transpose"(%842) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %844 = "mix.prim.unsqueeze"(%841) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %845 = "mix.prim.permute"(%844) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %846 = "mix.prim.unsqueeze"(%843) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %847 = "mix.prim.permute"(%846) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %848 = "mix.prim.permute"(%845) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %849 = "mix.prim.reshape"(%848) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %850 = "mix.prim.permute"(%847) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %851 = "mix.prim.reshape"(%850) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %852 = "mix.prim.batch_matmul"(%849, %851) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %853 = "mix.prim.reshape"(%852) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %854 = "mix.prim.permute"(%853) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %855 = "mix.prim.reshape"(%854) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %856 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %857 = "mix.prim.mul"(%855, %856) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %858 = "mix.prim.reshape"(%857) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %859 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %860 = "mix.comp.masked_fill"(%858, %14, %859) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %861 = "mix.comp.softmax"(%860) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %862 = "mix.prim.reshape"(%861) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %863 = "mix.prim.reshape"(%790) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %864 = "mix.prim.transpose"(%863) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %865 = "mix.prim.batch_matmul"(%862, %864) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %866 = "mix.prim.reshape"(%865) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %867 = "mix.prim.permute"(%866) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %868 = "mix.prim.reshape"(%867) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %869 = "mix.prim.reshape"(%868) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %870 = "mix.prim.transpose"(%776) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %871 = "mix.prim.matmul"(%869, %870) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %872 = "mix.prim.add"(%871, %777) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %873 = "mix.prim.reshape"(%872) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %874 = "mix.prim.mul"(%764, %873) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %875 = "mix.comp.weight"() <{param_loc = "transformer.h.6.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %876 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %877 = "mix.prim.pow"(%874, %876) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %878 = "mix.comp.mean"(%877) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %879 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %880 = "mix.prim.add"(%878, %879) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %881 = "mix.prim.rsqrt"(%880) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %882 = "mix.prim.mul"(%874, %881) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %883 = "mix.prim.mul"(%875, %882) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %884 = "mix.module.linear"(%883) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.6.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %885 = "mix.comp.silu"(%884) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %886 = "mix.module.linear"(%883) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.6.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %887 = "mix.prim.mul"(%885, %886) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %888 = "mix.module.linear"(%887) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.6.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %889 = "mix.prim.add"(%888, %874) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %890 = "mix.comp.weight"() <{param_loc = "transformer.h.7.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %891 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %892 = "mix.prim.pow"(%889, %891) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %893 = "mix.comp.mean"(%892) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %894 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %895 = "mix.prim.add"(%893, %894) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %896 = "mix.prim.rsqrt"(%895) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %897 = "mix.prim.mul"(%889, %896) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %898 = "mix.prim.mul"(%890, %897) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %899 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %900 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %901 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %902 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %903 = "mix.prim.transpose"(%898) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %904 = "mix.prim.transpose"(%899) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %905 = "mix.prim.reshape"(%903) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %906 = "mix.prim.matmul"(%905, %904) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %907 = "mix.prim.reshape"(%906) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %908 = "mix.prim.reshape"(%907) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %909 = "mix.prim.transpose"(%900) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %910 = "mix.prim.reshape"(%903) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %911 = "mix.prim.matmul"(%910, %909) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %912 = "mix.prim.reshape"(%911) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %913 = "mix.prim.reshape"(%912) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %914 = "mix.prim.slice"(%913) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %915 = "mix.prim.slice"(%913) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %916 = "mix.prim.reshape"(%908) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %917 = "mix.prim.reshape"(%914) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %918 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %919 = "mix.prim.convert"(%918) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %920 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %921 = "mix.prim.div"(%919, %920) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %922 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %923 = "mix.prim.pow"(%922, %921) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %924 = "mix.prim.reciprocal"(%923) : (tensor<80xf16>) -> tensor<80xf16>
    %925 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %926 = "mix.prim.mul"(%925, %924) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %927 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %928 = "mix.prim.unsqueeze"(%927) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %929 = "mix.prim.permute"(%928) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %930 = "mix.prim.unsqueeze"(%926) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %931 = "mix.prim.permute"(%930) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %932 = "mix.prim.mul"(%929, %931) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %933 = "mix.prim.concat"(%932, %932) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %934 = "mix.prim.cos"(%933) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %935 = "mix.prim.slice"(%934) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %936 = "mix.prim.unsqueeze"(%935) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %937 = "mix.prim.slice"(%936) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %938 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %939 = "mix.prim.mul"(%937, %938) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %940 = "mix.prim.sin"(%933) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %941 = "mix.prim.slice"(%940) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %942 = "mix.prim.unsqueeze"(%941) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %943 = "mix.prim.slice"(%942) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %944 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %945 = "mix.prim.mul"(%943, %944) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %946 = "mix.prim.slice"(%939) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %947 = "mix.prim.slice"(%945) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %948 = "mix.prim.mul"(%916, %946) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %949 = "mix.prim.slice"(%916) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %950 = "mix.prim.slice"(%916) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %951 = "mix.prim.neg"(%950) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %952 = "mix.prim.concat"(%951, %949) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %953 = "mix.prim.mul"(%952, %947) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %954 = "mix.prim.add"(%948, %953) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %955 = "mix.prim.mul"(%917, %946) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %956 = "mix.prim.slice"(%917) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %957 = "mix.prim.slice"(%917) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %958 = "mix.prim.neg"(%957) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %959 = "mix.prim.concat"(%958, %956) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %960 = "mix.prim.mul"(%959, %947) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %961 = "mix.prim.add"(%955, %960) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %962 = "mix.prim.reshape"(%954) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %963 = "mix.prim.reshape"(%961) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %964 = "mix.prim.reshape"(%962) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %965 = "mix.prim.reshape"(%963) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %966 = "mix.prim.transpose"(%964) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %967 = "mix.prim.transpose"(%965) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %968 = "mix.prim.transpose"(%967) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %969 = "mix.prim.unsqueeze"(%966) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %970 = "mix.prim.permute"(%969) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %971 = "mix.prim.unsqueeze"(%968) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %972 = "mix.prim.permute"(%971) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %973 = "mix.prim.permute"(%970) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %974 = "mix.prim.reshape"(%973) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %975 = "mix.prim.permute"(%972) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %976 = "mix.prim.reshape"(%975) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %977 = "mix.prim.batch_matmul"(%974, %976) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %978 = "mix.prim.reshape"(%977) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %979 = "mix.prim.permute"(%978) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %980 = "mix.prim.reshape"(%979) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %981 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %982 = "mix.prim.mul"(%980, %981) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %983 = "mix.prim.reshape"(%982) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %984 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %985 = "mix.comp.masked_fill"(%983, %14, %984) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %986 = "mix.comp.softmax"(%985) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %987 = "mix.prim.reshape"(%986) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %988 = "mix.prim.reshape"(%915) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %989 = "mix.prim.transpose"(%988) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %990 = "mix.prim.batch_matmul"(%987, %989) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %991 = "mix.prim.reshape"(%990) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %992 = "mix.prim.permute"(%991) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %993 = "mix.prim.reshape"(%992) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %994 = "mix.prim.reshape"(%993) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %995 = "mix.prim.transpose"(%901) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %996 = "mix.prim.matmul"(%994, %995) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %997 = "mix.prim.add"(%996, %902) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %998 = "mix.prim.reshape"(%997) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %999 = "mix.prim.mul"(%889, %998) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1000 = "mix.comp.weight"() <{param_loc = "transformer.h.7.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1001 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1002 = "mix.prim.pow"(%999, %1001) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1003 = "mix.comp.mean"(%1002) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1004 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1005 = "mix.prim.add"(%1003, %1004) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1006 = "mix.prim.rsqrt"(%1005) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1007 = "mix.prim.mul"(%999, %1006) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1008 = "mix.prim.mul"(%1000, %1007) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1009 = "mix.module.linear"(%1008) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.7.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1010 = "mix.comp.silu"(%1009) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1011 = "mix.module.linear"(%1008) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.7.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1012 = "mix.prim.mul"(%1010, %1011) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1013 = "mix.module.linear"(%1012) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.7.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1014 = "mix.prim.add"(%1013, %999) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1015 = "mix.comp.weight"() <{param_loc = "transformer.h.8.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1016 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1017 = "mix.prim.pow"(%1014, %1016) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1018 = "mix.comp.mean"(%1017) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1019 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1020 = "mix.prim.add"(%1018, %1019) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1021 = "mix.prim.rsqrt"(%1020) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1022 = "mix.prim.mul"(%1014, %1021) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1023 = "mix.prim.mul"(%1015, %1022) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1024 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1025 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1026 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1027 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1028 = "mix.prim.transpose"(%1023) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1029 = "mix.prim.transpose"(%1024) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1030 = "mix.prim.reshape"(%1028) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1031 = "mix.prim.matmul"(%1030, %1029) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1032 = "mix.prim.reshape"(%1031) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1033 = "mix.prim.reshape"(%1032) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1034 = "mix.prim.transpose"(%1025) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1035 = "mix.prim.reshape"(%1028) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1036 = "mix.prim.matmul"(%1035, %1034) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1037 = "mix.prim.reshape"(%1036) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1038 = "mix.prim.reshape"(%1037) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1039 = "mix.prim.slice"(%1038) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1040 = "mix.prim.slice"(%1038) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1041 = "mix.prim.reshape"(%1033) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1042 = "mix.prim.reshape"(%1039) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1043 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1044 = "mix.prim.convert"(%1043) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1045 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %1046 = "mix.prim.div"(%1044, %1045) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %1047 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1048 = "mix.prim.pow"(%1047, %1046) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1049 = "mix.prim.reciprocal"(%1048) : (tensor<80xf16>) -> tensor<80xf16>
    %1050 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1051 = "mix.prim.mul"(%1050, %1049) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1052 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1053 = "mix.prim.unsqueeze"(%1052) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1054 = "mix.prim.permute"(%1053) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1055 = "mix.prim.unsqueeze"(%1051) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1056 = "mix.prim.permute"(%1055) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1057 = "mix.prim.mul"(%1054, %1056) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1058 = "mix.prim.concat"(%1057, %1057) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1059 = "mix.prim.cos"(%1058) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1060 = "mix.prim.slice"(%1059) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1061 = "mix.prim.unsqueeze"(%1060) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1062 = "mix.prim.slice"(%1061) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1063 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1064 = "mix.prim.mul"(%1062, %1063) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1065 = "mix.prim.sin"(%1058) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1066 = "mix.prim.slice"(%1065) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1067 = "mix.prim.unsqueeze"(%1066) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1068 = "mix.prim.slice"(%1067) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1069 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1070 = "mix.prim.mul"(%1068, %1069) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1071 = "mix.prim.slice"(%1064) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1072 = "mix.prim.slice"(%1070) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1073 = "mix.prim.mul"(%1041, %1071) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1074 = "mix.prim.slice"(%1041) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1075 = "mix.prim.slice"(%1041) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1076 = "mix.prim.neg"(%1075) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1077 = "mix.prim.concat"(%1076, %1074) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1078 = "mix.prim.mul"(%1077, %1072) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1079 = "mix.prim.add"(%1073, %1078) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1080 = "mix.prim.mul"(%1042, %1071) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1081 = "mix.prim.slice"(%1042) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1082 = "mix.prim.slice"(%1042) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1083 = "mix.prim.neg"(%1082) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1084 = "mix.prim.concat"(%1083, %1081) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1085 = "mix.prim.mul"(%1084, %1072) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1086 = "mix.prim.add"(%1080, %1085) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1087 = "mix.prim.reshape"(%1079) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1088 = "mix.prim.reshape"(%1086) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1089 = "mix.prim.reshape"(%1087) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1090 = "mix.prim.reshape"(%1088) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1091 = "mix.prim.transpose"(%1089) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1092 = "mix.prim.transpose"(%1090) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1093 = "mix.prim.transpose"(%1092) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1094 = "mix.prim.unsqueeze"(%1091) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1095 = "mix.prim.permute"(%1094) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1096 = "mix.prim.unsqueeze"(%1093) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1097 = "mix.prim.permute"(%1096) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1098 = "mix.prim.permute"(%1095) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1099 = "mix.prim.reshape"(%1098) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1100 = "mix.prim.permute"(%1097) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1101 = "mix.prim.reshape"(%1100) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1102 = "mix.prim.batch_matmul"(%1099, %1101) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1103 = "mix.prim.reshape"(%1102) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1104 = "mix.prim.permute"(%1103) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1105 = "mix.prim.reshape"(%1104) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1106 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1107 = "mix.prim.mul"(%1105, %1106) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1108 = "mix.prim.reshape"(%1107) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1109 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1110 = "mix.comp.masked_fill"(%1108, %14, %1109) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1111 = "mix.comp.softmax"(%1110) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1112 = "mix.prim.reshape"(%1111) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1113 = "mix.prim.reshape"(%1040) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1114 = "mix.prim.transpose"(%1113) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1115 = "mix.prim.batch_matmul"(%1112, %1114) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1116 = "mix.prim.reshape"(%1115) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1117 = "mix.prim.permute"(%1116) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1118 = "mix.prim.reshape"(%1117) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1119 = "mix.prim.reshape"(%1118) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1120 = "mix.prim.transpose"(%1026) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1121 = "mix.prim.matmul"(%1119, %1120) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1122 = "mix.prim.add"(%1121, %1027) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1123 = "mix.prim.reshape"(%1122) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1124 = "mix.prim.mul"(%1014, %1123) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1125 = "mix.comp.weight"() <{param_loc = "transformer.h.8.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1126 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1127 = "mix.prim.pow"(%1124, %1126) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1128 = "mix.comp.mean"(%1127) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1129 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1130 = "mix.prim.add"(%1128, %1129) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1131 = "mix.prim.rsqrt"(%1130) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1132 = "mix.prim.mul"(%1124, %1131) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1133 = "mix.prim.mul"(%1125, %1132) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1134 = "mix.module.linear"(%1133) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.8.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1135 = "mix.comp.silu"(%1134) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1136 = "mix.module.linear"(%1133) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.8.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1137 = "mix.prim.mul"(%1135, %1136) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1138 = "mix.module.linear"(%1137) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.8.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1139 = "mix.prim.add"(%1138, %1124) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1140 = "mix.comp.weight"() <{param_loc = "transformer.h.9.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1141 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1142 = "mix.prim.pow"(%1139, %1141) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1143 = "mix.comp.mean"(%1142) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1144 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1145 = "mix.prim.add"(%1143, %1144) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1146 = "mix.prim.rsqrt"(%1145) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1147 = "mix.prim.mul"(%1139, %1146) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1148 = "mix.prim.mul"(%1140, %1147) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1149 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1150 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1151 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1152 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1153 = "mix.prim.transpose"(%1148) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1154 = "mix.prim.transpose"(%1149) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1155 = "mix.prim.reshape"(%1153) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1156 = "mix.prim.matmul"(%1155, %1154) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1157 = "mix.prim.reshape"(%1156) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1158 = "mix.prim.reshape"(%1157) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1159 = "mix.prim.transpose"(%1150) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1160 = "mix.prim.reshape"(%1153) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1161 = "mix.prim.matmul"(%1160, %1159) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1162 = "mix.prim.reshape"(%1161) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1163 = "mix.prim.reshape"(%1162) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1164 = "mix.prim.slice"(%1163) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1165 = "mix.prim.slice"(%1163) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1166 = "mix.prim.reshape"(%1158) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1167 = "mix.prim.reshape"(%1164) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1168 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1169 = "mix.prim.convert"(%1168) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1170 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %1171 = "mix.prim.div"(%1169, %1170) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %1172 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1173 = "mix.prim.pow"(%1172, %1171) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1174 = "mix.prim.reciprocal"(%1173) : (tensor<80xf16>) -> tensor<80xf16>
    %1175 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1176 = "mix.prim.mul"(%1175, %1174) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1177 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1178 = "mix.prim.unsqueeze"(%1177) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1179 = "mix.prim.permute"(%1178) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1180 = "mix.prim.unsqueeze"(%1176) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1181 = "mix.prim.permute"(%1180) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1182 = "mix.prim.mul"(%1179, %1181) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1183 = "mix.prim.concat"(%1182, %1182) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1184 = "mix.prim.cos"(%1183) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1185 = "mix.prim.slice"(%1184) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1186 = "mix.prim.unsqueeze"(%1185) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1187 = "mix.prim.slice"(%1186) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1188 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1189 = "mix.prim.mul"(%1187, %1188) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1190 = "mix.prim.sin"(%1183) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1191 = "mix.prim.slice"(%1190) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1192 = "mix.prim.unsqueeze"(%1191) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1193 = "mix.prim.slice"(%1192) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1194 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1195 = "mix.prim.mul"(%1193, %1194) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1196 = "mix.prim.slice"(%1189) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1197 = "mix.prim.slice"(%1195) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1198 = "mix.prim.mul"(%1166, %1196) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1199 = "mix.prim.slice"(%1166) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1200 = "mix.prim.slice"(%1166) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1201 = "mix.prim.neg"(%1200) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1202 = "mix.prim.concat"(%1201, %1199) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1203 = "mix.prim.mul"(%1202, %1197) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1204 = "mix.prim.add"(%1198, %1203) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1205 = "mix.prim.mul"(%1167, %1196) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1206 = "mix.prim.slice"(%1167) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1207 = "mix.prim.slice"(%1167) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1208 = "mix.prim.neg"(%1207) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1209 = "mix.prim.concat"(%1208, %1206) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1210 = "mix.prim.mul"(%1209, %1197) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1211 = "mix.prim.add"(%1205, %1210) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1212 = "mix.prim.reshape"(%1204) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1213 = "mix.prim.reshape"(%1211) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1214 = "mix.prim.reshape"(%1212) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1215 = "mix.prim.reshape"(%1213) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1216 = "mix.prim.transpose"(%1214) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1217 = "mix.prim.transpose"(%1215) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1218 = "mix.prim.transpose"(%1217) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1219 = "mix.prim.unsqueeze"(%1216) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1220 = "mix.prim.permute"(%1219) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1221 = "mix.prim.unsqueeze"(%1218) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1222 = "mix.prim.permute"(%1221) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1223 = "mix.prim.permute"(%1220) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1224 = "mix.prim.reshape"(%1223) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1225 = "mix.prim.permute"(%1222) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1226 = "mix.prim.reshape"(%1225) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1227 = "mix.prim.batch_matmul"(%1224, %1226) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1228 = "mix.prim.reshape"(%1227) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1229 = "mix.prim.permute"(%1228) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1230 = "mix.prim.reshape"(%1229) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1231 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1232 = "mix.prim.mul"(%1230, %1231) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1233 = "mix.prim.reshape"(%1232) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1234 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1235 = "mix.comp.masked_fill"(%1233, %14, %1234) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1236 = "mix.comp.softmax"(%1235) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1237 = "mix.prim.reshape"(%1236) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1238 = "mix.prim.reshape"(%1165) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1239 = "mix.prim.transpose"(%1238) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1240 = "mix.prim.batch_matmul"(%1237, %1239) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1241 = "mix.prim.reshape"(%1240) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1242 = "mix.prim.permute"(%1241) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1243 = "mix.prim.reshape"(%1242) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1244 = "mix.prim.reshape"(%1243) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1245 = "mix.prim.transpose"(%1151) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1246 = "mix.prim.matmul"(%1244, %1245) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1247 = "mix.prim.add"(%1246, %1152) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1248 = "mix.prim.reshape"(%1247) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1249 = "mix.prim.mul"(%1139, %1248) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1250 = "mix.comp.weight"() <{param_loc = "transformer.h.9.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1251 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1252 = "mix.prim.pow"(%1249, %1251) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1253 = "mix.comp.mean"(%1252) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1254 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1255 = "mix.prim.add"(%1253, %1254) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1256 = "mix.prim.rsqrt"(%1255) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1257 = "mix.prim.mul"(%1249, %1256) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1258 = "mix.prim.mul"(%1250, %1257) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1259 = "mix.module.linear"(%1258) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.9.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1260 = "mix.comp.silu"(%1259) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1261 = "mix.module.linear"(%1258) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.9.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1262 = "mix.prim.mul"(%1260, %1261) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1263 = "mix.module.linear"(%1262) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.9.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1264 = "mix.prim.add"(%1263, %1249) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1265 = "mix.comp.weight"() <{param_loc = "transformer.h.10.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1266 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1267 = "mix.prim.pow"(%1264, %1266) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1268 = "mix.comp.mean"(%1267) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1269 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1270 = "mix.prim.add"(%1268, %1269) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1271 = "mix.prim.rsqrt"(%1270) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1272 = "mix.prim.mul"(%1264, %1271) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1273 = "mix.prim.mul"(%1265, %1272) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1274 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1275 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1276 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1277 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1278 = "mix.prim.transpose"(%1273) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1279 = "mix.prim.transpose"(%1274) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1280 = "mix.prim.reshape"(%1278) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1281 = "mix.prim.matmul"(%1280, %1279) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1282 = "mix.prim.reshape"(%1281) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1283 = "mix.prim.reshape"(%1282) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1284 = "mix.prim.transpose"(%1275) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1285 = "mix.prim.reshape"(%1278) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1286 = "mix.prim.matmul"(%1285, %1284) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1287 = "mix.prim.reshape"(%1286) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1288 = "mix.prim.reshape"(%1287) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1289 = "mix.prim.slice"(%1288) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1290 = "mix.prim.slice"(%1288) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1291 = "mix.prim.reshape"(%1283) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1292 = "mix.prim.reshape"(%1289) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1293 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1294 = "mix.prim.convert"(%1293) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1295 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %1296 = "mix.prim.div"(%1294, %1295) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %1297 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1298 = "mix.prim.pow"(%1297, %1296) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1299 = "mix.prim.reciprocal"(%1298) : (tensor<80xf16>) -> tensor<80xf16>
    %1300 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1301 = "mix.prim.mul"(%1300, %1299) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1302 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1303 = "mix.prim.unsqueeze"(%1302) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1304 = "mix.prim.permute"(%1303) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1305 = "mix.prim.unsqueeze"(%1301) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1306 = "mix.prim.permute"(%1305) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1307 = "mix.prim.mul"(%1304, %1306) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1308 = "mix.prim.concat"(%1307, %1307) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1309 = "mix.prim.cos"(%1308) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1310 = "mix.prim.slice"(%1309) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1311 = "mix.prim.unsqueeze"(%1310) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1312 = "mix.prim.slice"(%1311) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1313 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1314 = "mix.prim.mul"(%1312, %1313) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1315 = "mix.prim.sin"(%1308) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1316 = "mix.prim.slice"(%1315) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1317 = "mix.prim.unsqueeze"(%1316) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1318 = "mix.prim.slice"(%1317) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1319 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1320 = "mix.prim.mul"(%1318, %1319) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1321 = "mix.prim.slice"(%1314) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1322 = "mix.prim.slice"(%1320) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1323 = "mix.prim.mul"(%1291, %1321) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1324 = "mix.prim.slice"(%1291) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1325 = "mix.prim.slice"(%1291) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1326 = "mix.prim.neg"(%1325) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1327 = "mix.prim.concat"(%1326, %1324) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1328 = "mix.prim.mul"(%1327, %1322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1329 = "mix.prim.add"(%1323, %1328) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1330 = "mix.prim.mul"(%1292, %1321) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1331 = "mix.prim.slice"(%1292) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1332 = "mix.prim.slice"(%1292) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1333 = "mix.prim.neg"(%1332) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1334 = "mix.prim.concat"(%1333, %1331) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1335 = "mix.prim.mul"(%1334, %1322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1336 = "mix.prim.add"(%1330, %1335) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1337 = "mix.prim.reshape"(%1329) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1338 = "mix.prim.reshape"(%1336) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1339 = "mix.prim.reshape"(%1337) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1340 = "mix.prim.reshape"(%1338) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1341 = "mix.prim.transpose"(%1339) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1342 = "mix.prim.transpose"(%1340) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1343 = "mix.prim.transpose"(%1342) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1344 = "mix.prim.unsqueeze"(%1341) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1345 = "mix.prim.permute"(%1344) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1346 = "mix.prim.unsqueeze"(%1343) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1347 = "mix.prim.permute"(%1346) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1348 = "mix.prim.permute"(%1345) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1349 = "mix.prim.reshape"(%1348) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1350 = "mix.prim.permute"(%1347) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1351 = "mix.prim.reshape"(%1350) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1352 = "mix.prim.batch_matmul"(%1349, %1351) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1353 = "mix.prim.reshape"(%1352) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1354 = "mix.prim.permute"(%1353) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1355 = "mix.prim.reshape"(%1354) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1356 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1357 = "mix.prim.mul"(%1355, %1356) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1358 = "mix.prim.reshape"(%1357) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1359 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1360 = "mix.comp.masked_fill"(%1358, %14, %1359) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1361 = "mix.comp.softmax"(%1360) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1362 = "mix.prim.reshape"(%1361) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1363 = "mix.prim.reshape"(%1290) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1364 = "mix.prim.transpose"(%1363) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1365 = "mix.prim.batch_matmul"(%1362, %1364) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1366 = "mix.prim.reshape"(%1365) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1367 = "mix.prim.permute"(%1366) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1368 = "mix.prim.reshape"(%1367) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1369 = "mix.prim.reshape"(%1368) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1370 = "mix.prim.transpose"(%1276) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1371 = "mix.prim.matmul"(%1369, %1370) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1372 = "mix.prim.add"(%1371, %1277) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1373 = "mix.prim.reshape"(%1372) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1374 = "mix.prim.mul"(%1264, %1373) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1375 = "mix.comp.weight"() <{param_loc = "transformer.h.10.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1376 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1377 = "mix.prim.pow"(%1374, %1376) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1378 = "mix.comp.mean"(%1377) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1379 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1380 = "mix.prim.add"(%1378, %1379) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1381 = "mix.prim.rsqrt"(%1380) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1382 = "mix.prim.mul"(%1374, %1381) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1383 = "mix.prim.mul"(%1375, %1382) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1384 = "mix.module.linear"(%1383) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.10.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1385 = "mix.comp.silu"(%1384) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1386 = "mix.module.linear"(%1383) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.10.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1387 = "mix.prim.mul"(%1385, %1386) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1388 = "mix.module.linear"(%1387) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.10.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1389 = "mix.prim.add"(%1388, %1374) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1390 = "mix.comp.weight"() <{param_loc = "transformer.h.11.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1391 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1392 = "mix.prim.pow"(%1389, %1391) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1393 = "mix.comp.mean"(%1392) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1394 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1395 = "mix.prim.add"(%1393, %1394) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1396 = "mix.prim.rsqrt"(%1395) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1397 = "mix.prim.mul"(%1389, %1396) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1398 = "mix.prim.mul"(%1390, %1397) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1399 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1400 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1401 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1402 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1403 = "mix.prim.transpose"(%1398) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1404 = "mix.prim.transpose"(%1399) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1405 = "mix.prim.reshape"(%1403) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1406 = "mix.prim.matmul"(%1405, %1404) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1407 = "mix.prim.reshape"(%1406) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1408 = "mix.prim.reshape"(%1407) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1409 = "mix.prim.transpose"(%1400) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1410 = "mix.prim.reshape"(%1403) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1411 = "mix.prim.matmul"(%1410, %1409) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1412 = "mix.prim.reshape"(%1411) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1413 = "mix.prim.reshape"(%1412) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1414 = "mix.prim.slice"(%1413) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1415 = "mix.prim.slice"(%1413) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1416 = "mix.prim.reshape"(%1408) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1417 = "mix.prim.reshape"(%1414) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1418 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1419 = "mix.prim.convert"(%1418) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1420 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %1421 = "mix.prim.div"(%1419, %1420) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %1422 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1423 = "mix.prim.pow"(%1422, %1421) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1424 = "mix.prim.reciprocal"(%1423) : (tensor<80xf16>) -> tensor<80xf16>
    %1425 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1426 = "mix.prim.mul"(%1425, %1424) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1427 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1428 = "mix.prim.unsqueeze"(%1427) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1429 = "mix.prim.permute"(%1428) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1430 = "mix.prim.unsqueeze"(%1426) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1431 = "mix.prim.permute"(%1430) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1432 = "mix.prim.mul"(%1429, %1431) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1433 = "mix.prim.concat"(%1432, %1432) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1434 = "mix.prim.cos"(%1433) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1435 = "mix.prim.slice"(%1434) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1436 = "mix.prim.unsqueeze"(%1435) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1437 = "mix.prim.slice"(%1436) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1438 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1439 = "mix.prim.mul"(%1437, %1438) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1440 = "mix.prim.sin"(%1433) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1441 = "mix.prim.slice"(%1440) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1442 = "mix.prim.unsqueeze"(%1441) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1443 = "mix.prim.slice"(%1442) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1444 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1445 = "mix.prim.mul"(%1443, %1444) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1446 = "mix.prim.slice"(%1439) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1447 = "mix.prim.slice"(%1445) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1448 = "mix.prim.mul"(%1416, %1446) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1449 = "mix.prim.slice"(%1416) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1450 = "mix.prim.slice"(%1416) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1451 = "mix.prim.neg"(%1450) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1452 = "mix.prim.concat"(%1451, %1449) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1453 = "mix.prim.mul"(%1452, %1447) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1454 = "mix.prim.add"(%1448, %1453) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1455 = "mix.prim.mul"(%1417, %1446) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1456 = "mix.prim.slice"(%1417) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1457 = "mix.prim.slice"(%1417) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1458 = "mix.prim.neg"(%1457) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1459 = "mix.prim.concat"(%1458, %1456) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1460 = "mix.prim.mul"(%1459, %1447) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1461 = "mix.prim.add"(%1455, %1460) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1462 = "mix.prim.reshape"(%1454) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1463 = "mix.prim.reshape"(%1461) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1464 = "mix.prim.reshape"(%1462) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1465 = "mix.prim.reshape"(%1463) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1466 = "mix.prim.transpose"(%1464) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1467 = "mix.prim.transpose"(%1465) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1468 = "mix.prim.transpose"(%1467) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1469 = "mix.prim.unsqueeze"(%1466) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1470 = "mix.prim.permute"(%1469) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1471 = "mix.prim.unsqueeze"(%1468) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1472 = "mix.prim.permute"(%1471) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1473 = "mix.prim.permute"(%1470) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1474 = "mix.prim.reshape"(%1473) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1475 = "mix.prim.permute"(%1472) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1476 = "mix.prim.reshape"(%1475) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1477 = "mix.prim.batch_matmul"(%1474, %1476) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1478 = "mix.prim.reshape"(%1477) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1479 = "mix.prim.permute"(%1478) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1480 = "mix.prim.reshape"(%1479) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1481 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1482 = "mix.prim.mul"(%1480, %1481) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1483 = "mix.prim.reshape"(%1482) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1484 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1485 = "mix.comp.masked_fill"(%1483, %14, %1484) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1486 = "mix.comp.softmax"(%1485) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1487 = "mix.prim.reshape"(%1486) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1488 = "mix.prim.reshape"(%1415) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1489 = "mix.prim.transpose"(%1488) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1490 = "mix.prim.batch_matmul"(%1487, %1489) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1491 = "mix.prim.reshape"(%1490) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1492 = "mix.prim.permute"(%1491) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1493 = "mix.prim.reshape"(%1492) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1494 = "mix.prim.reshape"(%1493) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1495 = "mix.prim.transpose"(%1401) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1496 = "mix.prim.matmul"(%1494, %1495) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1497 = "mix.prim.add"(%1496, %1402) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1498 = "mix.prim.reshape"(%1497) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1499 = "mix.prim.mul"(%1389, %1498) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1500 = "mix.comp.weight"() <{param_loc = "transformer.h.11.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1501 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1502 = "mix.prim.pow"(%1499, %1501) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1503 = "mix.comp.mean"(%1502) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1504 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1505 = "mix.prim.add"(%1503, %1504) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1506 = "mix.prim.rsqrt"(%1505) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1507 = "mix.prim.mul"(%1499, %1506) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1508 = "mix.prim.mul"(%1500, %1507) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1509 = "mix.module.linear"(%1508) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.11.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1510 = "mix.comp.silu"(%1509) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1511 = "mix.module.linear"(%1508) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.11.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1512 = "mix.prim.mul"(%1510, %1511) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1513 = "mix.module.linear"(%1512) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.11.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1514 = "mix.prim.add"(%1513, %1499) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1515 = "mix.comp.weight"() <{param_loc = "transformer.h.12.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1516 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1517 = "mix.prim.pow"(%1514, %1516) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1518 = "mix.comp.mean"(%1517) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1519 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1520 = "mix.prim.add"(%1518, %1519) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1521 = "mix.prim.rsqrt"(%1520) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1522 = "mix.prim.mul"(%1514, %1521) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1523 = "mix.prim.mul"(%1515, %1522) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1524 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1525 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1526 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1527 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1528 = "mix.prim.transpose"(%1523) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1529 = "mix.prim.transpose"(%1524) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1530 = "mix.prim.reshape"(%1528) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1531 = "mix.prim.matmul"(%1530, %1529) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1532 = "mix.prim.reshape"(%1531) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1533 = "mix.prim.reshape"(%1532) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1534 = "mix.prim.transpose"(%1525) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1535 = "mix.prim.reshape"(%1528) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1536 = "mix.prim.matmul"(%1535, %1534) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1537 = "mix.prim.reshape"(%1536) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1538 = "mix.prim.reshape"(%1537) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1539 = "mix.prim.slice"(%1538) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1540 = "mix.prim.slice"(%1538) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1541 = "mix.prim.reshape"(%1533) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1542 = "mix.prim.reshape"(%1539) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1543 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1544 = "mix.prim.convert"(%1543) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1545 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %1546 = "mix.prim.div"(%1544, %1545) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %1547 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1548 = "mix.prim.pow"(%1547, %1546) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1549 = "mix.prim.reciprocal"(%1548) : (tensor<80xf16>) -> tensor<80xf16>
    %1550 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1551 = "mix.prim.mul"(%1550, %1549) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1552 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1553 = "mix.prim.unsqueeze"(%1552) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1554 = "mix.prim.permute"(%1553) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1555 = "mix.prim.unsqueeze"(%1551) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1556 = "mix.prim.permute"(%1555) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1557 = "mix.prim.mul"(%1554, %1556) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1558 = "mix.prim.concat"(%1557, %1557) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1559 = "mix.prim.cos"(%1558) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1560 = "mix.prim.slice"(%1559) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1561 = "mix.prim.unsqueeze"(%1560) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1562 = "mix.prim.slice"(%1561) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1563 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1564 = "mix.prim.mul"(%1562, %1563) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1565 = "mix.prim.sin"(%1558) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1566 = "mix.prim.slice"(%1565) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1567 = "mix.prim.unsqueeze"(%1566) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1568 = "mix.prim.slice"(%1567) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1569 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1570 = "mix.prim.mul"(%1568, %1569) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1571 = "mix.prim.slice"(%1564) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1572 = "mix.prim.slice"(%1570) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1573 = "mix.prim.mul"(%1541, %1571) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1574 = "mix.prim.slice"(%1541) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1575 = "mix.prim.slice"(%1541) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1576 = "mix.prim.neg"(%1575) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1577 = "mix.prim.concat"(%1576, %1574) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1578 = "mix.prim.mul"(%1577, %1572) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1579 = "mix.prim.add"(%1573, %1578) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1580 = "mix.prim.mul"(%1542, %1571) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1581 = "mix.prim.slice"(%1542) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1582 = "mix.prim.slice"(%1542) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1583 = "mix.prim.neg"(%1582) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1584 = "mix.prim.concat"(%1583, %1581) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1585 = "mix.prim.mul"(%1584, %1572) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1586 = "mix.prim.add"(%1580, %1585) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1587 = "mix.prim.reshape"(%1579) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1588 = "mix.prim.reshape"(%1586) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1589 = "mix.prim.reshape"(%1587) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1590 = "mix.prim.reshape"(%1588) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1591 = "mix.prim.transpose"(%1589) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1592 = "mix.prim.transpose"(%1590) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1593 = "mix.prim.transpose"(%1592) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1594 = "mix.prim.unsqueeze"(%1591) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1595 = "mix.prim.permute"(%1594) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1596 = "mix.prim.unsqueeze"(%1593) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1597 = "mix.prim.permute"(%1596) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1598 = "mix.prim.permute"(%1595) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1599 = "mix.prim.reshape"(%1598) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1600 = "mix.prim.permute"(%1597) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1601 = "mix.prim.reshape"(%1600) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1602 = "mix.prim.batch_matmul"(%1599, %1601) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1603 = "mix.prim.reshape"(%1602) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1604 = "mix.prim.permute"(%1603) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1605 = "mix.prim.reshape"(%1604) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1606 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1607 = "mix.prim.mul"(%1605, %1606) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1608 = "mix.prim.reshape"(%1607) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1609 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1610 = "mix.comp.masked_fill"(%1608, %14, %1609) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1611 = "mix.comp.softmax"(%1610) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1612 = "mix.prim.reshape"(%1611) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1613 = "mix.prim.reshape"(%1540) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1614 = "mix.prim.transpose"(%1613) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1615 = "mix.prim.batch_matmul"(%1612, %1614) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1616 = "mix.prim.reshape"(%1615) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1617 = "mix.prim.permute"(%1616) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1618 = "mix.prim.reshape"(%1617) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1619 = "mix.prim.reshape"(%1618) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1620 = "mix.prim.transpose"(%1526) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1621 = "mix.prim.matmul"(%1619, %1620) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1622 = "mix.prim.add"(%1621, %1527) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1623 = "mix.prim.reshape"(%1622) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1624 = "mix.prim.mul"(%1514, %1623) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1625 = "mix.comp.weight"() <{param_loc = "transformer.h.12.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1626 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1627 = "mix.prim.pow"(%1624, %1626) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1628 = "mix.comp.mean"(%1627) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1629 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1630 = "mix.prim.add"(%1628, %1629) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1631 = "mix.prim.rsqrt"(%1630) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1632 = "mix.prim.mul"(%1624, %1631) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1633 = "mix.prim.mul"(%1625, %1632) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1634 = "mix.module.linear"(%1633) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.12.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1635 = "mix.comp.silu"(%1634) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1636 = "mix.module.linear"(%1633) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.12.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1637 = "mix.prim.mul"(%1635, %1636) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1638 = "mix.module.linear"(%1637) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.12.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1639 = "mix.prim.add"(%1638, %1624) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1640 = "mix.comp.weight"() <{param_loc = "transformer.h.13.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1641 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1642 = "mix.prim.pow"(%1639, %1641) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1643 = "mix.comp.mean"(%1642) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1644 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1645 = "mix.prim.add"(%1643, %1644) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1646 = "mix.prim.rsqrt"(%1645) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1647 = "mix.prim.mul"(%1639, %1646) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1648 = "mix.prim.mul"(%1640, %1647) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1649 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1650 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1651 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1652 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1653 = "mix.prim.transpose"(%1648) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1654 = "mix.prim.transpose"(%1649) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1655 = "mix.prim.reshape"(%1653) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1656 = "mix.prim.matmul"(%1655, %1654) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1657 = "mix.prim.reshape"(%1656) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1658 = "mix.prim.reshape"(%1657) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1659 = "mix.prim.transpose"(%1650) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1660 = "mix.prim.reshape"(%1653) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1661 = "mix.prim.matmul"(%1660, %1659) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1662 = "mix.prim.reshape"(%1661) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1663 = "mix.prim.reshape"(%1662) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1664 = "mix.prim.slice"(%1663) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1665 = "mix.prim.slice"(%1663) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1666 = "mix.prim.reshape"(%1658) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1667 = "mix.prim.reshape"(%1664) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1668 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1669 = "mix.prim.convert"(%1668) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1670 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %1671 = "mix.prim.div"(%1669, %1670) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %1672 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1673 = "mix.prim.pow"(%1672, %1671) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1674 = "mix.prim.reciprocal"(%1673) : (tensor<80xf16>) -> tensor<80xf16>
    %1675 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1676 = "mix.prim.mul"(%1675, %1674) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1677 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1678 = "mix.prim.unsqueeze"(%1677) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1679 = "mix.prim.permute"(%1678) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1680 = "mix.prim.unsqueeze"(%1676) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1681 = "mix.prim.permute"(%1680) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1682 = "mix.prim.mul"(%1679, %1681) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1683 = "mix.prim.concat"(%1682, %1682) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1684 = "mix.prim.cos"(%1683) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1685 = "mix.prim.slice"(%1684) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1686 = "mix.prim.unsqueeze"(%1685) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1687 = "mix.prim.slice"(%1686) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1688 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1689 = "mix.prim.mul"(%1687, %1688) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1690 = "mix.prim.sin"(%1683) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1691 = "mix.prim.slice"(%1690) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1692 = "mix.prim.unsqueeze"(%1691) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1693 = "mix.prim.slice"(%1692) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1694 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1695 = "mix.prim.mul"(%1693, %1694) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1696 = "mix.prim.slice"(%1689) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1697 = "mix.prim.slice"(%1695) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1698 = "mix.prim.mul"(%1666, %1696) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1699 = "mix.prim.slice"(%1666) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1700 = "mix.prim.slice"(%1666) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1701 = "mix.prim.neg"(%1700) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1702 = "mix.prim.concat"(%1701, %1699) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1703 = "mix.prim.mul"(%1702, %1697) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1704 = "mix.prim.add"(%1698, %1703) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1705 = "mix.prim.mul"(%1667, %1696) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1706 = "mix.prim.slice"(%1667) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1707 = "mix.prim.slice"(%1667) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1708 = "mix.prim.neg"(%1707) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1709 = "mix.prim.concat"(%1708, %1706) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1710 = "mix.prim.mul"(%1709, %1697) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1711 = "mix.prim.add"(%1705, %1710) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1712 = "mix.prim.reshape"(%1704) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1713 = "mix.prim.reshape"(%1711) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1714 = "mix.prim.reshape"(%1712) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1715 = "mix.prim.reshape"(%1713) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1716 = "mix.prim.transpose"(%1714) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1717 = "mix.prim.transpose"(%1715) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1718 = "mix.prim.transpose"(%1717) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1719 = "mix.prim.unsqueeze"(%1716) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1720 = "mix.prim.permute"(%1719) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1721 = "mix.prim.unsqueeze"(%1718) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1722 = "mix.prim.permute"(%1721) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1723 = "mix.prim.permute"(%1720) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1724 = "mix.prim.reshape"(%1723) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1725 = "mix.prim.permute"(%1722) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1726 = "mix.prim.reshape"(%1725) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1727 = "mix.prim.batch_matmul"(%1724, %1726) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1728 = "mix.prim.reshape"(%1727) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1729 = "mix.prim.permute"(%1728) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1730 = "mix.prim.reshape"(%1729) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1731 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1732 = "mix.prim.mul"(%1730, %1731) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1733 = "mix.prim.reshape"(%1732) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1734 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1735 = "mix.comp.masked_fill"(%1733, %14, %1734) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1736 = "mix.comp.softmax"(%1735) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1737 = "mix.prim.reshape"(%1736) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1738 = "mix.prim.reshape"(%1665) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1739 = "mix.prim.transpose"(%1738) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1740 = "mix.prim.batch_matmul"(%1737, %1739) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1741 = "mix.prim.reshape"(%1740) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1742 = "mix.prim.permute"(%1741) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1743 = "mix.prim.reshape"(%1742) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1744 = "mix.prim.reshape"(%1743) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1745 = "mix.prim.transpose"(%1651) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1746 = "mix.prim.matmul"(%1744, %1745) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1747 = "mix.prim.add"(%1746, %1652) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1748 = "mix.prim.reshape"(%1747) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1749 = "mix.prim.mul"(%1639, %1748) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1750 = "mix.comp.weight"() <{param_loc = "transformer.h.13.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1751 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1752 = "mix.prim.pow"(%1749, %1751) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1753 = "mix.comp.mean"(%1752) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1754 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1755 = "mix.prim.add"(%1753, %1754) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1756 = "mix.prim.rsqrt"(%1755) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1757 = "mix.prim.mul"(%1749, %1756) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1758 = "mix.prim.mul"(%1750, %1757) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1759 = "mix.module.linear"(%1758) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.13.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1760 = "mix.comp.silu"(%1759) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1761 = "mix.module.linear"(%1758) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.13.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1762 = "mix.prim.mul"(%1760, %1761) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1763 = "mix.module.linear"(%1762) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.13.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1764 = "mix.prim.add"(%1763, %1749) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1765 = "mix.comp.weight"() <{param_loc = "transformer.h.14.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1766 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1767 = "mix.prim.pow"(%1764, %1766) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1768 = "mix.comp.mean"(%1767) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1769 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1770 = "mix.prim.add"(%1768, %1769) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1771 = "mix.prim.rsqrt"(%1770) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1772 = "mix.prim.mul"(%1764, %1771) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1773 = "mix.prim.mul"(%1765, %1772) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1774 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1775 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1776 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1777 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1778 = "mix.prim.transpose"(%1773) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1779 = "mix.prim.transpose"(%1774) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1780 = "mix.prim.reshape"(%1778) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1781 = "mix.prim.matmul"(%1780, %1779) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1782 = "mix.prim.reshape"(%1781) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1783 = "mix.prim.reshape"(%1782) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1784 = "mix.prim.transpose"(%1775) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1785 = "mix.prim.reshape"(%1778) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1786 = "mix.prim.matmul"(%1785, %1784) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1787 = "mix.prim.reshape"(%1786) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1788 = "mix.prim.reshape"(%1787) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1789 = "mix.prim.slice"(%1788) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1790 = "mix.prim.slice"(%1788) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1791 = "mix.prim.reshape"(%1783) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1792 = "mix.prim.reshape"(%1789) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1793 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1794 = "mix.prim.convert"(%1793) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1795 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %1796 = "mix.prim.div"(%1794, %1795) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %1797 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1798 = "mix.prim.pow"(%1797, %1796) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1799 = "mix.prim.reciprocal"(%1798) : (tensor<80xf16>) -> tensor<80xf16>
    %1800 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1801 = "mix.prim.mul"(%1800, %1799) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1802 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1803 = "mix.prim.unsqueeze"(%1802) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1804 = "mix.prim.permute"(%1803) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1805 = "mix.prim.unsqueeze"(%1801) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1806 = "mix.prim.permute"(%1805) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1807 = "mix.prim.mul"(%1804, %1806) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1808 = "mix.prim.concat"(%1807, %1807) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1809 = "mix.prim.cos"(%1808) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1810 = "mix.prim.slice"(%1809) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1811 = "mix.prim.unsqueeze"(%1810) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1812 = "mix.prim.slice"(%1811) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1813 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1814 = "mix.prim.mul"(%1812, %1813) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1815 = "mix.prim.sin"(%1808) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1816 = "mix.prim.slice"(%1815) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1817 = "mix.prim.unsqueeze"(%1816) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1818 = "mix.prim.slice"(%1817) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1819 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1820 = "mix.prim.mul"(%1818, %1819) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1821 = "mix.prim.slice"(%1814) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1822 = "mix.prim.slice"(%1820) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1823 = "mix.prim.mul"(%1791, %1821) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1824 = "mix.prim.slice"(%1791) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1825 = "mix.prim.slice"(%1791) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1826 = "mix.prim.neg"(%1825) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1827 = "mix.prim.concat"(%1826, %1824) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1828 = "mix.prim.mul"(%1827, %1822) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1829 = "mix.prim.add"(%1823, %1828) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1830 = "mix.prim.mul"(%1792, %1821) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1831 = "mix.prim.slice"(%1792) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1832 = "mix.prim.slice"(%1792) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1833 = "mix.prim.neg"(%1832) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1834 = "mix.prim.concat"(%1833, %1831) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1835 = "mix.prim.mul"(%1834, %1822) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1836 = "mix.prim.add"(%1830, %1835) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1837 = "mix.prim.reshape"(%1829) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1838 = "mix.prim.reshape"(%1836) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1839 = "mix.prim.reshape"(%1837) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1840 = "mix.prim.reshape"(%1838) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1841 = "mix.prim.transpose"(%1839) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1842 = "mix.prim.transpose"(%1840) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1843 = "mix.prim.transpose"(%1842) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1844 = "mix.prim.unsqueeze"(%1841) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1845 = "mix.prim.permute"(%1844) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1846 = "mix.prim.unsqueeze"(%1843) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1847 = "mix.prim.permute"(%1846) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1848 = "mix.prim.permute"(%1845) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1849 = "mix.prim.reshape"(%1848) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1850 = "mix.prim.permute"(%1847) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1851 = "mix.prim.reshape"(%1850) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1852 = "mix.prim.batch_matmul"(%1849, %1851) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1853 = "mix.prim.reshape"(%1852) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1854 = "mix.prim.permute"(%1853) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1855 = "mix.prim.reshape"(%1854) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1856 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1857 = "mix.prim.mul"(%1855, %1856) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1858 = "mix.prim.reshape"(%1857) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1859 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1860 = "mix.comp.masked_fill"(%1858, %14, %1859) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1861 = "mix.comp.softmax"(%1860) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1862 = "mix.prim.reshape"(%1861) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1863 = "mix.prim.reshape"(%1790) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1864 = "mix.prim.transpose"(%1863) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1865 = "mix.prim.batch_matmul"(%1862, %1864) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1866 = "mix.prim.reshape"(%1865) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1867 = "mix.prim.permute"(%1866) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1868 = "mix.prim.reshape"(%1867) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1869 = "mix.prim.reshape"(%1868) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1870 = "mix.prim.transpose"(%1776) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1871 = "mix.prim.matmul"(%1869, %1870) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1872 = "mix.prim.add"(%1871, %1777) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1873 = "mix.prim.reshape"(%1872) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1874 = "mix.prim.mul"(%1764, %1873) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1875 = "mix.comp.weight"() <{param_loc = "transformer.h.14.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1876 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1877 = "mix.prim.pow"(%1874, %1876) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1878 = "mix.comp.mean"(%1877) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1879 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1880 = "mix.prim.add"(%1878, %1879) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1881 = "mix.prim.rsqrt"(%1880) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1882 = "mix.prim.mul"(%1874, %1881) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1883 = "mix.prim.mul"(%1875, %1882) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1884 = "mix.module.linear"(%1883) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.14.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1885 = "mix.comp.silu"(%1884) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1886 = "mix.module.linear"(%1883) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.14.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1887 = "mix.prim.mul"(%1885, %1886) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1888 = "mix.module.linear"(%1887) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.14.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1889 = "mix.prim.add"(%1888, %1874) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1890 = "mix.comp.weight"() <{param_loc = "transformer.h.15.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1891 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1892 = "mix.prim.pow"(%1889, %1891) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1893 = "mix.comp.mean"(%1892) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1894 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1895 = "mix.prim.add"(%1893, %1894) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1896 = "mix.prim.rsqrt"(%1895) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1897 = "mix.prim.mul"(%1889, %1896) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1898 = "mix.prim.mul"(%1890, %1897) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1899 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1900 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1901 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1902 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1903 = "mix.prim.transpose"(%1898) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1904 = "mix.prim.transpose"(%1899) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1905 = "mix.prim.reshape"(%1903) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1906 = "mix.prim.matmul"(%1905, %1904) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1907 = "mix.prim.reshape"(%1906) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1908 = "mix.prim.reshape"(%1907) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1909 = "mix.prim.transpose"(%1900) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1910 = "mix.prim.reshape"(%1903) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1911 = "mix.prim.matmul"(%1910, %1909) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1912 = "mix.prim.reshape"(%1911) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1913 = "mix.prim.reshape"(%1912) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1914 = "mix.prim.slice"(%1913) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1915 = "mix.prim.slice"(%1913) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1916 = "mix.prim.reshape"(%1908) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1917 = "mix.prim.reshape"(%1914) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1918 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1919 = "mix.prim.convert"(%1918) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1920 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %1921 = "mix.prim.div"(%1919, %1920) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %1922 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1923 = "mix.prim.pow"(%1922, %1921) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1924 = "mix.prim.reciprocal"(%1923) : (tensor<80xf16>) -> tensor<80xf16>
    %1925 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1926 = "mix.prim.mul"(%1925, %1924) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1927 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1928 = "mix.prim.unsqueeze"(%1927) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1929 = "mix.prim.permute"(%1928) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1930 = "mix.prim.unsqueeze"(%1926) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1931 = "mix.prim.permute"(%1930) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1932 = "mix.prim.mul"(%1929, %1931) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1933 = "mix.prim.concat"(%1932, %1932) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1934 = "mix.prim.cos"(%1933) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1935 = "mix.prim.slice"(%1934) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1936 = "mix.prim.unsqueeze"(%1935) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1937 = "mix.prim.slice"(%1936) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1938 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1939 = "mix.prim.mul"(%1937, %1938) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1940 = "mix.prim.sin"(%1933) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1941 = "mix.prim.slice"(%1940) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1942 = "mix.prim.unsqueeze"(%1941) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1943 = "mix.prim.slice"(%1942) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1944 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1945 = "mix.prim.mul"(%1943, %1944) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1946 = "mix.prim.slice"(%1939) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1947 = "mix.prim.slice"(%1945) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1948 = "mix.prim.mul"(%1916, %1946) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1949 = "mix.prim.slice"(%1916) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1950 = "mix.prim.slice"(%1916) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1951 = "mix.prim.neg"(%1950) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1952 = "mix.prim.concat"(%1951, %1949) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1953 = "mix.prim.mul"(%1952, %1947) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1954 = "mix.prim.add"(%1948, %1953) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1955 = "mix.prim.mul"(%1917, %1946) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1956 = "mix.prim.slice"(%1917) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1957 = "mix.prim.slice"(%1917) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1958 = "mix.prim.neg"(%1957) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1959 = "mix.prim.concat"(%1958, %1956) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1960 = "mix.prim.mul"(%1959, %1947) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1961 = "mix.prim.add"(%1955, %1960) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1962 = "mix.prim.reshape"(%1954) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1963 = "mix.prim.reshape"(%1961) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1964 = "mix.prim.reshape"(%1962) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1965 = "mix.prim.reshape"(%1963) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1966 = "mix.prim.transpose"(%1964) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1967 = "mix.prim.transpose"(%1965) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1968 = "mix.prim.transpose"(%1967) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1969 = "mix.prim.unsqueeze"(%1966) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1970 = "mix.prim.permute"(%1969) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1971 = "mix.prim.unsqueeze"(%1968) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1972 = "mix.prim.permute"(%1971) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1973 = "mix.prim.permute"(%1970) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1974 = "mix.prim.reshape"(%1973) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1975 = "mix.prim.permute"(%1972) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1976 = "mix.prim.reshape"(%1975) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1977 = "mix.prim.batch_matmul"(%1974, %1976) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1978 = "mix.prim.reshape"(%1977) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1979 = "mix.prim.permute"(%1978) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1980 = "mix.prim.reshape"(%1979) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1981 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1982 = "mix.prim.mul"(%1980, %1981) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1983 = "mix.prim.reshape"(%1982) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1984 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %1985 = "mix.comp.masked_fill"(%1983, %14, %1984) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %1986 = "mix.comp.softmax"(%1985) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1987 = "mix.prim.reshape"(%1986) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1988 = "mix.prim.reshape"(%1915) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1989 = "mix.prim.transpose"(%1988) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1990 = "mix.prim.batch_matmul"(%1987, %1989) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1991 = "mix.prim.reshape"(%1990) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1992 = "mix.prim.permute"(%1991) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1993 = "mix.prim.reshape"(%1992) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1994 = "mix.prim.reshape"(%1993) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1995 = "mix.prim.transpose"(%1901) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1996 = "mix.prim.matmul"(%1994, %1995) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1997 = "mix.prim.add"(%1996, %1902) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1998 = "mix.prim.reshape"(%1997) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1999 = "mix.prim.mul"(%1889, %1998) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2000 = "mix.comp.weight"() <{param_loc = "transformer.h.15.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2001 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2002 = "mix.prim.pow"(%1999, %2001) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2003 = "mix.comp.mean"(%2002) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2004 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2005 = "mix.prim.add"(%2003, %2004) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2006 = "mix.prim.rsqrt"(%2005) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2007 = "mix.prim.mul"(%1999, %2006) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2008 = "mix.prim.mul"(%2000, %2007) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2009 = "mix.module.linear"(%2008) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.15.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2010 = "mix.comp.silu"(%2009) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2011 = "mix.module.linear"(%2008) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.15.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2012 = "mix.prim.mul"(%2010, %2011) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2013 = "mix.module.linear"(%2012) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.15.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2014 = "mix.prim.add"(%2013, %1999) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2015 = "mix.comp.weight"() <{param_loc = "transformer.h.16.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2016 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2017 = "mix.prim.pow"(%2014, %2016) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2018 = "mix.comp.mean"(%2017) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2019 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2020 = "mix.prim.add"(%2018, %2019) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2021 = "mix.prim.rsqrt"(%2020) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2022 = "mix.prim.mul"(%2014, %2021) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2023 = "mix.prim.mul"(%2015, %2022) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2024 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2025 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2026 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2027 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2028 = "mix.prim.transpose"(%2023) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2029 = "mix.prim.transpose"(%2024) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2030 = "mix.prim.reshape"(%2028) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2031 = "mix.prim.matmul"(%2030, %2029) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2032 = "mix.prim.reshape"(%2031) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2033 = "mix.prim.reshape"(%2032) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2034 = "mix.prim.transpose"(%2025) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2035 = "mix.prim.reshape"(%2028) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2036 = "mix.prim.matmul"(%2035, %2034) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2037 = "mix.prim.reshape"(%2036) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2038 = "mix.prim.reshape"(%2037) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2039 = "mix.prim.slice"(%2038) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2040 = "mix.prim.slice"(%2038) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2041 = "mix.prim.reshape"(%2033) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2042 = "mix.prim.reshape"(%2039) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2043 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2044 = "mix.prim.convert"(%2043) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2045 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %2046 = "mix.prim.div"(%2044, %2045) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %2047 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2048 = "mix.prim.pow"(%2047, %2046) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2049 = "mix.prim.reciprocal"(%2048) : (tensor<80xf16>) -> tensor<80xf16>
    %2050 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2051 = "mix.prim.mul"(%2050, %2049) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2052 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2053 = "mix.prim.unsqueeze"(%2052) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2054 = "mix.prim.permute"(%2053) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2055 = "mix.prim.unsqueeze"(%2051) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2056 = "mix.prim.permute"(%2055) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2057 = "mix.prim.mul"(%2054, %2056) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2058 = "mix.prim.concat"(%2057, %2057) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2059 = "mix.prim.cos"(%2058) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2060 = "mix.prim.slice"(%2059) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2061 = "mix.prim.unsqueeze"(%2060) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2062 = "mix.prim.slice"(%2061) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2063 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2064 = "mix.prim.mul"(%2062, %2063) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2065 = "mix.prim.sin"(%2058) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2066 = "mix.prim.slice"(%2065) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2067 = "mix.prim.unsqueeze"(%2066) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2068 = "mix.prim.slice"(%2067) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2069 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2070 = "mix.prim.mul"(%2068, %2069) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2071 = "mix.prim.slice"(%2064) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2072 = "mix.prim.slice"(%2070) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2073 = "mix.prim.mul"(%2041, %2071) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2074 = "mix.prim.slice"(%2041) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2075 = "mix.prim.slice"(%2041) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2076 = "mix.prim.neg"(%2075) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2077 = "mix.prim.concat"(%2076, %2074) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2078 = "mix.prim.mul"(%2077, %2072) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2079 = "mix.prim.add"(%2073, %2078) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2080 = "mix.prim.mul"(%2042, %2071) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2081 = "mix.prim.slice"(%2042) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2082 = "mix.prim.slice"(%2042) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2083 = "mix.prim.neg"(%2082) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2084 = "mix.prim.concat"(%2083, %2081) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2085 = "mix.prim.mul"(%2084, %2072) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2086 = "mix.prim.add"(%2080, %2085) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2087 = "mix.prim.reshape"(%2079) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2088 = "mix.prim.reshape"(%2086) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2089 = "mix.prim.reshape"(%2087) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2090 = "mix.prim.reshape"(%2088) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2091 = "mix.prim.transpose"(%2089) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2092 = "mix.prim.transpose"(%2090) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2093 = "mix.prim.transpose"(%2092) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2094 = "mix.prim.unsqueeze"(%2091) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2095 = "mix.prim.permute"(%2094) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2096 = "mix.prim.unsqueeze"(%2093) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2097 = "mix.prim.permute"(%2096) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2098 = "mix.prim.permute"(%2095) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2099 = "mix.prim.reshape"(%2098) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2100 = "mix.prim.permute"(%2097) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2101 = "mix.prim.reshape"(%2100) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2102 = "mix.prim.batch_matmul"(%2099, %2101) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2103 = "mix.prim.reshape"(%2102) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2104 = "mix.prim.permute"(%2103) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2105 = "mix.prim.reshape"(%2104) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2106 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2107 = "mix.prim.mul"(%2105, %2106) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2108 = "mix.prim.reshape"(%2107) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2109 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2110 = "mix.comp.masked_fill"(%2108, %14, %2109) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2111 = "mix.comp.softmax"(%2110) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2112 = "mix.prim.reshape"(%2111) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2113 = "mix.prim.reshape"(%2040) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2114 = "mix.prim.transpose"(%2113) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2115 = "mix.prim.batch_matmul"(%2112, %2114) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2116 = "mix.prim.reshape"(%2115) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2117 = "mix.prim.permute"(%2116) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2118 = "mix.prim.reshape"(%2117) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2119 = "mix.prim.reshape"(%2118) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2120 = "mix.prim.transpose"(%2026) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2121 = "mix.prim.matmul"(%2119, %2120) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2122 = "mix.prim.add"(%2121, %2027) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2123 = "mix.prim.reshape"(%2122) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2124 = "mix.prim.mul"(%2014, %2123) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2125 = "mix.comp.weight"() <{param_loc = "transformer.h.16.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2126 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2127 = "mix.prim.pow"(%2124, %2126) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2128 = "mix.comp.mean"(%2127) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2129 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2130 = "mix.prim.add"(%2128, %2129) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2131 = "mix.prim.rsqrt"(%2130) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2132 = "mix.prim.mul"(%2124, %2131) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2133 = "mix.prim.mul"(%2125, %2132) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2134 = "mix.module.linear"(%2133) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.16.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2135 = "mix.comp.silu"(%2134) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2136 = "mix.module.linear"(%2133) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.16.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2137 = "mix.prim.mul"(%2135, %2136) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2138 = "mix.module.linear"(%2137) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.16.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2139 = "mix.prim.add"(%2138, %2124) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2140 = "mix.comp.weight"() <{param_loc = "transformer.h.17.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2141 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2142 = "mix.prim.pow"(%2139, %2141) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2143 = "mix.comp.mean"(%2142) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2144 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2145 = "mix.prim.add"(%2143, %2144) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2146 = "mix.prim.rsqrt"(%2145) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2147 = "mix.prim.mul"(%2139, %2146) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2148 = "mix.prim.mul"(%2140, %2147) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2149 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2150 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2151 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2152 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2153 = "mix.prim.transpose"(%2148) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2154 = "mix.prim.transpose"(%2149) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2155 = "mix.prim.reshape"(%2153) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2156 = "mix.prim.matmul"(%2155, %2154) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2157 = "mix.prim.reshape"(%2156) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2158 = "mix.prim.reshape"(%2157) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2159 = "mix.prim.transpose"(%2150) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2160 = "mix.prim.reshape"(%2153) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2161 = "mix.prim.matmul"(%2160, %2159) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2162 = "mix.prim.reshape"(%2161) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2163 = "mix.prim.reshape"(%2162) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2164 = "mix.prim.slice"(%2163) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2165 = "mix.prim.slice"(%2163) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2166 = "mix.prim.reshape"(%2158) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2167 = "mix.prim.reshape"(%2164) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2168 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2169 = "mix.prim.convert"(%2168) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2170 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %2171 = "mix.prim.div"(%2169, %2170) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %2172 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2173 = "mix.prim.pow"(%2172, %2171) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2174 = "mix.prim.reciprocal"(%2173) : (tensor<80xf16>) -> tensor<80xf16>
    %2175 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2176 = "mix.prim.mul"(%2175, %2174) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2177 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2178 = "mix.prim.unsqueeze"(%2177) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2179 = "mix.prim.permute"(%2178) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2180 = "mix.prim.unsqueeze"(%2176) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2181 = "mix.prim.permute"(%2180) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2182 = "mix.prim.mul"(%2179, %2181) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2183 = "mix.prim.concat"(%2182, %2182) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2184 = "mix.prim.cos"(%2183) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2185 = "mix.prim.slice"(%2184) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2186 = "mix.prim.unsqueeze"(%2185) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2187 = "mix.prim.slice"(%2186) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2188 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2189 = "mix.prim.mul"(%2187, %2188) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2190 = "mix.prim.sin"(%2183) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2191 = "mix.prim.slice"(%2190) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2192 = "mix.prim.unsqueeze"(%2191) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2193 = "mix.prim.slice"(%2192) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2194 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2195 = "mix.prim.mul"(%2193, %2194) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2196 = "mix.prim.slice"(%2189) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2197 = "mix.prim.slice"(%2195) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2198 = "mix.prim.mul"(%2166, %2196) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2199 = "mix.prim.slice"(%2166) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2200 = "mix.prim.slice"(%2166) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2201 = "mix.prim.neg"(%2200) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2202 = "mix.prim.concat"(%2201, %2199) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2203 = "mix.prim.mul"(%2202, %2197) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2204 = "mix.prim.add"(%2198, %2203) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2205 = "mix.prim.mul"(%2167, %2196) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2206 = "mix.prim.slice"(%2167) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2207 = "mix.prim.slice"(%2167) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2208 = "mix.prim.neg"(%2207) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2209 = "mix.prim.concat"(%2208, %2206) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2210 = "mix.prim.mul"(%2209, %2197) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2211 = "mix.prim.add"(%2205, %2210) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2212 = "mix.prim.reshape"(%2204) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2213 = "mix.prim.reshape"(%2211) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2214 = "mix.prim.reshape"(%2212) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2215 = "mix.prim.reshape"(%2213) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2216 = "mix.prim.transpose"(%2214) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2217 = "mix.prim.transpose"(%2215) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2218 = "mix.prim.transpose"(%2217) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2219 = "mix.prim.unsqueeze"(%2216) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2220 = "mix.prim.permute"(%2219) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2221 = "mix.prim.unsqueeze"(%2218) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2222 = "mix.prim.permute"(%2221) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2223 = "mix.prim.permute"(%2220) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2224 = "mix.prim.reshape"(%2223) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2225 = "mix.prim.permute"(%2222) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2226 = "mix.prim.reshape"(%2225) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2227 = "mix.prim.batch_matmul"(%2224, %2226) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2228 = "mix.prim.reshape"(%2227) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2229 = "mix.prim.permute"(%2228) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2230 = "mix.prim.reshape"(%2229) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2231 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2232 = "mix.prim.mul"(%2230, %2231) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2233 = "mix.prim.reshape"(%2232) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2234 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2235 = "mix.comp.masked_fill"(%2233, %14, %2234) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2236 = "mix.comp.softmax"(%2235) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2237 = "mix.prim.reshape"(%2236) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2238 = "mix.prim.reshape"(%2165) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2239 = "mix.prim.transpose"(%2238) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2240 = "mix.prim.batch_matmul"(%2237, %2239) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2241 = "mix.prim.reshape"(%2240) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2242 = "mix.prim.permute"(%2241) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2243 = "mix.prim.reshape"(%2242) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2244 = "mix.prim.reshape"(%2243) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2245 = "mix.prim.transpose"(%2151) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2246 = "mix.prim.matmul"(%2244, %2245) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2247 = "mix.prim.add"(%2246, %2152) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2248 = "mix.prim.reshape"(%2247) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2249 = "mix.prim.mul"(%2139, %2248) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2250 = "mix.comp.weight"() <{param_loc = "transformer.h.17.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2251 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2252 = "mix.prim.pow"(%2249, %2251) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2253 = "mix.comp.mean"(%2252) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2254 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2255 = "mix.prim.add"(%2253, %2254) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2256 = "mix.prim.rsqrt"(%2255) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2257 = "mix.prim.mul"(%2249, %2256) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2258 = "mix.prim.mul"(%2250, %2257) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2259 = "mix.module.linear"(%2258) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.17.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2260 = "mix.comp.silu"(%2259) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2261 = "mix.module.linear"(%2258) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.17.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2262 = "mix.prim.mul"(%2260, %2261) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2263 = "mix.module.linear"(%2262) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.17.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2264 = "mix.prim.add"(%2263, %2249) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2265 = "mix.comp.weight"() <{param_loc = "transformer.h.18.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2266 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2267 = "mix.prim.pow"(%2264, %2266) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2268 = "mix.comp.mean"(%2267) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2269 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2270 = "mix.prim.add"(%2268, %2269) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2271 = "mix.prim.rsqrt"(%2270) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2272 = "mix.prim.mul"(%2264, %2271) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2273 = "mix.prim.mul"(%2265, %2272) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2274 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2275 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2276 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2277 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2278 = "mix.prim.transpose"(%2273) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2279 = "mix.prim.transpose"(%2274) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2280 = "mix.prim.reshape"(%2278) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2281 = "mix.prim.matmul"(%2280, %2279) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2282 = "mix.prim.reshape"(%2281) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2283 = "mix.prim.reshape"(%2282) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2284 = "mix.prim.transpose"(%2275) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2285 = "mix.prim.reshape"(%2278) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2286 = "mix.prim.matmul"(%2285, %2284) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2287 = "mix.prim.reshape"(%2286) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2288 = "mix.prim.reshape"(%2287) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2289 = "mix.prim.slice"(%2288) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2290 = "mix.prim.slice"(%2288) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2291 = "mix.prim.reshape"(%2283) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2292 = "mix.prim.reshape"(%2289) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2293 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2294 = "mix.prim.convert"(%2293) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2295 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %2296 = "mix.prim.div"(%2294, %2295) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %2297 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2298 = "mix.prim.pow"(%2297, %2296) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2299 = "mix.prim.reciprocal"(%2298) : (tensor<80xf16>) -> tensor<80xf16>
    %2300 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2301 = "mix.prim.mul"(%2300, %2299) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2302 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2303 = "mix.prim.unsqueeze"(%2302) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2304 = "mix.prim.permute"(%2303) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2305 = "mix.prim.unsqueeze"(%2301) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2306 = "mix.prim.permute"(%2305) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2307 = "mix.prim.mul"(%2304, %2306) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2308 = "mix.prim.concat"(%2307, %2307) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2309 = "mix.prim.cos"(%2308) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2310 = "mix.prim.slice"(%2309) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2311 = "mix.prim.unsqueeze"(%2310) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2312 = "mix.prim.slice"(%2311) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2313 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2314 = "mix.prim.mul"(%2312, %2313) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2315 = "mix.prim.sin"(%2308) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2316 = "mix.prim.slice"(%2315) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2317 = "mix.prim.unsqueeze"(%2316) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2318 = "mix.prim.slice"(%2317) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2319 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2320 = "mix.prim.mul"(%2318, %2319) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2321 = "mix.prim.slice"(%2314) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2322 = "mix.prim.slice"(%2320) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2323 = "mix.prim.mul"(%2291, %2321) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2324 = "mix.prim.slice"(%2291) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2325 = "mix.prim.slice"(%2291) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2326 = "mix.prim.neg"(%2325) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2327 = "mix.prim.concat"(%2326, %2324) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2328 = "mix.prim.mul"(%2327, %2322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2329 = "mix.prim.add"(%2323, %2328) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2330 = "mix.prim.mul"(%2292, %2321) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2331 = "mix.prim.slice"(%2292) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2332 = "mix.prim.slice"(%2292) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2333 = "mix.prim.neg"(%2332) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2334 = "mix.prim.concat"(%2333, %2331) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2335 = "mix.prim.mul"(%2334, %2322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2336 = "mix.prim.add"(%2330, %2335) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2337 = "mix.prim.reshape"(%2329) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2338 = "mix.prim.reshape"(%2336) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2339 = "mix.prim.reshape"(%2337) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2340 = "mix.prim.reshape"(%2338) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2341 = "mix.prim.transpose"(%2339) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2342 = "mix.prim.transpose"(%2340) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2343 = "mix.prim.transpose"(%2342) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2344 = "mix.prim.unsqueeze"(%2341) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2345 = "mix.prim.permute"(%2344) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2346 = "mix.prim.unsqueeze"(%2343) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2347 = "mix.prim.permute"(%2346) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2348 = "mix.prim.permute"(%2345) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2349 = "mix.prim.reshape"(%2348) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2350 = "mix.prim.permute"(%2347) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2351 = "mix.prim.reshape"(%2350) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2352 = "mix.prim.batch_matmul"(%2349, %2351) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2353 = "mix.prim.reshape"(%2352) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2354 = "mix.prim.permute"(%2353) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2355 = "mix.prim.reshape"(%2354) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2356 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2357 = "mix.prim.mul"(%2355, %2356) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2358 = "mix.prim.reshape"(%2357) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2359 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2360 = "mix.comp.masked_fill"(%2358, %14, %2359) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2361 = "mix.comp.softmax"(%2360) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2362 = "mix.prim.reshape"(%2361) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2363 = "mix.prim.reshape"(%2290) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2364 = "mix.prim.transpose"(%2363) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2365 = "mix.prim.batch_matmul"(%2362, %2364) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2366 = "mix.prim.reshape"(%2365) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2367 = "mix.prim.permute"(%2366) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2368 = "mix.prim.reshape"(%2367) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2369 = "mix.prim.reshape"(%2368) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2370 = "mix.prim.transpose"(%2276) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2371 = "mix.prim.matmul"(%2369, %2370) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2372 = "mix.prim.add"(%2371, %2277) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2373 = "mix.prim.reshape"(%2372) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2374 = "mix.prim.mul"(%2264, %2373) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2375 = "mix.comp.weight"() <{param_loc = "transformer.h.18.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2376 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2377 = "mix.prim.pow"(%2374, %2376) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2378 = "mix.comp.mean"(%2377) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2379 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2380 = "mix.prim.add"(%2378, %2379) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2381 = "mix.prim.rsqrt"(%2380) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2382 = "mix.prim.mul"(%2374, %2381) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2383 = "mix.prim.mul"(%2375, %2382) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2384 = "mix.module.linear"(%2383) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.18.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2385 = "mix.comp.silu"(%2384) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2386 = "mix.module.linear"(%2383) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.18.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2387 = "mix.prim.mul"(%2385, %2386) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2388 = "mix.module.linear"(%2387) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.18.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2389 = "mix.prim.add"(%2388, %2374) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2390 = "mix.comp.weight"() <{param_loc = "transformer.h.19.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2391 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2392 = "mix.prim.pow"(%2389, %2391) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2393 = "mix.comp.mean"(%2392) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2394 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2395 = "mix.prim.add"(%2393, %2394) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2396 = "mix.prim.rsqrt"(%2395) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2397 = "mix.prim.mul"(%2389, %2396) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2398 = "mix.prim.mul"(%2390, %2397) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2399 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2400 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2401 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2402 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2403 = "mix.prim.transpose"(%2398) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2404 = "mix.prim.transpose"(%2399) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2405 = "mix.prim.reshape"(%2403) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2406 = "mix.prim.matmul"(%2405, %2404) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2407 = "mix.prim.reshape"(%2406) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2408 = "mix.prim.reshape"(%2407) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2409 = "mix.prim.transpose"(%2400) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2410 = "mix.prim.reshape"(%2403) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2411 = "mix.prim.matmul"(%2410, %2409) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2412 = "mix.prim.reshape"(%2411) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2413 = "mix.prim.reshape"(%2412) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2414 = "mix.prim.slice"(%2413) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2415 = "mix.prim.slice"(%2413) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2416 = "mix.prim.reshape"(%2408) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2417 = "mix.prim.reshape"(%2414) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2418 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2419 = "mix.prim.convert"(%2418) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2420 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %2421 = "mix.prim.div"(%2419, %2420) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %2422 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2423 = "mix.prim.pow"(%2422, %2421) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2424 = "mix.prim.reciprocal"(%2423) : (tensor<80xf16>) -> tensor<80xf16>
    %2425 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2426 = "mix.prim.mul"(%2425, %2424) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2427 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2428 = "mix.prim.unsqueeze"(%2427) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2429 = "mix.prim.permute"(%2428) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2430 = "mix.prim.unsqueeze"(%2426) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2431 = "mix.prim.permute"(%2430) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2432 = "mix.prim.mul"(%2429, %2431) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2433 = "mix.prim.concat"(%2432, %2432) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2434 = "mix.prim.cos"(%2433) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2435 = "mix.prim.slice"(%2434) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2436 = "mix.prim.unsqueeze"(%2435) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2437 = "mix.prim.slice"(%2436) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2438 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2439 = "mix.prim.mul"(%2437, %2438) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2440 = "mix.prim.sin"(%2433) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2441 = "mix.prim.slice"(%2440) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2442 = "mix.prim.unsqueeze"(%2441) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2443 = "mix.prim.slice"(%2442) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2444 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2445 = "mix.prim.mul"(%2443, %2444) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2446 = "mix.prim.slice"(%2439) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2447 = "mix.prim.slice"(%2445) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2448 = "mix.prim.mul"(%2416, %2446) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2449 = "mix.prim.slice"(%2416) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2450 = "mix.prim.slice"(%2416) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2451 = "mix.prim.neg"(%2450) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2452 = "mix.prim.concat"(%2451, %2449) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2453 = "mix.prim.mul"(%2452, %2447) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2454 = "mix.prim.add"(%2448, %2453) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2455 = "mix.prim.mul"(%2417, %2446) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2456 = "mix.prim.slice"(%2417) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2457 = "mix.prim.slice"(%2417) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2458 = "mix.prim.neg"(%2457) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2459 = "mix.prim.concat"(%2458, %2456) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2460 = "mix.prim.mul"(%2459, %2447) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2461 = "mix.prim.add"(%2455, %2460) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2462 = "mix.prim.reshape"(%2454) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2463 = "mix.prim.reshape"(%2461) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2464 = "mix.prim.reshape"(%2462) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2465 = "mix.prim.reshape"(%2463) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2466 = "mix.prim.transpose"(%2464) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2467 = "mix.prim.transpose"(%2465) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2468 = "mix.prim.transpose"(%2467) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2469 = "mix.prim.unsqueeze"(%2466) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2470 = "mix.prim.permute"(%2469) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2471 = "mix.prim.unsqueeze"(%2468) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2472 = "mix.prim.permute"(%2471) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2473 = "mix.prim.permute"(%2470) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2474 = "mix.prim.reshape"(%2473) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2475 = "mix.prim.permute"(%2472) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2476 = "mix.prim.reshape"(%2475) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2477 = "mix.prim.batch_matmul"(%2474, %2476) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2478 = "mix.prim.reshape"(%2477) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2479 = "mix.prim.permute"(%2478) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2480 = "mix.prim.reshape"(%2479) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2481 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2482 = "mix.prim.mul"(%2480, %2481) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2483 = "mix.prim.reshape"(%2482) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2484 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2485 = "mix.comp.masked_fill"(%2483, %14, %2484) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2486 = "mix.comp.softmax"(%2485) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2487 = "mix.prim.reshape"(%2486) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2488 = "mix.prim.reshape"(%2415) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2489 = "mix.prim.transpose"(%2488) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2490 = "mix.prim.batch_matmul"(%2487, %2489) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2491 = "mix.prim.reshape"(%2490) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2492 = "mix.prim.permute"(%2491) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2493 = "mix.prim.reshape"(%2492) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2494 = "mix.prim.reshape"(%2493) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2495 = "mix.prim.transpose"(%2401) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2496 = "mix.prim.matmul"(%2494, %2495) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2497 = "mix.prim.add"(%2496, %2402) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2498 = "mix.prim.reshape"(%2497) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2499 = "mix.prim.mul"(%2389, %2498) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2500 = "mix.comp.weight"() <{param_loc = "transformer.h.19.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2501 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2502 = "mix.prim.pow"(%2499, %2501) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2503 = "mix.comp.mean"(%2502) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2504 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2505 = "mix.prim.add"(%2503, %2504) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2506 = "mix.prim.rsqrt"(%2505) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2507 = "mix.prim.mul"(%2499, %2506) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2508 = "mix.prim.mul"(%2500, %2507) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2509 = "mix.module.linear"(%2508) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.19.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2510 = "mix.comp.silu"(%2509) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2511 = "mix.module.linear"(%2508) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.19.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2512 = "mix.prim.mul"(%2510, %2511) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2513 = "mix.module.linear"(%2512) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.19.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2514 = "mix.prim.add"(%2513, %2499) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2515 = "mix.comp.weight"() <{param_loc = "transformer.h.20.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2516 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2517 = "mix.prim.pow"(%2514, %2516) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2518 = "mix.comp.mean"(%2517) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2519 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2520 = "mix.prim.add"(%2518, %2519) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2521 = "mix.prim.rsqrt"(%2520) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2522 = "mix.prim.mul"(%2514, %2521) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2523 = "mix.prim.mul"(%2515, %2522) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2524 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2525 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2526 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2527 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2528 = "mix.prim.transpose"(%2523) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2529 = "mix.prim.transpose"(%2524) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2530 = "mix.prim.reshape"(%2528) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2531 = "mix.prim.matmul"(%2530, %2529) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2532 = "mix.prim.reshape"(%2531) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2533 = "mix.prim.reshape"(%2532) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2534 = "mix.prim.transpose"(%2525) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2535 = "mix.prim.reshape"(%2528) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2536 = "mix.prim.matmul"(%2535, %2534) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2537 = "mix.prim.reshape"(%2536) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2538 = "mix.prim.reshape"(%2537) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2539 = "mix.prim.slice"(%2538) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2540 = "mix.prim.slice"(%2538) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2541 = "mix.prim.reshape"(%2533) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2542 = "mix.prim.reshape"(%2539) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2543 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2544 = "mix.prim.convert"(%2543) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2545 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %2546 = "mix.prim.div"(%2544, %2545) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %2547 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2548 = "mix.prim.pow"(%2547, %2546) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2549 = "mix.prim.reciprocal"(%2548) : (tensor<80xf16>) -> tensor<80xf16>
    %2550 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2551 = "mix.prim.mul"(%2550, %2549) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2552 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2553 = "mix.prim.unsqueeze"(%2552) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2554 = "mix.prim.permute"(%2553) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2555 = "mix.prim.unsqueeze"(%2551) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2556 = "mix.prim.permute"(%2555) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2557 = "mix.prim.mul"(%2554, %2556) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2558 = "mix.prim.concat"(%2557, %2557) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2559 = "mix.prim.cos"(%2558) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2560 = "mix.prim.slice"(%2559) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2561 = "mix.prim.unsqueeze"(%2560) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2562 = "mix.prim.slice"(%2561) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2563 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2564 = "mix.prim.mul"(%2562, %2563) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2565 = "mix.prim.sin"(%2558) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2566 = "mix.prim.slice"(%2565) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2567 = "mix.prim.unsqueeze"(%2566) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2568 = "mix.prim.slice"(%2567) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2569 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2570 = "mix.prim.mul"(%2568, %2569) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2571 = "mix.prim.slice"(%2564) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2572 = "mix.prim.slice"(%2570) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2573 = "mix.prim.mul"(%2541, %2571) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2574 = "mix.prim.slice"(%2541) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2575 = "mix.prim.slice"(%2541) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2576 = "mix.prim.neg"(%2575) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2577 = "mix.prim.concat"(%2576, %2574) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2578 = "mix.prim.mul"(%2577, %2572) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2579 = "mix.prim.add"(%2573, %2578) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2580 = "mix.prim.mul"(%2542, %2571) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2581 = "mix.prim.slice"(%2542) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2582 = "mix.prim.slice"(%2542) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2583 = "mix.prim.neg"(%2582) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2584 = "mix.prim.concat"(%2583, %2581) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2585 = "mix.prim.mul"(%2584, %2572) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2586 = "mix.prim.add"(%2580, %2585) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2587 = "mix.prim.reshape"(%2579) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2588 = "mix.prim.reshape"(%2586) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2589 = "mix.prim.reshape"(%2587) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2590 = "mix.prim.reshape"(%2588) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2591 = "mix.prim.transpose"(%2589) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2592 = "mix.prim.transpose"(%2590) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2593 = "mix.prim.transpose"(%2592) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2594 = "mix.prim.unsqueeze"(%2591) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2595 = "mix.prim.permute"(%2594) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2596 = "mix.prim.unsqueeze"(%2593) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2597 = "mix.prim.permute"(%2596) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2598 = "mix.prim.permute"(%2595) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2599 = "mix.prim.reshape"(%2598) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2600 = "mix.prim.permute"(%2597) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2601 = "mix.prim.reshape"(%2600) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2602 = "mix.prim.batch_matmul"(%2599, %2601) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2603 = "mix.prim.reshape"(%2602) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2604 = "mix.prim.permute"(%2603) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2605 = "mix.prim.reshape"(%2604) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2606 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2607 = "mix.prim.mul"(%2605, %2606) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2608 = "mix.prim.reshape"(%2607) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2609 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2610 = "mix.comp.masked_fill"(%2608, %14, %2609) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2611 = "mix.comp.softmax"(%2610) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2612 = "mix.prim.reshape"(%2611) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2613 = "mix.prim.reshape"(%2540) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2614 = "mix.prim.transpose"(%2613) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2615 = "mix.prim.batch_matmul"(%2612, %2614) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2616 = "mix.prim.reshape"(%2615) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2617 = "mix.prim.permute"(%2616) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2618 = "mix.prim.reshape"(%2617) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2619 = "mix.prim.reshape"(%2618) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2620 = "mix.prim.transpose"(%2526) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2621 = "mix.prim.matmul"(%2619, %2620) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2622 = "mix.prim.add"(%2621, %2527) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2623 = "mix.prim.reshape"(%2622) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2624 = "mix.prim.mul"(%2514, %2623) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2625 = "mix.comp.weight"() <{param_loc = "transformer.h.20.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2626 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2627 = "mix.prim.pow"(%2624, %2626) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2628 = "mix.comp.mean"(%2627) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2629 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2630 = "mix.prim.add"(%2628, %2629) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2631 = "mix.prim.rsqrt"(%2630) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2632 = "mix.prim.mul"(%2624, %2631) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2633 = "mix.prim.mul"(%2625, %2632) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2634 = "mix.module.linear"(%2633) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.20.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2635 = "mix.comp.silu"(%2634) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2636 = "mix.module.linear"(%2633) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.20.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2637 = "mix.prim.mul"(%2635, %2636) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2638 = "mix.module.linear"(%2637) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.20.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2639 = "mix.prim.add"(%2638, %2624) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2640 = "mix.comp.weight"() <{param_loc = "transformer.h.21.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2641 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2642 = "mix.prim.pow"(%2639, %2641) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2643 = "mix.comp.mean"(%2642) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2644 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2645 = "mix.prim.add"(%2643, %2644) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2646 = "mix.prim.rsqrt"(%2645) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2647 = "mix.prim.mul"(%2639, %2646) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2648 = "mix.prim.mul"(%2640, %2647) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2649 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2650 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2651 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2652 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2653 = "mix.prim.transpose"(%2648) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2654 = "mix.prim.transpose"(%2649) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2655 = "mix.prim.reshape"(%2653) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2656 = "mix.prim.matmul"(%2655, %2654) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2657 = "mix.prim.reshape"(%2656) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2658 = "mix.prim.reshape"(%2657) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2659 = "mix.prim.transpose"(%2650) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2660 = "mix.prim.reshape"(%2653) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2661 = "mix.prim.matmul"(%2660, %2659) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2662 = "mix.prim.reshape"(%2661) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2663 = "mix.prim.reshape"(%2662) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2664 = "mix.prim.slice"(%2663) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2665 = "mix.prim.slice"(%2663) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2666 = "mix.prim.reshape"(%2658) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2667 = "mix.prim.reshape"(%2664) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2668 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2669 = "mix.prim.convert"(%2668) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2670 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %2671 = "mix.prim.div"(%2669, %2670) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %2672 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2673 = "mix.prim.pow"(%2672, %2671) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2674 = "mix.prim.reciprocal"(%2673) : (tensor<80xf16>) -> tensor<80xf16>
    %2675 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2676 = "mix.prim.mul"(%2675, %2674) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2677 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2678 = "mix.prim.unsqueeze"(%2677) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2679 = "mix.prim.permute"(%2678) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2680 = "mix.prim.unsqueeze"(%2676) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2681 = "mix.prim.permute"(%2680) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2682 = "mix.prim.mul"(%2679, %2681) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2683 = "mix.prim.concat"(%2682, %2682) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2684 = "mix.prim.cos"(%2683) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2685 = "mix.prim.slice"(%2684) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2686 = "mix.prim.unsqueeze"(%2685) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2687 = "mix.prim.slice"(%2686) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2688 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2689 = "mix.prim.mul"(%2687, %2688) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2690 = "mix.prim.sin"(%2683) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2691 = "mix.prim.slice"(%2690) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2692 = "mix.prim.unsqueeze"(%2691) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2693 = "mix.prim.slice"(%2692) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2694 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2695 = "mix.prim.mul"(%2693, %2694) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2696 = "mix.prim.slice"(%2689) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2697 = "mix.prim.slice"(%2695) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2698 = "mix.prim.mul"(%2666, %2696) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2699 = "mix.prim.slice"(%2666) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2700 = "mix.prim.slice"(%2666) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2701 = "mix.prim.neg"(%2700) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2702 = "mix.prim.concat"(%2701, %2699) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2703 = "mix.prim.mul"(%2702, %2697) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2704 = "mix.prim.add"(%2698, %2703) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2705 = "mix.prim.mul"(%2667, %2696) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2706 = "mix.prim.slice"(%2667) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2707 = "mix.prim.slice"(%2667) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2708 = "mix.prim.neg"(%2707) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2709 = "mix.prim.concat"(%2708, %2706) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2710 = "mix.prim.mul"(%2709, %2697) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2711 = "mix.prim.add"(%2705, %2710) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2712 = "mix.prim.reshape"(%2704) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2713 = "mix.prim.reshape"(%2711) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2714 = "mix.prim.reshape"(%2712) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2715 = "mix.prim.reshape"(%2713) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2716 = "mix.prim.transpose"(%2714) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2717 = "mix.prim.transpose"(%2715) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2718 = "mix.prim.transpose"(%2717) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2719 = "mix.prim.unsqueeze"(%2716) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2720 = "mix.prim.permute"(%2719) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2721 = "mix.prim.unsqueeze"(%2718) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2722 = "mix.prim.permute"(%2721) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2723 = "mix.prim.permute"(%2720) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2724 = "mix.prim.reshape"(%2723) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2725 = "mix.prim.permute"(%2722) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2726 = "mix.prim.reshape"(%2725) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2727 = "mix.prim.batch_matmul"(%2724, %2726) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2728 = "mix.prim.reshape"(%2727) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2729 = "mix.prim.permute"(%2728) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2730 = "mix.prim.reshape"(%2729) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2731 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2732 = "mix.prim.mul"(%2730, %2731) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2733 = "mix.prim.reshape"(%2732) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2734 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2735 = "mix.comp.masked_fill"(%2733, %14, %2734) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2736 = "mix.comp.softmax"(%2735) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2737 = "mix.prim.reshape"(%2736) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2738 = "mix.prim.reshape"(%2665) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2739 = "mix.prim.transpose"(%2738) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2740 = "mix.prim.batch_matmul"(%2737, %2739) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2741 = "mix.prim.reshape"(%2740) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2742 = "mix.prim.permute"(%2741) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2743 = "mix.prim.reshape"(%2742) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2744 = "mix.prim.reshape"(%2743) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2745 = "mix.prim.transpose"(%2651) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2746 = "mix.prim.matmul"(%2744, %2745) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2747 = "mix.prim.add"(%2746, %2652) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2748 = "mix.prim.reshape"(%2747) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2749 = "mix.prim.mul"(%2639, %2748) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2750 = "mix.comp.weight"() <{param_loc = "transformer.h.21.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2751 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2752 = "mix.prim.pow"(%2749, %2751) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2753 = "mix.comp.mean"(%2752) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2754 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2755 = "mix.prim.add"(%2753, %2754) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2756 = "mix.prim.rsqrt"(%2755) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2757 = "mix.prim.mul"(%2749, %2756) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2758 = "mix.prim.mul"(%2750, %2757) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2759 = "mix.module.linear"(%2758) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.21.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2760 = "mix.comp.silu"(%2759) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2761 = "mix.module.linear"(%2758) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.21.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2762 = "mix.prim.mul"(%2760, %2761) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2763 = "mix.module.linear"(%2762) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.21.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2764 = "mix.prim.add"(%2763, %2749) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2765 = "mix.comp.weight"() <{param_loc = "transformer.h.22.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2766 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2767 = "mix.prim.pow"(%2764, %2766) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2768 = "mix.comp.mean"(%2767) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2769 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2770 = "mix.prim.add"(%2768, %2769) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2771 = "mix.prim.rsqrt"(%2770) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2772 = "mix.prim.mul"(%2764, %2771) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2773 = "mix.prim.mul"(%2765, %2772) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2774 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2775 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2776 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2777 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2778 = "mix.prim.transpose"(%2773) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2779 = "mix.prim.transpose"(%2774) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2780 = "mix.prim.reshape"(%2778) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2781 = "mix.prim.matmul"(%2780, %2779) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2782 = "mix.prim.reshape"(%2781) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2783 = "mix.prim.reshape"(%2782) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2784 = "mix.prim.transpose"(%2775) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2785 = "mix.prim.reshape"(%2778) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2786 = "mix.prim.matmul"(%2785, %2784) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2787 = "mix.prim.reshape"(%2786) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2788 = "mix.prim.reshape"(%2787) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2789 = "mix.prim.slice"(%2788) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2790 = "mix.prim.slice"(%2788) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2791 = "mix.prim.reshape"(%2783) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2792 = "mix.prim.reshape"(%2789) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2793 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2794 = "mix.prim.convert"(%2793) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2795 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %2796 = "mix.prim.div"(%2794, %2795) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %2797 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2798 = "mix.prim.pow"(%2797, %2796) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2799 = "mix.prim.reciprocal"(%2798) : (tensor<80xf16>) -> tensor<80xf16>
    %2800 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2801 = "mix.prim.mul"(%2800, %2799) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2802 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2803 = "mix.prim.unsqueeze"(%2802) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2804 = "mix.prim.permute"(%2803) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2805 = "mix.prim.unsqueeze"(%2801) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2806 = "mix.prim.permute"(%2805) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2807 = "mix.prim.mul"(%2804, %2806) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2808 = "mix.prim.concat"(%2807, %2807) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2809 = "mix.prim.cos"(%2808) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2810 = "mix.prim.slice"(%2809) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2811 = "mix.prim.unsqueeze"(%2810) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2812 = "mix.prim.slice"(%2811) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2813 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2814 = "mix.prim.mul"(%2812, %2813) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2815 = "mix.prim.sin"(%2808) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2816 = "mix.prim.slice"(%2815) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2817 = "mix.prim.unsqueeze"(%2816) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2818 = "mix.prim.slice"(%2817) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2819 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2820 = "mix.prim.mul"(%2818, %2819) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2821 = "mix.prim.slice"(%2814) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2822 = "mix.prim.slice"(%2820) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2823 = "mix.prim.mul"(%2791, %2821) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2824 = "mix.prim.slice"(%2791) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2825 = "mix.prim.slice"(%2791) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2826 = "mix.prim.neg"(%2825) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2827 = "mix.prim.concat"(%2826, %2824) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2828 = "mix.prim.mul"(%2827, %2822) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2829 = "mix.prim.add"(%2823, %2828) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2830 = "mix.prim.mul"(%2792, %2821) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2831 = "mix.prim.slice"(%2792) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2832 = "mix.prim.slice"(%2792) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2833 = "mix.prim.neg"(%2832) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2834 = "mix.prim.concat"(%2833, %2831) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2835 = "mix.prim.mul"(%2834, %2822) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2836 = "mix.prim.add"(%2830, %2835) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2837 = "mix.prim.reshape"(%2829) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2838 = "mix.prim.reshape"(%2836) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2839 = "mix.prim.reshape"(%2837) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2840 = "mix.prim.reshape"(%2838) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2841 = "mix.prim.transpose"(%2839) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2842 = "mix.prim.transpose"(%2840) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2843 = "mix.prim.transpose"(%2842) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2844 = "mix.prim.unsqueeze"(%2841) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2845 = "mix.prim.permute"(%2844) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2846 = "mix.prim.unsqueeze"(%2843) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2847 = "mix.prim.permute"(%2846) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2848 = "mix.prim.permute"(%2845) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2849 = "mix.prim.reshape"(%2848) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2850 = "mix.prim.permute"(%2847) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2851 = "mix.prim.reshape"(%2850) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2852 = "mix.prim.batch_matmul"(%2849, %2851) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2853 = "mix.prim.reshape"(%2852) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2854 = "mix.prim.permute"(%2853) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2855 = "mix.prim.reshape"(%2854) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2856 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2857 = "mix.prim.mul"(%2855, %2856) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2858 = "mix.prim.reshape"(%2857) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2859 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2860 = "mix.comp.masked_fill"(%2858, %14, %2859) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2861 = "mix.comp.softmax"(%2860) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2862 = "mix.prim.reshape"(%2861) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2863 = "mix.prim.reshape"(%2790) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2864 = "mix.prim.transpose"(%2863) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2865 = "mix.prim.batch_matmul"(%2862, %2864) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2866 = "mix.prim.reshape"(%2865) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2867 = "mix.prim.permute"(%2866) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2868 = "mix.prim.reshape"(%2867) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2869 = "mix.prim.reshape"(%2868) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2870 = "mix.prim.transpose"(%2776) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2871 = "mix.prim.matmul"(%2869, %2870) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2872 = "mix.prim.add"(%2871, %2777) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2873 = "mix.prim.reshape"(%2872) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2874 = "mix.prim.mul"(%2764, %2873) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2875 = "mix.comp.weight"() <{param_loc = "transformer.h.22.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2876 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2877 = "mix.prim.pow"(%2874, %2876) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2878 = "mix.comp.mean"(%2877) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2879 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2880 = "mix.prim.add"(%2878, %2879) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2881 = "mix.prim.rsqrt"(%2880) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2882 = "mix.prim.mul"(%2874, %2881) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2883 = "mix.prim.mul"(%2875, %2882) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2884 = "mix.module.linear"(%2883) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.22.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2885 = "mix.comp.silu"(%2884) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2886 = "mix.module.linear"(%2883) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.22.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2887 = "mix.prim.mul"(%2885, %2886) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2888 = "mix.module.linear"(%2887) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.22.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2889 = "mix.prim.add"(%2888, %2874) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2890 = "mix.comp.weight"() <{param_loc = "transformer.h.23.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2891 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2892 = "mix.prim.pow"(%2889, %2891) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2893 = "mix.comp.mean"(%2892) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2894 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2895 = "mix.prim.add"(%2893, %2894) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2896 = "mix.prim.rsqrt"(%2895) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2897 = "mix.prim.mul"(%2889, %2896) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2898 = "mix.prim.mul"(%2890, %2897) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2899 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2900 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2901 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2902 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2903 = "mix.prim.transpose"(%2898) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2904 = "mix.prim.transpose"(%2899) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2905 = "mix.prim.reshape"(%2903) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2906 = "mix.prim.matmul"(%2905, %2904) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2907 = "mix.prim.reshape"(%2906) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2908 = "mix.prim.reshape"(%2907) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2909 = "mix.prim.transpose"(%2900) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2910 = "mix.prim.reshape"(%2903) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2911 = "mix.prim.matmul"(%2910, %2909) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2912 = "mix.prim.reshape"(%2911) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2913 = "mix.prim.reshape"(%2912) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2914 = "mix.prim.slice"(%2913) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2915 = "mix.prim.slice"(%2913) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2916 = "mix.prim.reshape"(%2908) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2917 = "mix.prim.reshape"(%2914) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2918 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2919 = "mix.prim.convert"(%2918) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2920 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %2921 = "mix.prim.div"(%2919, %2920) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %2922 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2923 = "mix.prim.pow"(%2922, %2921) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2924 = "mix.prim.reciprocal"(%2923) : (tensor<80xf16>) -> tensor<80xf16>
    %2925 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2926 = "mix.prim.mul"(%2925, %2924) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2927 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2928 = "mix.prim.unsqueeze"(%2927) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2929 = "mix.prim.permute"(%2928) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2930 = "mix.prim.unsqueeze"(%2926) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2931 = "mix.prim.permute"(%2930) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2932 = "mix.prim.mul"(%2929, %2931) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2933 = "mix.prim.concat"(%2932, %2932) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2934 = "mix.prim.cos"(%2933) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2935 = "mix.prim.slice"(%2934) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2936 = "mix.prim.unsqueeze"(%2935) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2937 = "mix.prim.slice"(%2936) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2938 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2939 = "mix.prim.mul"(%2937, %2938) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2940 = "mix.prim.sin"(%2933) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2941 = "mix.prim.slice"(%2940) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2942 = "mix.prim.unsqueeze"(%2941) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2943 = "mix.prim.slice"(%2942) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2944 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2945 = "mix.prim.mul"(%2943, %2944) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2946 = "mix.prim.slice"(%2939) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2947 = "mix.prim.slice"(%2945) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2948 = "mix.prim.mul"(%2916, %2946) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2949 = "mix.prim.slice"(%2916) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2950 = "mix.prim.slice"(%2916) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2951 = "mix.prim.neg"(%2950) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2952 = "mix.prim.concat"(%2951, %2949) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2953 = "mix.prim.mul"(%2952, %2947) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2954 = "mix.prim.add"(%2948, %2953) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2955 = "mix.prim.mul"(%2917, %2946) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2956 = "mix.prim.slice"(%2917) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2957 = "mix.prim.slice"(%2917) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2958 = "mix.prim.neg"(%2957) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2959 = "mix.prim.concat"(%2958, %2956) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2960 = "mix.prim.mul"(%2959, %2947) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2961 = "mix.prim.add"(%2955, %2960) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2962 = "mix.prim.reshape"(%2954) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2963 = "mix.prim.reshape"(%2961) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2964 = "mix.prim.reshape"(%2962) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2965 = "mix.prim.reshape"(%2963) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2966 = "mix.prim.transpose"(%2964) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2967 = "mix.prim.transpose"(%2965) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2968 = "mix.prim.transpose"(%2967) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2969 = "mix.prim.unsqueeze"(%2966) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2970 = "mix.prim.permute"(%2969) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2971 = "mix.prim.unsqueeze"(%2968) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2972 = "mix.prim.permute"(%2971) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2973 = "mix.prim.permute"(%2970) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2974 = "mix.prim.reshape"(%2973) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2975 = "mix.prim.permute"(%2972) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2976 = "mix.prim.reshape"(%2975) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2977 = "mix.prim.batch_matmul"(%2974, %2976) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2978 = "mix.prim.reshape"(%2977) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2979 = "mix.prim.permute"(%2978) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2980 = "mix.prim.reshape"(%2979) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2981 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2982 = "mix.prim.mul"(%2980, %2981) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2983 = "mix.prim.reshape"(%2982) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2984 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %2985 = "mix.comp.masked_fill"(%2983, %14, %2984) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %2986 = "mix.comp.softmax"(%2985) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2987 = "mix.prim.reshape"(%2986) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2988 = "mix.prim.reshape"(%2915) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2989 = "mix.prim.transpose"(%2988) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2990 = "mix.prim.batch_matmul"(%2987, %2989) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2991 = "mix.prim.reshape"(%2990) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2992 = "mix.prim.permute"(%2991) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2993 = "mix.prim.reshape"(%2992) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2994 = "mix.prim.reshape"(%2993) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2995 = "mix.prim.transpose"(%2901) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2996 = "mix.prim.matmul"(%2994, %2995) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2997 = "mix.prim.add"(%2996, %2902) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2998 = "mix.prim.reshape"(%2997) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2999 = "mix.prim.mul"(%2889, %2998) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3000 = "mix.comp.weight"() <{param_loc = "transformer.h.23.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3001 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3002 = "mix.prim.pow"(%2999, %3001) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3003 = "mix.comp.mean"(%3002) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3004 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3005 = "mix.prim.add"(%3003, %3004) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3006 = "mix.prim.rsqrt"(%3005) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3007 = "mix.prim.mul"(%2999, %3006) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3008 = "mix.prim.mul"(%3000, %3007) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3009 = "mix.module.linear"(%3008) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.23.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3010 = "mix.comp.silu"(%3009) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3011 = "mix.module.linear"(%3008) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.23.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3012 = "mix.prim.mul"(%3010, %3011) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3013 = "mix.module.linear"(%3012) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.23.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3014 = "mix.prim.add"(%3013, %2999) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3015 = "mix.comp.weight"() <{param_loc = "transformer.h.24.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3016 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3017 = "mix.prim.pow"(%3014, %3016) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3018 = "mix.comp.mean"(%3017) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3019 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3020 = "mix.prim.add"(%3018, %3019) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3021 = "mix.prim.rsqrt"(%3020) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3022 = "mix.prim.mul"(%3014, %3021) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3023 = "mix.prim.mul"(%3015, %3022) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3024 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3025 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3026 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3027 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3028 = "mix.prim.transpose"(%3023) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3029 = "mix.prim.transpose"(%3024) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3030 = "mix.prim.reshape"(%3028) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3031 = "mix.prim.matmul"(%3030, %3029) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3032 = "mix.prim.reshape"(%3031) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3033 = "mix.prim.reshape"(%3032) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3034 = "mix.prim.transpose"(%3025) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3035 = "mix.prim.reshape"(%3028) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3036 = "mix.prim.matmul"(%3035, %3034) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3037 = "mix.prim.reshape"(%3036) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3038 = "mix.prim.reshape"(%3037) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3039 = "mix.prim.slice"(%3038) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3040 = "mix.prim.slice"(%3038) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3041 = "mix.prim.reshape"(%3033) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3042 = "mix.prim.reshape"(%3039) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3043 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3044 = "mix.prim.convert"(%3043) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3045 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %3046 = "mix.prim.div"(%3044, %3045) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %3047 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3048 = "mix.prim.pow"(%3047, %3046) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3049 = "mix.prim.reciprocal"(%3048) : (tensor<80xf16>) -> tensor<80xf16>
    %3050 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3051 = "mix.prim.mul"(%3050, %3049) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3052 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3053 = "mix.prim.unsqueeze"(%3052) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3054 = "mix.prim.permute"(%3053) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3055 = "mix.prim.unsqueeze"(%3051) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3056 = "mix.prim.permute"(%3055) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3057 = "mix.prim.mul"(%3054, %3056) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3058 = "mix.prim.concat"(%3057, %3057) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3059 = "mix.prim.cos"(%3058) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3060 = "mix.prim.slice"(%3059) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3061 = "mix.prim.unsqueeze"(%3060) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3062 = "mix.prim.slice"(%3061) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3063 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3064 = "mix.prim.mul"(%3062, %3063) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3065 = "mix.prim.sin"(%3058) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3066 = "mix.prim.slice"(%3065) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3067 = "mix.prim.unsqueeze"(%3066) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3068 = "mix.prim.slice"(%3067) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3069 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3070 = "mix.prim.mul"(%3068, %3069) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3071 = "mix.prim.slice"(%3064) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3072 = "mix.prim.slice"(%3070) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3073 = "mix.prim.mul"(%3041, %3071) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3074 = "mix.prim.slice"(%3041) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3075 = "mix.prim.slice"(%3041) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3076 = "mix.prim.neg"(%3075) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3077 = "mix.prim.concat"(%3076, %3074) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3078 = "mix.prim.mul"(%3077, %3072) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3079 = "mix.prim.add"(%3073, %3078) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3080 = "mix.prim.mul"(%3042, %3071) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3081 = "mix.prim.slice"(%3042) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3082 = "mix.prim.slice"(%3042) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3083 = "mix.prim.neg"(%3082) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3084 = "mix.prim.concat"(%3083, %3081) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3085 = "mix.prim.mul"(%3084, %3072) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3086 = "mix.prim.add"(%3080, %3085) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3087 = "mix.prim.reshape"(%3079) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3088 = "mix.prim.reshape"(%3086) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3089 = "mix.prim.reshape"(%3087) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3090 = "mix.prim.reshape"(%3088) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3091 = "mix.prim.transpose"(%3089) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3092 = "mix.prim.transpose"(%3090) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3093 = "mix.prim.transpose"(%3092) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3094 = "mix.prim.unsqueeze"(%3091) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3095 = "mix.prim.permute"(%3094) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3096 = "mix.prim.unsqueeze"(%3093) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3097 = "mix.prim.permute"(%3096) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3098 = "mix.prim.permute"(%3095) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3099 = "mix.prim.reshape"(%3098) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3100 = "mix.prim.permute"(%3097) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3101 = "mix.prim.reshape"(%3100) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3102 = "mix.prim.batch_matmul"(%3099, %3101) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3103 = "mix.prim.reshape"(%3102) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3104 = "mix.prim.permute"(%3103) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3105 = "mix.prim.reshape"(%3104) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3106 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3107 = "mix.prim.mul"(%3105, %3106) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3108 = "mix.prim.reshape"(%3107) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3109 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3110 = "mix.comp.masked_fill"(%3108, %14, %3109) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3111 = "mix.comp.softmax"(%3110) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3112 = "mix.prim.reshape"(%3111) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3113 = "mix.prim.reshape"(%3040) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3114 = "mix.prim.transpose"(%3113) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3115 = "mix.prim.batch_matmul"(%3112, %3114) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3116 = "mix.prim.reshape"(%3115) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3117 = "mix.prim.permute"(%3116) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3118 = "mix.prim.reshape"(%3117) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3119 = "mix.prim.reshape"(%3118) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3120 = "mix.prim.transpose"(%3026) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3121 = "mix.prim.matmul"(%3119, %3120) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3122 = "mix.prim.add"(%3121, %3027) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3123 = "mix.prim.reshape"(%3122) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3124 = "mix.prim.mul"(%3014, %3123) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3125 = "mix.comp.weight"() <{param_loc = "transformer.h.24.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3126 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3127 = "mix.prim.pow"(%3124, %3126) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3128 = "mix.comp.mean"(%3127) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3129 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3130 = "mix.prim.add"(%3128, %3129) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3131 = "mix.prim.rsqrt"(%3130) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3132 = "mix.prim.mul"(%3124, %3131) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3133 = "mix.prim.mul"(%3125, %3132) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3134 = "mix.module.linear"(%3133) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.24.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3135 = "mix.comp.silu"(%3134) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3136 = "mix.module.linear"(%3133) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.24.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3137 = "mix.prim.mul"(%3135, %3136) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3138 = "mix.module.linear"(%3137) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.24.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3139 = "mix.prim.add"(%3138, %3124) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3140 = "mix.comp.weight"() <{param_loc = "transformer.h.25.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3141 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3142 = "mix.prim.pow"(%3139, %3141) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3143 = "mix.comp.mean"(%3142) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3144 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3145 = "mix.prim.add"(%3143, %3144) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3146 = "mix.prim.rsqrt"(%3145) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3147 = "mix.prim.mul"(%3139, %3146) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3148 = "mix.prim.mul"(%3140, %3147) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3149 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3150 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3151 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3152 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3153 = "mix.prim.transpose"(%3148) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3154 = "mix.prim.transpose"(%3149) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3155 = "mix.prim.reshape"(%3153) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3156 = "mix.prim.matmul"(%3155, %3154) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3157 = "mix.prim.reshape"(%3156) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3158 = "mix.prim.reshape"(%3157) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3159 = "mix.prim.transpose"(%3150) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3160 = "mix.prim.reshape"(%3153) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3161 = "mix.prim.matmul"(%3160, %3159) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3162 = "mix.prim.reshape"(%3161) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3163 = "mix.prim.reshape"(%3162) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3164 = "mix.prim.slice"(%3163) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3165 = "mix.prim.slice"(%3163) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3166 = "mix.prim.reshape"(%3158) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3167 = "mix.prim.reshape"(%3164) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3168 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3169 = "mix.prim.convert"(%3168) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3170 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %3171 = "mix.prim.div"(%3169, %3170) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %3172 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3173 = "mix.prim.pow"(%3172, %3171) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3174 = "mix.prim.reciprocal"(%3173) : (tensor<80xf16>) -> tensor<80xf16>
    %3175 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3176 = "mix.prim.mul"(%3175, %3174) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3177 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3178 = "mix.prim.unsqueeze"(%3177) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3179 = "mix.prim.permute"(%3178) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3180 = "mix.prim.unsqueeze"(%3176) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3181 = "mix.prim.permute"(%3180) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3182 = "mix.prim.mul"(%3179, %3181) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3183 = "mix.prim.concat"(%3182, %3182) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3184 = "mix.prim.cos"(%3183) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3185 = "mix.prim.slice"(%3184) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3186 = "mix.prim.unsqueeze"(%3185) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3187 = "mix.prim.slice"(%3186) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3188 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3189 = "mix.prim.mul"(%3187, %3188) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3190 = "mix.prim.sin"(%3183) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3191 = "mix.prim.slice"(%3190) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3192 = "mix.prim.unsqueeze"(%3191) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3193 = "mix.prim.slice"(%3192) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3194 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3195 = "mix.prim.mul"(%3193, %3194) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3196 = "mix.prim.slice"(%3189) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3197 = "mix.prim.slice"(%3195) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3198 = "mix.prim.mul"(%3166, %3196) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3199 = "mix.prim.slice"(%3166) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3200 = "mix.prim.slice"(%3166) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3201 = "mix.prim.neg"(%3200) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3202 = "mix.prim.concat"(%3201, %3199) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3203 = "mix.prim.mul"(%3202, %3197) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3204 = "mix.prim.add"(%3198, %3203) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3205 = "mix.prim.mul"(%3167, %3196) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3206 = "mix.prim.slice"(%3167) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3207 = "mix.prim.slice"(%3167) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3208 = "mix.prim.neg"(%3207) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3209 = "mix.prim.concat"(%3208, %3206) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3210 = "mix.prim.mul"(%3209, %3197) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3211 = "mix.prim.add"(%3205, %3210) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3212 = "mix.prim.reshape"(%3204) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3213 = "mix.prim.reshape"(%3211) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3214 = "mix.prim.reshape"(%3212) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3215 = "mix.prim.reshape"(%3213) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3216 = "mix.prim.transpose"(%3214) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3217 = "mix.prim.transpose"(%3215) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3218 = "mix.prim.transpose"(%3217) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3219 = "mix.prim.unsqueeze"(%3216) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3220 = "mix.prim.permute"(%3219) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3221 = "mix.prim.unsqueeze"(%3218) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3222 = "mix.prim.permute"(%3221) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3223 = "mix.prim.permute"(%3220) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3224 = "mix.prim.reshape"(%3223) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3225 = "mix.prim.permute"(%3222) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3226 = "mix.prim.reshape"(%3225) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3227 = "mix.prim.batch_matmul"(%3224, %3226) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3228 = "mix.prim.reshape"(%3227) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3229 = "mix.prim.permute"(%3228) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3230 = "mix.prim.reshape"(%3229) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3231 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3232 = "mix.prim.mul"(%3230, %3231) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3233 = "mix.prim.reshape"(%3232) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3234 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3235 = "mix.comp.masked_fill"(%3233, %14, %3234) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3236 = "mix.comp.softmax"(%3235) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3237 = "mix.prim.reshape"(%3236) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3238 = "mix.prim.reshape"(%3165) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3239 = "mix.prim.transpose"(%3238) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3240 = "mix.prim.batch_matmul"(%3237, %3239) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3241 = "mix.prim.reshape"(%3240) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3242 = "mix.prim.permute"(%3241) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3243 = "mix.prim.reshape"(%3242) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3244 = "mix.prim.reshape"(%3243) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3245 = "mix.prim.transpose"(%3151) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3246 = "mix.prim.matmul"(%3244, %3245) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3247 = "mix.prim.add"(%3246, %3152) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3248 = "mix.prim.reshape"(%3247) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3249 = "mix.prim.mul"(%3139, %3248) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3250 = "mix.comp.weight"() <{param_loc = "transformer.h.25.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3251 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3252 = "mix.prim.pow"(%3249, %3251) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3253 = "mix.comp.mean"(%3252) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3254 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3255 = "mix.prim.add"(%3253, %3254) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3256 = "mix.prim.rsqrt"(%3255) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3257 = "mix.prim.mul"(%3249, %3256) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3258 = "mix.prim.mul"(%3250, %3257) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3259 = "mix.module.linear"(%3258) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.25.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3260 = "mix.comp.silu"(%3259) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3261 = "mix.module.linear"(%3258) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.25.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3262 = "mix.prim.mul"(%3260, %3261) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3263 = "mix.module.linear"(%3262) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.25.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3264 = "mix.prim.add"(%3263, %3249) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3265 = "mix.comp.weight"() <{param_loc = "transformer.h.26.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3266 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3267 = "mix.prim.pow"(%3264, %3266) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3268 = "mix.comp.mean"(%3267) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3269 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3270 = "mix.prim.add"(%3268, %3269) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3271 = "mix.prim.rsqrt"(%3270) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3272 = "mix.prim.mul"(%3264, %3271) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3273 = "mix.prim.mul"(%3265, %3272) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3274 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3275 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3276 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3277 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3278 = "mix.prim.transpose"(%3273) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3279 = "mix.prim.transpose"(%3274) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3280 = "mix.prim.reshape"(%3278) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3281 = "mix.prim.matmul"(%3280, %3279) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3282 = "mix.prim.reshape"(%3281) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3283 = "mix.prim.reshape"(%3282) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3284 = "mix.prim.transpose"(%3275) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3285 = "mix.prim.reshape"(%3278) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3286 = "mix.prim.matmul"(%3285, %3284) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3287 = "mix.prim.reshape"(%3286) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3288 = "mix.prim.reshape"(%3287) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3289 = "mix.prim.slice"(%3288) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3290 = "mix.prim.slice"(%3288) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3291 = "mix.prim.reshape"(%3283) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3292 = "mix.prim.reshape"(%3289) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3293 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3294 = "mix.prim.convert"(%3293) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3295 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %3296 = "mix.prim.div"(%3294, %3295) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %3297 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3298 = "mix.prim.pow"(%3297, %3296) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3299 = "mix.prim.reciprocal"(%3298) : (tensor<80xf16>) -> tensor<80xf16>
    %3300 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3301 = "mix.prim.mul"(%3300, %3299) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3302 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3303 = "mix.prim.unsqueeze"(%3302) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3304 = "mix.prim.permute"(%3303) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3305 = "mix.prim.unsqueeze"(%3301) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3306 = "mix.prim.permute"(%3305) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3307 = "mix.prim.mul"(%3304, %3306) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3308 = "mix.prim.concat"(%3307, %3307) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3309 = "mix.prim.cos"(%3308) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3310 = "mix.prim.slice"(%3309) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3311 = "mix.prim.unsqueeze"(%3310) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3312 = "mix.prim.slice"(%3311) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3313 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3314 = "mix.prim.mul"(%3312, %3313) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3315 = "mix.prim.sin"(%3308) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3316 = "mix.prim.slice"(%3315) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3317 = "mix.prim.unsqueeze"(%3316) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3318 = "mix.prim.slice"(%3317) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3319 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3320 = "mix.prim.mul"(%3318, %3319) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3321 = "mix.prim.slice"(%3314) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3322 = "mix.prim.slice"(%3320) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3323 = "mix.prim.mul"(%3291, %3321) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3324 = "mix.prim.slice"(%3291) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3325 = "mix.prim.slice"(%3291) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3326 = "mix.prim.neg"(%3325) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3327 = "mix.prim.concat"(%3326, %3324) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3328 = "mix.prim.mul"(%3327, %3322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3329 = "mix.prim.add"(%3323, %3328) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3330 = "mix.prim.mul"(%3292, %3321) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3331 = "mix.prim.slice"(%3292) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3332 = "mix.prim.slice"(%3292) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3333 = "mix.prim.neg"(%3332) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3334 = "mix.prim.concat"(%3333, %3331) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3335 = "mix.prim.mul"(%3334, %3322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3336 = "mix.prim.add"(%3330, %3335) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3337 = "mix.prim.reshape"(%3329) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3338 = "mix.prim.reshape"(%3336) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3339 = "mix.prim.reshape"(%3337) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3340 = "mix.prim.reshape"(%3338) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3341 = "mix.prim.transpose"(%3339) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3342 = "mix.prim.transpose"(%3340) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3343 = "mix.prim.transpose"(%3342) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3344 = "mix.prim.unsqueeze"(%3341) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3345 = "mix.prim.permute"(%3344) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3346 = "mix.prim.unsqueeze"(%3343) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3347 = "mix.prim.permute"(%3346) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3348 = "mix.prim.permute"(%3345) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3349 = "mix.prim.reshape"(%3348) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3350 = "mix.prim.permute"(%3347) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3351 = "mix.prim.reshape"(%3350) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3352 = "mix.prim.batch_matmul"(%3349, %3351) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3353 = "mix.prim.reshape"(%3352) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3354 = "mix.prim.permute"(%3353) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3355 = "mix.prim.reshape"(%3354) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3356 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3357 = "mix.prim.mul"(%3355, %3356) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3358 = "mix.prim.reshape"(%3357) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3359 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3360 = "mix.comp.masked_fill"(%3358, %14, %3359) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3361 = "mix.comp.softmax"(%3360) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3362 = "mix.prim.reshape"(%3361) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3363 = "mix.prim.reshape"(%3290) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3364 = "mix.prim.transpose"(%3363) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3365 = "mix.prim.batch_matmul"(%3362, %3364) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3366 = "mix.prim.reshape"(%3365) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3367 = "mix.prim.permute"(%3366) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3368 = "mix.prim.reshape"(%3367) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3369 = "mix.prim.reshape"(%3368) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3370 = "mix.prim.transpose"(%3276) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3371 = "mix.prim.matmul"(%3369, %3370) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3372 = "mix.prim.add"(%3371, %3277) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3373 = "mix.prim.reshape"(%3372) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3374 = "mix.prim.mul"(%3264, %3373) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3375 = "mix.comp.weight"() <{param_loc = "transformer.h.26.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3376 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3377 = "mix.prim.pow"(%3374, %3376) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3378 = "mix.comp.mean"(%3377) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3379 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3380 = "mix.prim.add"(%3378, %3379) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3381 = "mix.prim.rsqrt"(%3380) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3382 = "mix.prim.mul"(%3374, %3381) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3383 = "mix.prim.mul"(%3375, %3382) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3384 = "mix.module.linear"(%3383) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.26.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3385 = "mix.comp.silu"(%3384) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3386 = "mix.module.linear"(%3383) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.26.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3387 = "mix.prim.mul"(%3385, %3386) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3388 = "mix.module.linear"(%3387) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.26.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3389 = "mix.prim.add"(%3388, %3374) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3390 = "mix.comp.weight"() <{param_loc = "transformer.h.27.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3391 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3392 = "mix.prim.pow"(%3389, %3391) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3393 = "mix.comp.mean"(%3392) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3394 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3395 = "mix.prim.add"(%3393, %3394) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3396 = "mix.prim.rsqrt"(%3395) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3397 = "mix.prim.mul"(%3389, %3396) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3398 = "mix.prim.mul"(%3390, %3397) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3399 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3400 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3401 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3402 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3403 = "mix.prim.transpose"(%3398) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3404 = "mix.prim.transpose"(%3399) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3405 = "mix.prim.reshape"(%3403) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3406 = "mix.prim.matmul"(%3405, %3404) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3407 = "mix.prim.reshape"(%3406) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3408 = "mix.prim.reshape"(%3407) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3409 = "mix.prim.transpose"(%3400) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3410 = "mix.prim.reshape"(%3403) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3411 = "mix.prim.matmul"(%3410, %3409) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3412 = "mix.prim.reshape"(%3411) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3413 = "mix.prim.reshape"(%3412) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3414 = "mix.prim.slice"(%3413) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3415 = "mix.prim.slice"(%3413) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3416 = "mix.prim.reshape"(%3408) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3417 = "mix.prim.reshape"(%3414) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3418 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3419 = "mix.prim.convert"(%3418) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3420 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %3421 = "mix.prim.div"(%3419, %3420) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %3422 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3423 = "mix.prim.pow"(%3422, %3421) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3424 = "mix.prim.reciprocal"(%3423) : (tensor<80xf16>) -> tensor<80xf16>
    %3425 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3426 = "mix.prim.mul"(%3425, %3424) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3427 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3428 = "mix.prim.unsqueeze"(%3427) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3429 = "mix.prim.permute"(%3428) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3430 = "mix.prim.unsqueeze"(%3426) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3431 = "mix.prim.permute"(%3430) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3432 = "mix.prim.mul"(%3429, %3431) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3433 = "mix.prim.concat"(%3432, %3432) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3434 = "mix.prim.cos"(%3433) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3435 = "mix.prim.slice"(%3434) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3436 = "mix.prim.unsqueeze"(%3435) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3437 = "mix.prim.slice"(%3436) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3438 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3439 = "mix.prim.mul"(%3437, %3438) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3440 = "mix.prim.sin"(%3433) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3441 = "mix.prim.slice"(%3440) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3442 = "mix.prim.unsqueeze"(%3441) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3443 = "mix.prim.slice"(%3442) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3444 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3445 = "mix.prim.mul"(%3443, %3444) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3446 = "mix.prim.slice"(%3439) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3447 = "mix.prim.slice"(%3445) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3448 = "mix.prim.mul"(%3416, %3446) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3449 = "mix.prim.slice"(%3416) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3450 = "mix.prim.slice"(%3416) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3451 = "mix.prim.neg"(%3450) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3452 = "mix.prim.concat"(%3451, %3449) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3453 = "mix.prim.mul"(%3452, %3447) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3454 = "mix.prim.add"(%3448, %3453) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3455 = "mix.prim.mul"(%3417, %3446) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3456 = "mix.prim.slice"(%3417) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3457 = "mix.prim.slice"(%3417) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3458 = "mix.prim.neg"(%3457) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3459 = "mix.prim.concat"(%3458, %3456) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3460 = "mix.prim.mul"(%3459, %3447) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3461 = "mix.prim.add"(%3455, %3460) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3462 = "mix.prim.reshape"(%3454) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3463 = "mix.prim.reshape"(%3461) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3464 = "mix.prim.reshape"(%3462) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3465 = "mix.prim.reshape"(%3463) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3466 = "mix.prim.transpose"(%3464) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3467 = "mix.prim.transpose"(%3465) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3468 = "mix.prim.transpose"(%3467) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3469 = "mix.prim.unsqueeze"(%3466) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3470 = "mix.prim.permute"(%3469) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3471 = "mix.prim.unsqueeze"(%3468) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3472 = "mix.prim.permute"(%3471) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3473 = "mix.prim.permute"(%3470) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3474 = "mix.prim.reshape"(%3473) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3475 = "mix.prim.permute"(%3472) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3476 = "mix.prim.reshape"(%3475) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3477 = "mix.prim.batch_matmul"(%3474, %3476) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3478 = "mix.prim.reshape"(%3477) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3479 = "mix.prim.permute"(%3478) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3480 = "mix.prim.reshape"(%3479) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3481 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3482 = "mix.prim.mul"(%3480, %3481) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3483 = "mix.prim.reshape"(%3482) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3484 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3485 = "mix.comp.masked_fill"(%3483, %14, %3484) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3486 = "mix.comp.softmax"(%3485) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3487 = "mix.prim.reshape"(%3486) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3488 = "mix.prim.reshape"(%3415) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3489 = "mix.prim.transpose"(%3488) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3490 = "mix.prim.batch_matmul"(%3487, %3489) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3491 = "mix.prim.reshape"(%3490) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3492 = "mix.prim.permute"(%3491) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3493 = "mix.prim.reshape"(%3492) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3494 = "mix.prim.reshape"(%3493) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3495 = "mix.prim.transpose"(%3401) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3496 = "mix.prim.matmul"(%3494, %3495) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3497 = "mix.prim.add"(%3496, %3402) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3498 = "mix.prim.reshape"(%3497) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3499 = "mix.prim.mul"(%3389, %3498) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3500 = "mix.comp.weight"() <{param_loc = "transformer.h.27.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3501 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3502 = "mix.prim.pow"(%3499, %3501) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3503 = "mix.comp.mean"(%3502) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3504 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3505 = "mix.prim.add"(%3503, %3504) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3506 = "mix.prim.rsqrt"(%3505) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3507 = "mix.prim.mul"(%3499, %3506) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3508 = "mix.prim.mul"(%3500, %3507) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3509 = "mix.module.linear"(%3508) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.27.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3510 = "mix.comp.silu"(%3509) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3511 = "mix.module.linear"(%3508) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.27.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3512 = "mix.prim.mul"(%3510, %3511) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3513 = "mix.module.linear"(%3512) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.27.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3514 = "mix.prim.add"(%3513, %3499) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3515 = "mix.comp.weight"() <{param_loc = "transformer.h.28.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3516 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3517 = "mix.prim.pow"(%3514, %3516) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3518 = "mix.comp.mean"(%3517) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3519 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3520 = "mix.prim.add"(%3518, %3519) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3521 = "mix.prim.rsqrt"(%3520) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3522 = "mix.prim.mul"(%3514, %3521) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3523 = "mix.prim.mul"(%3515, %3522) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3524 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3525 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3526 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3527 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3528 = "mix.prim.transpose"(%3523) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3529 = "mix.prim.transpose"(%3524) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3530 = "mix.prim.reshape"(%3528) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3531 = "mix.prim.matmul"(%3530, %3529) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3532 = "mix.prim.reshape"(%3531) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3533 = "mix.prim.reshape"(%3532) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3534 = "mix.prim.transpose"(%3525) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3535 = "mix.prim.reshape"(%3528) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3536 = "mix.prim.matmul"(%3535, %3534) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3537 = "mix.prim.reshape"(%3536) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3538 = "mix.prim.reshape"(%3537) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3539 = "mix.prim.slice"(%3538) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3540 = "mix.prim.slice"(%3538) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3541 = "mix.prim.reshape"(%3533) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3542 = "mix.prim.reshape"(%3539) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3543 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3544 = "mix.prim.convert"(%3543) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3545 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %3546 = "mix.prim.div"(%3544, %3545) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %3547 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3548 = "mix.prim.pow"(%3547, %3546) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3549 = "mix.prim.reciprocal"(%3548) : (tensor<80xf16>) -> tensor<80xf16>
    %3550 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3551 = "mix.prim.mul"(%3550, %3549) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3552 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3553 = "mix.prim.unsqueeze"(%3552) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3554 = "mix.prim.permute"(%3553) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3555 = "mix.prim.unsqueeze"(%3551) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3556 = "mix.prim.permute"(%3555) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3557 = "mix.prim.mul"(%3554, %3556) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3558 = "mix.prim.concat"(%3557, %3557) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3559 = "mix.prim.cos"(%3558) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3560 = "mix.prim.slice"(%3559) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3561 = "mix.prim.unsqueeze"(%3560) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3562 = "mix.prim.slice"(%3561) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3563 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3564 = "mix.prim.mul"(%3562, %3563) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3565 = "mix.prim.sin"(%3558) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3566 = "mix.prim.slice"(%3565) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3567 = "mix.prim.unsqueeze"(%3566) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3568 = "mix.prim.slice"(%3567) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3569 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3570 = "mix.prim.mul"(%3568, %3569) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3571 = "mix.prim.slice"(%3564) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3572 = "mix.prim.slice"(%3570) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3573 = "mix.prim.mul"(%3541, %3571) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3574 = "mix.prim.slice"(%3541) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3575 = "mix.prim.slice"(%3541) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3576 = "mix.prim.neg"(%3575) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3577 = "mix.prim.concat"(%3576, %3574) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3578 = "mix.prim.mul"(%3577, %3572) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3579 = "mix.prim.add"(%3573, %3578) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3580 = "mix.prim.mul"(%3542, %3571) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3581 = "mix.prim.slice"(%3542) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3582 = "mix.prim.slice"(%3542) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3583 = "mix.prim.neg"(%3582) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3584 = "mix.prim.concat"(%3583, %3581) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3585 = "mix.prim.mul"(%3584, %3572) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3586 = "mix.prim.add"(%3580, %3585) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3587 = "mix.prim.reshape"(%3579) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3588 = "mix.prim.reshape"(%3586) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3589 = "mix.prim.reshape"(%3587) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3590 = "mix.prim.reshape"(%3588) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3591 = "mix.prim.transpose"(%3589) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3592 = "mix.prim.transpose"(%3590) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3593 = "mix.prim.transpose"(%3592) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3594 = "mix.prim.unsqueeze"(%3591) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3595 = "mix.prim.permute"(%3594) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3596 = "mix.prim.unsqueeze"(%3593) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3597 = "mix.prim.permute"(%3596) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3598 = "mix.prim.permute"(%3595) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3599 = "mix.prim.reshape"(%3598) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3600 = "mix.prim.permute"(%3597) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3601 = "mix.prim.reshape"(%3600) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3602 = "mix.prim.batch_matmul"(%3599, %3601) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3603 = "mix.prim.reshape"(%3602) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3604 = "mix.prim.permute"(%3603) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3605 = "mix.prim.reshape"(%3604) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3606 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3607 = "mix.prim.mul"(%3605, %3606) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3608 = "mix.prim.reshape"(%3607) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3609 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3610 = "mix.comp.masked_fill"(%3608, %14, %3609) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3611 = "mix.comp.softmax"(%3610) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3612 = "mix.prim.reshape"(%3611) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3613 = "mix.prim.reshape"(%3540) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3614 = "mix.prim.transpose"(%3613) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3615 = "mix.prim.batch_matmul"(%3612, %3614) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3616 = "mix.prim.reshape"(%3615) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3617 = "mix.prim.permute"(%3616) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3618 = "mix.prim.reshape"(%3617) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3619 = "mix.prim.reshape"(%3618) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3620 = "mix.prim.transpose"(%3526) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3621 = "mix.prim.matmul"(%3619, %3620) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3622 = "mix.prim.add"(%3621, %3527) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3623 = "mix.prim.reshape"(%3622) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3624 = "mix.prim.mul"(%3514, %3623) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3625 = "mix.comp.weight"() <{param_loc = "transformer.h.28.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3626 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3627 = "mix.prim.pow"(%3624, %3626) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3628 = "mix.comp.mean"(%3627) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3629 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3630 = "mix.prim.add"(%3628, %3629) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3631 = "mix.prim.rsqrt"(%3630) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3632 = "mix.prim.mul"(%3624, %3631) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3633 = "mix.prim.mul"(%3625, %3632) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3634 = "mix.module.linear"(%3633) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.28.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3635 = "mix.comp.silu"(%3634) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3636 = "mix.module.linear"(%3633) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.28.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3637 = "mix.prim.mul"(%3635, %3636) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3638 = "mix.module.linear"(%3637) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.28.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3639 = "mix.prim.add"(%3638, %3624) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3640 = "mix.comp.weight"() <{param_loc = "transformer.h.29.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3641 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3642 = "mix.prim.pow"(%3639, %3641) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3643 = "mix.comp.mean"(%3642) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3644 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3645 = "mix.prim.add"(%3643, %3644) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3646 = "mix.prim.rsqrt"(%3645) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3647 = "mix.prim.mul"(%3639, %3646) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3648 = "mix.prim.mul"(%3640, %3647) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3649 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3650 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3651 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3652 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3653 = "mix.prim.transpose"(%3648) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3654 = "mix.prim.transpose"(%3649) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3655 = "mix.prim.reshape"(%3653) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3656 = "mix.prim.matmul"(%3655, %3654) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3657 = "mix.prim.reshape"(%3656) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3658 = "mix.prim.reshape"(%3657) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3659 = "mix.prim.transpose"(%3650) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3660 = "mix.prim.reshape"(%3653) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3661 = "mix.prim.matmul"(%3660, %3659) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3662 = "mix.prim.reshape"(%3661) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3663 = "mix.prim.reshape"(%3662) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3664 = "mix.prim.slice"(%3663) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3665 = "mix.prim.slice"(%3663) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3666 = "mix.prim.reshape"(%3658) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3667 = "mix.prim.reshape"(%3664) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3668 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3669 = "mix.prim.convert"(%3668) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3670 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %3671 = "mix.prim.div"(%3669, %3670) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %3672 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3673 = "mix.prim.pow"(%3672, %3671) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3674 = "mix.prim.reciprocal"(%3673) : (tensor<80xf16>) -> tensor<80xf16>
    %3675 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3676 = "mix.prim.mul"(%3675, %3674) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3677 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3678 = "mix.prim.unsqueeze"(%3677) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3679 = "mix.prim.permute"(%3678) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3680 = "mix.prim.unsqueeze"(%3676) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3681 = "mix.prim.permute"(%3680) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3682 = "mix.prim.mul"(%3679, %3681) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3683 = "mix.prim.concat"(%3682, %3682) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3684 = "mix.prim.cos"(%3683) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3685 = "mix.prim.slice"(%3684) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3686 = "mix.prim.unsqueeze"(%3685) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3687 = "mix.prim.slice"(%3686) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3688 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3689 = "mix.prim.mul"(%3687, %3688) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3690 = "mix.prim.sin"(%3683) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3691 = "mix.prim.slice"(%3690) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3692 = "mix.prim.unsqueeze"(%3691) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3693 = "mix.prim.slice"(%3692) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3694 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3695 = "mix.prim.mul"(%3693, %3694) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3696 = "mix.prim.slice"(%3689) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3697 = "mix.prim.slice"(%3695) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3698 = "mix.prim.mul"(%3666, %3696) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3699 = "mix.prim.slice"(%3666) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3700 = "mix.prim.slice"(%3666) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3701 = "mix.prim.neg"(%3700) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3702 = "mix.prim.concat"(%3701, %3699) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3703 = "mix.prim.mul"(%3702, %3697) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3704 = "mix.prim.add"(%3698, %3703) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3705 = "mix.prim.mul"(%3667, %3696) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3706 = "mix.prim.slice"(%3667) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3707 = "mix.prim.slice"(%3667) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3708 = "mix.prim.neg"(%3707) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3709 = "mix.prim.concat"(%3708, %3706) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3710 = "mix.prim.mul"(%3709, %3697) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3711 = "mix.prim.add"(%3705, %3710) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3712 = "mix.prim.reshape"(%3704) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3713 = "mix.prim.reshape"(%3711) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3714 = "mix.prim.reshape"(%3712) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3715 = "mix.prim.reshape"(%3713) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3716 = "mix.prim.transpose"(%3714) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3717 = "mix.prim.transpose"(%3715) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3718 = "mix.prim.transpose"(%3717) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3719 = "mix.prim.unsqueeze"(%3716) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3720 = "mix.prim.permute"(%3719) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3721 = "mix.prim.unsqueeze"(%3718) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3722 = "mix.prim.permute"(%3721) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3723 = "mix.prim.permute"(%3720) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3724 = "mix.prim.reshape"(%3723) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3725 = "mix.prim.permute"(%3722) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3726 = "mix.prim.reshape"(%3725) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3727 = "mix.prim.batch_matmul"(%3724, %3726) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3728 = "mix.prim.reshape"(%3727) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3729 = "mix.prim.permute"(%3728) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3730 = "mix.prim.reshape"(%3729) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3731 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3732 = "mix.prim.mul"(%3730, %3731) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3733 = "mix.prim.reshape"(%3732) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3734 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3735 = "mix.comp.masked_fill"(%3733, %14, %3734) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3736 = "mix.comp.softmax"(%3735) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3737 = "mix.prim.reshape"(%3736) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3738 = "mix.prim.reshape"(%3665) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3739 = "mix.prim.transpose"(%3738) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3740 = "mix.prim.batch_matmul"(%3737, %3739) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3741 = "mix.prim.reshape"(%3740) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3742 = "mix.prim.permute"(%3741) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3743 = "mix.prim.reshape"(%3742) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3744 = "mix.prim.reshape"(%3743) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3745 = "mix.prim.transpose"(%3651) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3746 = "mix.prim.matmul"(%3744, %3745) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3747 = "mix.prim.add"(%3746, %3652) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3748 = "mix.prim.reshape"(%3747) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3749 = "mix.prim.mul"(%3639, %3748) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3750 = "mix.comp.weight"() <{param_loc = "transformer.h.29.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3751 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3752 = "mix.prim.pow"(%3749, %3751) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3753 = "mix.comp.mean"(%3752) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3754 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3755 = "mix.prim.add"(%3753, %3754) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3756 = "mix.prim.rsqrt"(%3755) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3757 = "mix.prim.mul"(%3749, %3756) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3758 = "mix.prim.mul"(%3750, %3757) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3759 = "mix.module.linear"(%3758) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.29.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3760 = "mix.comp.silu"(%3759) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3761 = "mix.module.linear"(%3758) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.29.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3762 = "mix.prim.mul"(%3760, %3761) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3763 = "mix.module.linear"(%3762) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.29.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3764 = "mix.prim.add"(%3763, %3749) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3765 = "mix.comp.weight"() <{param_loc = "transformer.h.30.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3766 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3767 = "mix.prim.pow"(%3764, %3766) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3768 = "mix.comp.mean"(%3767) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3769 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3770 = "mix.prim.add"(%3768, %3769) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3771 = "mix.prim.rsqrt"(%3770) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3772 = "mix.prim.mul"(%3764, %3771) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3773 = "mix.prim.mul"(%3765, %3772) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3774 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3775 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3776 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3777 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3778 = "mix.prim.transpose"(%3773) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3779 = "mix.prim.transpose"(%3774) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3780 = "mix.prim.reshape"(%3778) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3781 = "mix.prim.matmul"(%3780, %3779) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3782 = "mix.prim.reshape"(%3781) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3783 = "mix.prim.reshape"(%3782) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3784 = "mix.prim.transpose"(%3775) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3785 = "mix.prim.reshape"(%3778) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3786 = "mix.prim.matmul"(%3785, %3784) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3787 = "mix.prim.reshape"(%3786) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3788 = "mix.prim.reshape"(%3787) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3789 = "mix.prim.slice"(%3788) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3790 = "mix.prim.slice"(%3788) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3791 = "mix.prim.reshape"(%3783) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3792 = "mix.prim.reshape"(%3789) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3793 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3794 = "mix.prim.convert"(%3793) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3795 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %3796 = "mix.prim.div"(%3794, %3795) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %3797 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3798 = "mix.prim.pow"(%3797, %3796) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3799 = "mix.prim.reciprocal"(%3798) : (tensor<80xf16>) -> tensor<80xf16>
    %3800 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3801 = "mix.prim.mul"(%3800, %3799) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3802 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3803 = "mix.prim.unsqueeze"(%3802) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3804 = "mix.prim.permute"(%3803) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3805 = "mix.prim.unsqueeze"(%3801) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3806 = "mix.prim.permute"(%3805) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3807 = "mix.prim.mul"(%3804, %3806) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3808 = "mix.prim.concat"(%3807, %3807) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3809 = "mix.prim.cos"(%3808) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3810 = "mix.prim.slice"(%3809) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3811 = "mix.prim.unsqueeze"(%3810) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3812 = "mix.prim.slice"(%3811) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3813 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3814 = "mix.prim.mul"(%3812, %3813) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3815 = "mix.prim.sin"(%3808) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3816 = "mix.prim.slice"(%3815) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3817 = "mix.prim.unsqueeze"(%3816) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3818 = "mix.prim.slice"(%3817) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3819 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3820 = "mix.prim.mul"(%3818, %3819) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3821 = "mix.prim.slice"(%3814) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3822 = "mix.prim.slice"(%3820) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3823 = "mix.prim.mul"(%3791, %3821) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3824 = "mix.prim.slice"(%3791) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3825 = "mix.prim.slice"(%3791) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3826 = "mix.prim.neg"(%3825) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3827 = "mix.prim.concat"(%3826, %3824) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3828 = "mix.prim.mul"(%3827, %3822) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3829 = "mix.prim.add"(%3823, %3828) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3830 = "mix.prim.mul"(%3792, %3821) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3831 = "mix.prim.slice"(%3792) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3832 = "mix.prim.slice"(%3792) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3833 = "mix.prim.neg"(%3832) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3834 = "mix.prim.concat"(%3833, %3831) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3835 = "mix.prim.mul"(%3834, %3822) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3836 = "mix.prim.add"(%3830, %3835) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3837 = "mix.prim.reshape"(%3829) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3838 = "mix.prim.reshape"(%3836) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3839 = "mix.prim.reshape"(%3837) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3840 = "mix.prim.reshape"(%3838) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3841 = "mix.prim.transpose"(%3839) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3842 = "mix.prim.transpose"(%3840) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3843 = "mix.prim.transpose"(%3842) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3844 = "mix.prim.unsqueeze"(%3841) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3845 = "mix.prim.permute"(%3844) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3846 = "mix.prim.unsqueeze"(%3843) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3847 = "mix.prim.permute"(%3846) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3848 = "mix.prim.permute"(%3845) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3849 = "mix.prim.reshape"(%3848) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3850 = "mix.prim.permute"(%3847) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3851 = "mix.prim.reshape"(%3850) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3852 = "mix.prim.batch_matmul"(%3849, %3851) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3853 = "mix.prim.reshape"(%3852) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3854 = "mix.prim.permute"(%3853) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3855 = "mix.prim.reshape"(%3854) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3856 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3857 = "mix.prim.mul"(%3855, %3856) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3858 = "mix.prim.reshape"(%3857) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3859 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3860 = "mix.comp.masked_fill"(%3858, %14, %3859) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3861 = "mix.comp.softmax"(%3860) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3862 = "mix.prim.reshape"(%3861) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3863 = "mix.prim.reshape"(%3790) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3864 = "mix.prim.transpose"(%3863) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3865 = "mix.prim.batch_matmul"(%3862, %3864) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3866 = "mix.prim.reshape"(%3865) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3867 = "mix.prim.permute"(%3866) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3868 = "mix.prim.reshape"(%3867) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3869 = "mix.prim.reshape"(%3868) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3870 = "mix.prim.transpose"(%3776) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3871 = "mix.prim.matmul"(%3869, %3870) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3872 = "mix.prim.add"(%3871, %3777) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3873 = "mix.prim.reshape"(%3872) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3874 = "mix.prim.mul"(%3764, %3873) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3875 = "mix.comp.weight"() <{param_loc = "transformer.h.30.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3876 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3877 = "mix.prim.pow"(%3874, %3876) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3878 = "mix.comp.mean"(%3877) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3879 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3880 = "mix.prim.add"(%3878, %3879) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3881 = "mix.prim.rsqrt"(%3880) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3882 = "mix.prim.mul"(%3874, %3881) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3883 = "mix.prim.mul"(%3875, %3882) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3884 = "mix.module.linear"(%3883) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.30.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3885 = "mix.comp.silu"(%3884) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3886 = "mix.module.linear"(%3883) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.30.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3887 = "mix.prim.mul"(%3885, %3886) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3888 = "mix.module.linear"(%3887) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.30.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3889 = "mix.prim.add"(%3888, %3874) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3890 = "mix.comp.weight"() <{param_loc = "transformer.h.31.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3891 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3892 = "mix.prim.pow"(%3889, %3891) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3893 = "mix.comp.mean"(%3892) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3894 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3895 = "mix.prim.add"(%3893, %3894) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3896 = "mix.prim.rsqrt"(%3895) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3897 = "mix.prim.mul"(%3889, %3896) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3898 = "mix.prim.mul"(%3890, %3897) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3899 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3900 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3901 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3902 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3903 = "mix.prim.transpose"(%3898) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3904 = "mix.prim.transpose"(%3899) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3905 = "mix.prim.reshape"(%3903) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3906 = "mix.prim.matmul"(%3905, %3904) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3907 = "mix.prim.reshape"(%3906) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3908 = "mix.prim.reshape"(%3907) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3909 = "mix.prim.transpose"(%3900) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3910 = "mix.prim.reshape"(%3903) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3911 = "mix.prim.matmul"(%3910, %3909) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3912 = "mix.prim.reshape"(%3911) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3913 = "mix.prim.reshape"(%3912) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3914 = "mix.prim.slice"(%3913) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3915 = "mix.prim.slice"(%3913) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3916 = "mix.prim.reshape"(%3908) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3917 = "mix.prim.reshape"(%3914) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3918 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3919 = "mix.prim.convert"(%3918) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3920 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %3921 = "mix.prim.div"(%3919, %3920) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %3922 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3923 = "mix.prim.pow"(%3922, %3921) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3924 = "mix.prim.reciprocal"(%3923) : (tensor<80xf16>) -> tensor<80xf16>
    %3925 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3926 = "mix.prim.mul"(%3925, %3924) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3927 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3928 = "mix.prim.unsqueeze"(%3927) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3929 = "mix.prim.permute"(%3928) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3930 = "mix.prim.unsqueeze"(%3926) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3931 = "mix.prim.permute"(%3930) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3932 = "mix.prim.mul"(%3929, %3931) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3933 = "mix.prim.concat"(%3932, %3932) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3934 = "mix.prim.cos"(%3933) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3935 = "mix.prim.slice"(%3934) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3936 = "mix.prim.unsqueeze"(%3935) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3937 = "mix.prim.slice"(%3936) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3938 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3939 = "mix.prim.mul"(%3937, %3938) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3940 = "mix.prim.sin"(%3933) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3941 = "mix.prim.slice"(%3940) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3942 = "mix.prim.unsqueeze"(%3941) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3943 = "mix.prim.slice"(%3942) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3944 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3945 = "mix.prim.mul"(%3943, %3944) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3946 = "mix.prim.slice"(%3939) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3947 = "mix.prim.slice"(%3945) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3948 = "mix.prim.mul"(%3916, %3946) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3949 = "mix.prim.slice"(%3916) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3950 = "mix.prim.slice"(%3916) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3951 = "mix.prim.neg"(%3950) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3952 = "mix.prim.concat"(%3951, %3949) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3953 = "mix.prim.mul"(%3952, %3947) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3954 = "mix.prim.add"(%3948, %3953) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3955 = "mix.prim.mul"(%3917, %3946) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3956 = "mix.prim.slice"(%3917) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3957 = "mix.prim.slice"(%3917) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3958 = "mix.prim.neg"(%3957) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3959 = "mix.prim.concat"(%3958, %3956) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3960 = "mix.prim.mul"(%3959, %3947) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3961 = "mix.prim.add"(%3955, %3960) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3962 = "mix.prim.reshape"(%3954) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3963 = "mix.prim.reshape"(%3961) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3964 = "mix.prim.reshape"(%3962) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3965 = "mix.prim.reshape"(%3963) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3966 = "mix.prim.transpose"(%3964) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3967 = "mix.prim.transpose"(%3965) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3968 = "mix.prim.transpose"(%3967) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3969 = "mix.prim.unsqueeze"(%3966) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3970 = "mix.prim.permute"(%3969) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3971 = "mix.prim.unsqueeze"(%3968) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3972 = "mix.prim.permute"(%3971) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3973 = "mix.prim.permute"(%3970) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3974 = "mix.prim.reshape"(%3973) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3975 = "mix.prim.permute"(%3972) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3976 = "mix.prim.reshape"(%3975) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3977 = "mix.prim.batch_matmul"(%3974, %3976) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3978 = "mix.prim.reshape"(%3977) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3979 = "mix.prim.permute"(%3978) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3980 = "mix.prim.reshape"(%3979) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3981 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3982 = "mix.prim.mul"(%3980, %3981) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3983 = "mix.prim.reshape"(%3982) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3984 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %3985 = "mix.comp.masked_fill"(%3983, %14, %3984) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %3986 = "mix.comp.softmax"(%3985) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3987 = "mix.prim.reshape"(%3986) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3988 = "mix.prim.reshape"(%3915) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3989 = "mix.prim.transpose"(%3988) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3990 = "mix.prim.batch_matmul"(%3987, %3989) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3991 = "mix.prim.reshape"(%3990) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3992 = "mix.prim.permute"(%3991) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3993 = "mix.prim.reshape"(%3992) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3994 = "mix.prim.reshape"(%3993) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3995 = "mix.prim.transpose"(%3901) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3996 = "mix.prim.matmul"(%3994, %3995) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3997 = "mix.prim.add"(%3996, %3902) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3998 = "mix.prim.reshape"(%3997) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3999 = "mix.prim.mul"(%3889, %3998) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4000 = "mix.comp.weight"() <{param_loc = "transformer.h.31.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4001 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4002 = "mix.prim.pow"(%3999, %4001) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4003 = "mix.comp.mean"(%4002) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4004 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4005 = "mix.prim.add"(%4003, %4004) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4006 = "mix.prim.rsqrt"(%4005) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4007 = "mix.prim.mul"(%3999, %4006) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4008 = "mix.prim.mul"(%4000, %4007) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4009 = "mix.module.linear"(%4008) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.31.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4010 = "mix.comp.silu"(%4009) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4011 = "mix.module.linear"(%4008) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.31.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4012 = "mix.prim.mul"(%4010, %4011) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4013 = "mix.module.linear"(%4012) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.31.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4014 = "mix.prim.add"(%4013, %3999) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4015 = "mix.comp.weight"() <{param_loc = "transformer.h.32.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4016 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4017 = "mix.prim.pow"(%4014, %4016) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4018 = "mix.comp.mean"(%4017) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4019 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4020 = "mix.prim.add"(%4018, %4019) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4021 = "mix.prim.rsqrt"(%4020) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4022 = "mix.prim.mul"(%4014, %4021) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4023 = "mix.prim.mul"(%4015, %4022) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4024 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4025 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4026 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4027 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4028 = "mix.prim.transpose"(%4023) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4029 = "mix.prim.transpose"(%4024) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4030 = "mix.prim.reshape"(%4028) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4031 = "mix.prim.matmul"(%4030, %4029) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4032 = "mix.prim.reshape"(%4031) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4033 = "mix.prim.reshape"(%4032) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4034 = "mix.prim.transpose"(%4025) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4035 = "mix.prim.reshape"(%4028) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4036 = "mix.prim.matmul"(%4035, %4034) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4037 = "mix.prim.reshape"(%4036) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4038 = "mix.prim.reshape"(%4037) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4039 = "mix.prim.slice"(%4038) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4040 = "mix.prim.slice"(%4038) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4041 = "mix.prim.reshape"(%4033) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4042 = "mix.prim.reshape"(%4039) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4043 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4044 = "mix.prim.convert"(%4043) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4045 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %4046 = "mix.prim.div"(%4044, %4045) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %4047 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4048 = "mix.prim.pow"(%4047, %4046) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4049 = "mix.prim.reciprocal"(%4048) : (tensor<80xf16>) -> tensor<80xf16>
    %4050 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4051 = "mix.prim.mul"(%4050, %4049) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4052 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4053 = "mix.prim.unsqueeze"(%4052) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4054 = "mix.prim.permute"(%4053) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4055 = "mix.prim.unsqueeze"(%4051) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4056 = "mix.prim.permute"(%4055) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4057 = "mix.prim.mul"(%4054, %4056) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4058 = "mix.prim.concat"(%4057, %4057) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4059 = "mix.prim.cos"(%4058) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4060 = "mix.prim.slice"(%4059) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4061 = "mix.prim.unsqueeze"(%4060) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4062 = "mix.prim.slice"(%4061) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4063 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4064 = "mix.prim.mul"(%4062, %4063) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4065 = "mix.prim.sin"(%4058) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4066 = "mix.prim.slice"(%4065) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4067 = "mix.prim.unsqueeze"(%4066) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4068 = "mix.prim.slice"(%4067) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4069 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4070 = "mix.prim.mul"(%4068, %4069) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4071 = "mix.prim.slice"(%4064) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4072 = "mix.prim.slice"(%4070) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4073 = "mix.prim.mul"(%4041, %4071) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4074 = "mix.prim.slice"(%4041) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4075 = "mix.prim.slice"(%4041) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4076 = "mix.prim.neg"(%4075) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4077 = "mix.prim.concat"(%4076, %4074) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4078 = "mix.prim.mul"(%4077, %4072) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4079 = "mix.prim.add"(%4073, %4078) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4080 = "mix.prim.mul"(%4042, %4071) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4081 = "mix.prim.slice"(%4042) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4082 = "mix.prim.slice"(%4042) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4083 = "mix.prim.neg"(%4082) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4084 = "mix.prim.concat"(%4083, %4081) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4085 = "mix.prim.mul"(%4084, %4072) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4086 = "mix.prim.add"(%4080, %4085) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4087 = "mix.prim.reshape"(%4079) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4088 = "mix.prim.reshape"(%4086) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4089 = "mix.prim.reshape"(%4087) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4090 = "mix.prim.reshape"(%4088) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4091 = "mix.prim.transpose"(%4089) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4092 = "mix.prim.transpose"(%4090) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4093 = "mix.prim.transpose"(%4092) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4094 = "mix.prim.unsqueeze"(%4091) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4095 = "mix.prim.permute"(%4094) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4096 = "mix.prim.unsqueeze"(%4093) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4097 = "mix.prim.permute"(%4096) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4098 = "mix.prim.permute"(%4095) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4099 = "mix.prim.reshape"(%4098) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4100 = "mix.prim.permute"(%4097) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4101 = "mix.prim.reshape"(%4100) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4102 = "mix.prim.batch_matmul"(%4099, %4101) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4103 = "mix.prim.reshape"(%4102) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4104 = "mix.prim.permute"(%4103) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4105 = "mix.prim.reshape"(%4104) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4106 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4107 = "mix.prim.mul"(%4105, %4106) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4108 = "mix.prim.reshape"(%4107) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4109 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4110 = "mix.comp.masked_fill"(%4108, %14, %4109) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4111 = "mix.comp.softmax"(%4110) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4112 = "mix.prim.reshape"(%4111) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4113 = "mix.prim.reshape"(%4040) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4114 = "mix.prim.transpose"(%4113) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4115 = "mix.prim.batch_matmul"(%4112, %4114) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4116 = "mix.prim.reshape"(%4115) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4117 = "mix.prim.permute"(%4116) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4118 = "mix.prim.reshape"(%4117) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4119 = "mix.prim.reshape"(%4118) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4120 = "mix.prim.transpose"(%4026) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4121 = "mix.prim.matmul"(%4119, %4120) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4122 = "mix.prim.add"(%4121, %4027) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4123 = "mix.prim.reshape"(%4122) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4124 = "mix.prim.mul"(%4014, %4123) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4125 = "mix.comp.weight"() <{param_loc = "transformer.h.32.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4126 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4127 = "mix.prim.pow"(%4124, %4126) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4128 = "mix.comp.mean"(%4127) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4129 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4130 = "mix.prim.add"(%4128, %4129) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4131 = "mix.prim.rsqrt"(%4130) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4132 = "mix.prim.mul"(%4124, %4131) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4133 = "mix.prim.mul"(%4125, %4132) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4134 = "mix.module.linear"(%4133) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.32.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4135 = "mix.comp.silu"(%4134) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4136 = "mix.module.linear"(%4133) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.32.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4137 = "mix.prim.mul"(%4135, %4136) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4138 = "mix.module.linear"(%4137) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.32.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4139 = "mix.prim.add"(%4138, %4124) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4140 = "mix.comp.weight"() <{param_loc = "transformer.h.33.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4141 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4142 = "mix.prim.pow"(%4139, %4141) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4143 = "mix.comp.mean"(%4142) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4144 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4145 = "mix.prim.add"(%4143, %4144) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4146 = "mix.prim.rsqrt"(%4145) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4147 = "mix.prim.mul"(%4139, %4146) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4148 = "mix.prim.mul"(%4140, %4147) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4149 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4150 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4151 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4152 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4153 = "mix.prim.transpose"(%4148) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4154 = "mix.prim.transpose"(%4149) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4155 = "mix.prim.reshape"(%4153) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4156 = "mix.prim.matmul"(%4155, %4154) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4157 = "mix.prim.reshape"(%4156) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4158 = "mix.prim.reshape"(%4157) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4159 = "mix.prim.transpose"(%4150) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4160 = "mix.prim.reshape"(%4153) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4161 = "mix.prim.matmul"(%4160, %4159) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4162 = "mix.prim.reshape"(%4161) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4163 = "mix.prim.reshape"(%4162) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4164 = "mix.prim.slice"(%4163) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4165 = "mix.prim.slice"(%4163) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4166 = "mix.prim.reshape"(%4158) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4167 = "mix.prim.reshape"(%4164) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4168 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4169 = "mix.prim.convert"(%4168) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4170 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %4171 = "mix.prim.div"(%4169, %4170) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %4172 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4173 = "mix.prim.pow"(%4172, %4171) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4174 = "mix.prim.reciprocal"(%4173) : (tensor<80xf16>) -> tensor<80xf16>
    %4175 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4176 = "mix.prim.mul"(%4175, %4174) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4177 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4178 = "mix.prim.unsqueeze"(%4177) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4179 = "mix.prim.permute"(%4178) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4180 = "mix.prim.unsqueeze"(%4176) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4181 = "mix.prim.permute"(%4180) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4182 = "mix.prim.mul"(%4179, %4181) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4183 = "mix.prim.concat"(%4182, %4182) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4184 = "mix.prim.cos"(%4183) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4185 = "mix.prim.slice"(%4184) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4186 = "mix.prim.unsqueeze"(%4185) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4187 = "mix.prim.slice"(%4186) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4188 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4189 = "mix.prim.mul"(%4187, %4188) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4190 = "mix.prim.sin"(%4183) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4191 = "mix.prim.slice"(%4190) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4192 = "mix.prim.unsqueeze"(%4191) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4193 = "mix.prim.slice"(%4192) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4194 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4195 = "mix.prim.mul"(%4193, %4194) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4196 = "mix.prim.slice"(%4189) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4197 = "mix.prim.slice"(%4195) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4198 = "mix.prim.mul"(%4166, %4196) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4199 = "mix.prim.slice"(%4166) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4200 = "mix.prim.slice"(%4166) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4201 = "mix.prim.neg"(%4200) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4202 = "mix.prim.concat"(%4201, %4199) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4203 = "mix.prim.mul"(%4202, %4197) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4204 = "mix.prim.add"(%4198, %4203) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4205 = "mix.prim.mul"(%4167, %4196) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4206 = "mix.prim.slice"(%4167) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4207 = "mix.prim.slice"(%4167) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4208 = "mix.prim.neg"(%4207) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4209 = "mix.prim.concat"(%4208, %4206) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4210 = "mix.prim.mul"(%4209, %4197) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4211 = "mix.prim.add"(%4205, %4210) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4212 = "mix.prim.reshape"(%4204) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4213 = "mix.prim.reshape"(%4211) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4214 = "mix.prim.reshape"(%4212) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4215 = "mix.prim.reshape"(%4213) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4216 = "mix.prim.transpose"(%4214) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4217 = "mix.prim.transpose"(%4215) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4218 = "mix.prim.transpose"(%4217) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4219 = "mix.prim.unsqueeze"(%4216) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4220 = "mix.prim.permute"(%4219) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4221 = "mix.prim.unsqueeze"(%4218) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4222 = "mix.prim.permute"(%4221) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4223 = "mix.prim.permute"(%4220) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4224 = "mix.prim.reshape"(%4223) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4225 = "mix.prim.permute"(%4222) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4226 = "mix.prim.reshape"(%4225) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4227 = "mix.prim.batch_matmul"(%4224, %4226) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4228 = "mix.prim.reshape"(%4227) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4229 = "mix.prim.permute"(%4228) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4230 = "mix.prim.reshape"(%4229) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4231 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4232 = "mix.prim.mul"(%4230, %4231) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4233 = "mix.prim.reshape"(%4232) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4234 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4235 = "mix.comp.masked_fill"(%4233, %14, %4234) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4236 = "mix.comp.softmax"(%4235) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4237 = "mix.prim.reshape"(%4236) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4238 = "mix.prim.reshape"(%4165) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4239 = "mix.prim.transpose"(%4238) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4240 = "mix.prim.batch_matmul"(%4237, %4239) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4241 = "mix.prim.reshape"(%4240) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4242 = "mix.prim.permute"(%4241) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4243 = "mix.prim.reshape"(%4242) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4244 = "mix.prim.reshape"(%4243) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4245 = "mix.prim.transpose"(%4151) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4246 = "mix.prim.matmul"(%4244, %4245) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4247 = "mix.prim.add"(%4246, %4152) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4248 = "mix.prim.reshape"(%4247) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4249 = "mix.prim.mul"(%4139, %4248) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4250 = "mix.comp.weight"() <{param_loc = "transformer.h.33.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4251 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4252 = "mix.prim.pow"(%4249, %4251) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4253 = "mix.comp.mean"(%4252) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4254 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4255 = "mix.prim.add"(%4253, %4254) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4256 = "mix.prim.rsqrt"(%4255) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4257 = "mix.prim.mul"(%4249, %4256) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4258 = "mix.prim.mul"(%4250, %4257) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4259 = "mix.module.linear"(%4258) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.33.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4260 = "mix.comp.silu"(%4259) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4261 = "mix.module.linear"(%4258) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.33.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4262 = "mix.prim.mul"(%4260, %4261) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4263 = "mix.module.linear"(%4262) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.33.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4264 = "mix.prim.add"(%4263, %4249) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4265 = "mix.comp.weight"() <{param_loc = "transformer.h.34.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4266 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4267 = "mix.prim.pow"(%4264, %4266) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4268 = "mix.comp.mean"(%4267) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4269 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4270 = "mix.prim.add"(%4268, %4269) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4271 = "mix.prim.rsqrt"(%4270) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4272 = "mix.prim.mul"(%4264, %4271) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4273 = "mix.prim.mul"(%4265, %4272) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4274 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4275 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4276 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4277 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4278 = "mix.prim.transpose"(%4273) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4279 = "mix.prim.transpose"(%4274) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4280 = "mix.prim.reshape"(%4278) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4281 = "mix.prim.matmul"(%4280, %4279) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4282 = "mix.prim.reshape"(%4281) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4283 = "mix.prim.reshape"(%4282) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4284 = "mix.prim.transpose"(%4275) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4285 = "mix.prim.reshape"(%4278) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4286 = "mix.prim.matmul"(%4285, %4284) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4287 = "mix.prim.reshape"(%4286) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4288 = "mix.prim.reshape"(%4287) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4289 = "mix.prim.slice"(%4288) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4290 = "mix.prim.slice"(%4288) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4291 = "mix.prim.reshape"(%4283) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4292 = "mix.prim.reshape"(%4289) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4293 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4294 = "mix.prim.convert"(%4293) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4295 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %4296 = "mix.prim.div"(%4294, %4295) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %4297 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4298 = "mix.prim.pow"(%4297, %4296) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4299 = "mix.prim.reciprocal"(%4298) : (tensor<80xf16>) -> tensor<80xf16>
    %4300 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4301 = "mix.prim.mul"(%4300, %4299) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4302 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4303 = "mix.prim.unsqueeze"(%4302) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4304 = "mix.prim.permute"(%4303) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4305 = "mix.prim.unsqueeze"(%4301) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4306 = "mix.prim.permute"(%4305) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4307 = "mix.prim.mul"(%4304, %4306) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4308 = "mix.prim.concat"(%4307, %4307) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4309 = "mix.prim.cos"(%4308) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4310 = "mix.prim.slice"(%4309) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4311 = "mix.prim.unsqueeze"(%4310) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4312 = "mix.prim.slice"(%4311) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4313 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4314 = "mix.prim.mul"(%4312, %4313) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4315 = "mix.prim.sin"(%4308) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4316 = "mix.prim.slice"(%4315) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4317 = "mix.prim.unsqueeze"(%4316) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4318 = "mix.prim.slice"(%4317) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4319 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4320 = "mix.prim.mul"(%4318, %4319) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4321 = "mix.prim.slice"(%4314) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4322 = "mix.prim.slice"(%4320) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4323 = "mix.prim.mul"(%4291, %4321) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4324 = "mix.prim.slice"(%4291) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4325 = "mix.prim.slice"(%4291) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4326 = "mix.prim.neg"(%4325) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4327 = "mix.prim.concat"(%4326, %4324) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4328 = "mix.prim.mul"(%4327, %4322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4329 = "mix.prim.add"(%4323, %4328) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4330 = "mix.prim.mul"(%4292, %4321) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4331 = "mix.prim.slice"(%4292) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4332 = "mix.prim.slice"(%4292) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4333 = "mix.prim.neg"(%4332) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4334 = "mix.prim.concat"(%4333, %4331) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4335 = "mix.prim.mul"(%4334, %4322) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4336 = "mix.prim.add"(%4330, %4335) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4337 = "mix.prim.reshape"(%4329) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4338 = "mix.prim.reshape"(%4336) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4339 = "mix.prim.reshape"(%4337) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4340 = "mix.prim.reshape"(%4338) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4341 = "mix.prim.transpose"(%4339) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4342 = "mix.prim.transpose"(%4340) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4343 = "mix.prim.transpose"(%4342) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4344 = "mix.prim.unsqueeze"(%4341) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4345 = "mix.prim.permute"(%4344) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4346 = "mix.prim.unsqueeze"(%4343) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4347 = "mix.prim.permute"(%4346) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4348 = "mix.prim.permute"(%4345) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4349 = "mix.prim.reshape"(%4348) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4350 = "mix.prim.permute"(%4347) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4351 = "mix.prim.reshape"(%4350) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4352 = "mix.prim.batch_matmul"(%4349, %4351) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4353 = "mix.prim.reshape"(%4352) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4354 = "mix.prim.permute"(%4353) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4355 = "mix.prim.reshape"(%4354) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4356 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4357 = "mix.prim.mul"(%4355, %4356) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4358 = "mix.prim.reshape"(%4357) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4359 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4360 = "mix.comp.masked_fill"(%4358, %14, %4359) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4361 = "mix.comp.softmax"(%4360) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4362 = "mix.prim.reshape"(%4361) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4363 = "mix.prim.reshape"(%4290) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4364 = "mix.prim.transpose"(%4363) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4365 = "mix.prim.batch_matmul"(%4362, %4364) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4366 = "mix.prim.reshape"(%4365) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4367 = "mix.prim.permute"(%4366) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4368 = "mix.prim.reshape"(%4367) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4369 = "mix.prim.reshape"(%4368) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4370 = "mix.prim.transpose"(%4276) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4371 = "mix.prim.matmul"(%4369, %4370) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4372 = "mix.prim.add"(%4371, %4277) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4373 = "mix.prim.reshape"(%4372) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4374 = "mix.prim.mul"(%4264, %4373) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4375 = "mix.comp.weight"() <{param_loc = "transformer.h.34.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4376 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4377 = "mix.prim.pow"(%4374, %4376) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4378 = "mix.comp.mean"(%4377) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4379 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4380 = "mix.prim.add"(%4378, %4379) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4381 = "mix.prim.rsqrt"(%4380) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4382 = "mix.prim.mul"(%4374, %4381) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4383 = "mix.prim.mul"(%4375, %4382) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4384 = "mix.module.linear"(%4383) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.34.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4385 = "mix.comp.silu"(%4384) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4386 = "mix.module.linear"(%4383) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.34.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4387 = "mix.prim.mul"(%4385, %4386) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4388 = "mix.module.linear"(%4387) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.34.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4389 = "mix.prim.add"(%4388, %4374) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4390 = "mix.comp.weight"() <{param_loc = "transformer.h.35.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4391 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4392 = "mix.prim.pow"(%4389, %4391) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4393 = "mix.comp.mean"(%4392) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4394 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4395 = "mix.prim.add"(%4393, %4394) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4396 = "mix.prim.rsqrt"(%4395) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4397 = "mix.prim.mul"(%4389, %4396) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4398 = "mix.prim.mul"(%4390, %4397) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4399 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4400 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4401 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4402 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4403 = "mix.prim.transpose"(%4398) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4404 = "mix.prim.transpose"(%4399) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4405 = "mix.prim.reshape"(%4403) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4406 = "mix.prim.matmul"(%4405, %4404) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4407 = "mix.prim.reshape"(%4406) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4408 = "mix.prim.reshape"(%4407) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4409 = "mix.prim.transpose"(%4400) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4410 = "mix.prim.reshape"(%4403) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4411 = "mix.prim.matmul"(%4410, %4409) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4412 = "mix.prim.reshape"(%4411) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4413 = "mix.prim.reshape"(%4412) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4414 = "mix.prim.slice"(%4413) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4415 = "mix.prim.slice"(%4413) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4416 = "mix.prim.reshape"(%4408) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4417 = "mix.prim.reshape"(%4414) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4418 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4419 = "mix.prim.convert"(%4418) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4420 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %4421 = "mix.prim.div"(%4419, %4420) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %4422 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4423 = "mix.prim.pow"(%4422, %4421) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4424 = "mix.prim.reciprocal"(%4423) : (tensor<80xf16>) -> tensor<80xf16>
    %4425 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4426 = "mix.prim.mul"(%4425, %4424) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4427 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4428 = "mix.prim.unsqueeze"(%4427) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4429 = "mix.prim.permute"(%4428) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4430 = "mix.prim.unsqueeze"(%4426) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4431 = "mix.prim.permute"(%4430) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4432 = "mix.prim.mul"(%4429, %4431) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4433 = "mix.prim.concat"(%4432, %4432) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4434 = "mix.prim.cos"(%4433) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4435 = "mix.prim.slice"(%4434) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4436 = "mix.prim.unsqueeze"(%4435) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4437 = "mix.prim.slice"(%4436) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4438 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4439 = "mix.prim.mul"(%4437, %4438) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4440 = "mix.prim.sin"(%4433) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4441 = "mix.prim.slice"(%4440) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4442 = "mix.prim.unsqueeze"(%4441) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4443 = "mix.prim.slice"(%4442) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4444 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4445 = "mix.prim.mul"(%4443, %4444) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4446 = "mix.prim.slice"(%4439) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4447 = "mix.prim.slice"(%4445) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4448 = "mix.prim.mul"(%4416, %4446) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4449 = "mix.prim.slice"(%4416) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4450 = "mix.prim.slice"(%4416) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4451 = "mix.prim.neg"(%4450) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4452 = "mix.prim.concat"(%4451, %4449) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4453 = "mix.prim.mul"(%4452, %4447) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4454 = "mix.prim.add"(%4448, %4453) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4455 = "mix.prim.mul"(%4417, %4446) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4456 = "mix.prim.slice"(%4417) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4457 = "mix.prim.slice"(%4417) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4458 = "mix.prim.neg"(%4457) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4459 = "mix.prim.concat"(%4458, %4456) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4460 = "mix.prim.mul"(%4459, %4447) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4461 = "mix.prim.add"(%4455, %4460) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4462 = "mix.prim.reshape"(%4454) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4463 = "mix.prim.reshape"(%4461) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4464 = "mix.prim.reshape"(%4462) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4465 = "mix.prim.reshape"(%4463) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4466 = "mix.prim.transpose"(%4464) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4467 = "mix.prim.transpose"(%4465) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4468 = "mix.prim.transpose"(%4467) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4469 = "mix.prim.unsqueeze"(%4466) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4470 = "mix.prim.permute"(%4469) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4471 = "mix.prim.unsqueeze"(%4468) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4472 = "mix.prim.permute"(%4471) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4473 = "mix.prim.permute"(%4470) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4474 = "mix.prim.reshape"(%4473) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4475 = "mix.prim.permute"(%4472) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4476 = "mix.prim.reshape"(%4475) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4477 = "mix.prim.batch_matmul"(%4474, %4476) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4478 = "mix.prim.reshape"(%4477) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4479 = "mix.prim.permute"(%4478) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4480 = "mix.prim.reshape"(%4479) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4481 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4482 = "mix.prim.mul"(%4480, %4481) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4483 = "mix.prim.reshape"(%4482) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4484 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4485 = "mix.comp.masked_fill"(%4483, %14, %4484) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4486 = "mix.comp.softmax"(%4485) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4487 = "mix.prim.reshape"(%4486) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4488 = "mix.prim.reshape"(%4415) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4489 = "mix.prim.transpose"(%4488) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4490 = "mix.prim.batch_matmul"(%4487, %4489) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4491 = "mix.prim.reshape"(%4490) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4492 = "mix.prim.permute"(%4491) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4493 = "mix.prim.reshape"(%4492) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4494 = "mix.prim.reshape"(%4493) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4495 = "mix.prim.transpose"(%4401) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4496 = "mix.prim.matmul"(%4494, %4495) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4497 = "mix.prim.add"(%4496, %4402) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4498 = "mix.prim.reshape"(%4497) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4499 = "mix.prim.mul"(%4389, %4498) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4500 = "mix.comp.weight"() <{param_loc = "transformer.h.35.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4501 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4502 = "mix.prim.pow"(%4499, %4501) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4503 = "mix.comp.mean"(%4502) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4504 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4505 = "mix.prim.add"(%4503, %4504) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4506 = "mix.prim.rsqrt"(%4505) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4507 = "mix.prim.mul"(%4499, %4506) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4508 = "mix.prim.mul"(%4500, %4507) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4509 = "mix.module.linear"(%4508) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.35.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4510 = "mix.comp.silu"(%4509) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4511 = "mix.module.linear"(%4508) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.35.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4512 = "mix.prim.mul"(%4510, %4511) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4513 = "mix.module.linear"(%4512) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.35.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4514 = "mix.prim.add"(%4513, %4499) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4515 = "mix.comp.weight"() <{param_loc = "transformer.h.36.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4516 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4517 = "mix.prim.pow"(%4514, %4516) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4518 = "mix.comp.mean"(%4517) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4519 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4520 = "mix.prim.add"(%4518, %4519) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4521 = "mix.prim.rsqrt"(%4520) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4522 = "mix.prim.mul"(%4514, %4521) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4523 = "mix.prim.mul"(%4515, %4522) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4524 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4525 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4526 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4527 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4528 = "mix.prim.transpose"(%4523) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4529 = "mix.prim.transpose"(%4524) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4530 = "mix.prim.reshape"(%4528) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4531 = "mix.prim.matmul"(%4530, %4529) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4532 = "mix.prim.reshape"(%4531) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4533 = "mix.prim.reshape"(%4532) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4534 = "mix.prim.transpose"(%4525) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4535 = "mix.prim.reshape"(%4528) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4536 = "mix.prim.matmul"(%4535, %4534) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4537 = "mix.prim.reshape"(%4536) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4538 = "mix.prim.reshape"(%4537) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4539 = "mix.prim.slice"(%4538) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4540 = "mix.prim.slice"(%4538) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4541 = "mix.prim.reshape"(%4533) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4542 = "mix.prim.reshape"(%4539) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4543 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4544 = "mix.prim.convert"(%4543) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4545 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %4546 = "mix.prim.div"(%4544, %4545) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %4547 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4548 = "mix.prim.pow"(%4547, %4546) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4549 = "mix.prim.reciprocal"(%4548) : (tensor<80xf16>) -> tensor<80xf16>
    %4550 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4551 = "mix.prim.mul"(%4550, %4549) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4552 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4553 = "mix.prim.unsqueeze"(%4552) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4554 = "mix.prim.permute"(%4553) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4555 = "mix.prim.unsqueeze"(%4551) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4556 = "mix.prim.permute"(%4555) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4557 = "mix.prim.mul"(%4554, %4556) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4558 = "mix.prim.concat"(%4557, %4557) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4559 = "mix.prim.cos"(%4558) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4560 = "mix.prim.slice"(%4559) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4561 = "mix.prim.unsqueeze"(%4560) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4562 = "mix.prim.slice"(%4561) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4563 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4564 = "mix.prim.mul"(%4562, %4563) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4565 = "mix.prim.sin"(%4558) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4566 = "mix.prim.slice"(%4565) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4567 = "mix.prim.unsqueeze"(%4566) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4568 = "mix.prim.slice"(%4567) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4569 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4570 = "mix.prim.mul"(%4568, %4569) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4571 = "mix.prim.slice"(%4564) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4572 = "mix.prim.slice"(%4570) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4573 = "mix.prim.mul"(%4541, %4571) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4574 = "mix.prim.slice"(%4541) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4575 = "mix.prim.slice"(%4541) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4576 = "mix.prim.neg"(%4575) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4577 = "mix.prim.concat"(%4576, %4574) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4578 = "mix.prim.mul"(%4577, %4572) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4579 = "mix.prim.add"(%4573, %4578) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4580 = "mix.prim.mul"(%4542, %4571) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4581 = "mix.prim.slice"(%4542) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4582 = "mix.prim.slice"(%4542) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4583 = "mix.prim.neg"(%4582) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4584 = "mix.prim.concat"(%4583, %4581) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4585 = "mix.prim.mul"(%4584, %4572) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4586 = "mix.prim.add"(%4580, %4585) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4587 = "mix.prim.reshape"(%4579) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4588 = "mix.prim.reshape"(%4586) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4589 = "mix.prim.reshape"(%4587) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4590 = "mix.prim.reshape"(%4588) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4591 = "mix.prim.transpose"(%4589) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4592 = "mix.prim.transpose"(%4590) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4593 = "mix.prim.transpose"(%4592) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4594 = "mix.prim.unsqueeze"(%4591) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4595 = "mix.prim.permute"(%4594) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4596 = "mix.prim.unsqueeze"(%4593) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4597 = "mix.prim.permute"(%4596) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4598 = "mix.prim.permute"(%4595) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4599 = "mix.prim.reshape"(%4598) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4600 = "mix.prim.permute"(%4597) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4601 = "mix.prim.reshape"(%4600) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4602 = "mix.prim.batch_matmul"(%4599, %4601) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4603 = "mix.prim.reshape"(%4602) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4604 = "mix.prim.permute"(%4603) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4605 = "mix.prim.reshape"(%4604) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4606 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4607 = "mix.prim.mul"(%4605, %4606) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4608 = "mix.prim.reshape"(%4607) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4609 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4610 = "mix.comp.masked_fill"(%4608, %14, %4609) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4611 = "mix.comp.softmax"(%4610) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4612 = "mix.prim.reshape"(%4611) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4613 = "mix.prim.reshape"(%4540) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4614 = "mix.prim.transpose"(%4613) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4615 = "mix.prim.batch_matmul"(%4612, %4614) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4616 = "mix.prim.reshape"(%4615) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4617 = "mix.prim.permute"(%4616) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4618 = "mix.prim.reshape"(%4617) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4619 = "mix.prim.reshape"(%4618) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4620 = "mix.prim.transpose"(%4526) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4621 = "mix.prim.matmul"(%4619, %4620) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4622 = "mix.prim.add"(%4621, %4527) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4623 = "mix.prim.reshape"(%4622) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4624 = "mix.prim.mul"(%4514, %4623) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4625 = "mix.comp.weight"() <{param_loc = "transformer.h.36.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4626 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4627 = "mix.prim.pow"(%4624, %4626) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4628 = "mix.comp.mean"(%4627) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4629 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4630 = "mix.prim.add"(%4628, %4629) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4631 = "mix.prim.rsqrt"(%4630) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4632 = "mix.prim.mul"(%4624, %4631) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4633 = "mix.prim.mul"(%4625, %4632) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4634 = "mix.module.linear"(%4633) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.36.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4635 = "mix.comp.silu"(%4634) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4636 = "mix.module.linear"(%4633) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.36.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4637 = "mix.prim.mul"(%4635, %4636) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4638 = "mix.module.linear"(%4637) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.36.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4639 = "mix.prim.add"(%4638, %4624) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4640 = "mix.comp.weight"() <{param_loc = "transformer.h.37.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4641 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4642 = "mix.prim.pow"(%4639, %4641) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4643 = "mix.comp.mean"(%4642) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4644 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4645 = "mix.prim.add"(%4643, %4644) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4646 = "mix.prim.rsqrt"(%4645) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4647 = "mix.prim.mul"(%4639, %4646) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4648 = "mix.prim.mul"(%4640, %4647) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4649 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4650 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4651 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4652 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4653 = "mix.prim.transpose"(%4648) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4654 = "mix.prim.transpose"(%4649) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4655 = "mix.prim.reshape"(%4653) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4656 = "mix.prim.matmul"(%4655, %4654) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4657 = "mix.prim.reshape"(%4656) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4658 = "mix.prim.reshape"(%4657) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4659 = "mix.prim.transpose"(%4650) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4660 = "mix.prim.reshape"(%4653) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4661 = "mix.prim.matmul"(%4660, %4659) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4662 = "mix.prim.reshape"(%4661) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4663 = "mix.prim.reshape"(%4662) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4664 = "mix.prim.slice"(%4663) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4665 = "mix.prim.slice"(%4663) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4666 = "mix.prim.reshape"(%4658) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4667 = "mix.prim.reshape"(%4664) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4668 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4669 = "mix.prim.convert"(%4668) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4670 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %4671 = "mix.prim.div"(%4669, %4670) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %4672 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4673 = "mix.prim.pow"(%4672, %4671) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4674 = "mix.prim.reciprocal"(%4673) : (tensor<80xf16>) -> tensor<80xf16>
    %4675 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4676 = "mix.prim.mul"(%4675, %4674) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4677 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01, 3.300000e+01, 3.400000e+01, 3.500000e+01, 3.600000e+01, 3.700000e+01, 3.800000e+01, 3.900000e+01]> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4678 = "mix.prim.unsqueeze"(%4677) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4679 = "mix.prim.permute"(%4678) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4680 = "mix.prim.unsqueeze"(%4676) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4681 = "mix.prim.permute"(%4680) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4682 = "mix.prim.mul"(%4679, %4681) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4683 = "mix.prim.concat"(%4682, %4682) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4684 = "mix.prim.cos"(%4683) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4685 = "mix.prim.slice"(%4684) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4686 = "mix.prim.unsqueeze"(%4685) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4687 = "mix.prim.slice"(%4686) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4688 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4689 = "mix.prim.mul"(%4687, %4688) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4690 = "mix.prim.sin"(%4683) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4691 = "mix.prim.slice"(%4690) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4692 = "mix.prim.unsqueeze"(%4691) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4693 = "mix.prim.slice"(%4692) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4694 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4695 = "mix.prim.mul"(%4693, %4694) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4696 = "mix.prim.slice"(%4689) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4697 = "mix.prim.slice"(%4695) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4698 = "mix.prim.mul"(%4666, %4696) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4699 = "mix.prim.slice"(%4666) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4700 = "mix.prim.slice"(%4666) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4701 = "mix.prim.neg"(%4700) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4702 = "mix.prim.concat"(%4701, %4699) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4703 = "mix.prim.mul"(%4702, %4697) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4704 = "mix.prim.add"(%4698, %4703) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4705 = "mix.prim.mul"(%4667, %4696) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4706 = "mix.prim.slice"(%4667) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4707 = "mix.prim.slice"(%4667) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4708 = "mix.prim.neg"(%4707) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4709 = "mix.prim.concat"(%4708, %4706) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4710 = "mix.prim.mul"(%4709, %4697) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4711 = "mix.prim.add"(%4705, %4710) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4712 = "mix.prim.reshape"(%4704) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4713 = "mix.prim.reshape"(%4711) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4714 = "mix.prim.reshape"(%4712) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4715 = "mix.prim.reshape"(%4713) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4716 = "mix.prim.transpose"(%4714) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4717 = "mix.prim.transpose"(%4715) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4718 = "mix.prim.transpose"(%4717) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4719 = "mix.prim.unsqueeze"(%4716) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4720 = "mix.prim.permute"(%4719) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4721 = "mix.prim.unsqueeze"(%4718) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4722 = "mix.prim.permute"(%4721) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4723 = "mix.prim.permute"(%4720) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4724 = "mix.prim.reshape"(%4723) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4725 = "mix.prim.permute"(%4722) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4726 = "mix.prim.reshape"(%4725) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4727 = "mix.prim.batch_matmul"(%4724, %4726) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4728 = "mix.prim.reshape"(%4727) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4729 = "mix.prim.permute"(%4728) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4730 = "mix.prim.reshape"(%4729) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4731 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4732 = "mix.prim.mul"(%4730, %4731) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4733 = "mix.prim.reshape"(%4732) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4734 = "mix.prim.constant"() <{value = 0xFC00 : f16}> : () -> f16
    %4735 = "mix.comp.masked_fill"(%4733, %14, %4734) : (tensor<1x32x40x40xf16>, tensor<1x1x40x40xi1>, f16) -> tensor<1x32x40x40xf16>
    %4736 = "mix.comp.softmax"(%4735) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4737 = "mix.prim.reshape"(%4736) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4738 = "mix.prim.reshape"(%4665) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4739 = "mix.prim.transpose"(%4738) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4740 = "mix.prim.batch_matmul"(%4737, %4739) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4741 = "mix.prim.reshape"(%4740) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4742 = "mix.prim.permute"(%4741) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4743 = "mix.prim.reshape"(%4742) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4744 = "mix.prim.reshape"(%4743) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4745 = "mix.prim.transpose"(%4651) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4746 = "mix.prim.matmul"(%4744, %4745) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4747 = "mix.prim.add"(%4746, %4652) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4748 = "mix.prim.reshape"(%4747) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4749 = "mix.prim.mul"(%4639, %4748) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4750 = "mix.comp.weight"() <{param_loc = "transformer.h.37.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4751 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4752 = "mix.prim.pow"(%4749, %4751) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4753 = "mix.comp.mean"(%4752) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4754 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4755 = "mix.prim.add"(%4753, %4754) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4756 = "mix.prim.rsqrt"(%4755) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4757 = "mix.prim.mul"(%4749, %4756) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4758 = "mix.prim.mul"(%4750, %4757) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4759 = "mix.module.linear"(%4758) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.37.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4760 = "mix.comp.silu"(%4759) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4761 = "mix.module.linear"(%4758) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.37.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4762 = "mix.prim.mul"(%4760, %4761) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4763 = "mix.module.linear"(%4762) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.37.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4764 = "mix.prim.add"(%4763, %4749) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4765 = "mix.comp.weight"() <{param_loc = "transformer.ln_f.weight"}> : () -> tensor<5120xf16>
    %4766 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4767 = "mix.prim.pow"(%4764, %4766) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4768 = "mix.comp.mean"(%4767) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4769 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4770 = "mix.prim.add"(%4768, %4769) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4771 = "mix.prim.rsqrt"(%4770) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4772 = "mix.prim.mul"(%4764, %4771) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4773 = "mix.prim.mul"(%4765, %4772) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4774 = "mix.module.linear"(%4773) <{dtype = f16, in_feature = 5120 : i32, out_feature = 120000 : i32, params_loc = "lm_head"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x120000xf16>
    return %4774 : tensor<1x40x120000xf16>
  }
  func.func private @main() {
    %0 = "mix.prim.constant"() <{value = dense<1> : tensor<1x40xi32>}> : () -> tensor<1x40xi32>
    %1 = call @Telechat(%0) : (tensor<1x40xi32>) -> tensor<1x40x120000xf16>
    %cast = tensor.cast %1 : tensor<1x40x120000xf16> to tensor<*xf16>
    call @printMemrefF16(%cast) : (tensor<*xf16>) -> ()
    return
  }
}
