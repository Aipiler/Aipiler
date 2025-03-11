module {
  func.func private @printMemrefF16(tensor<*xf16>)
  func.func private @Telechat(%arg0: tensor<5xi32>) -> tensor<5x120000xf16> {
    %0 = "mix.module.embedding"(%arg0) <{dtype = f16, embedding_dim = 5120 : i32, num_embeddings = 120000 : i32, params_loc = "transformer.word_embeddings"}> : (tensor<5xi32>) -> tensor<5x5120xf16>
    %1 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %2 = "mix.prim.slice"(%1) <{dim = 0 : i64, end = 9223372036854775807 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5xi16>) -> tensor<5xi16>
    %3 = "mix.prim.unsqueeze"(%2) <{axis = 1 : i32}> : (tensor<5xi16>) -> tensor<5x1xi16>
    %4 = "mix.prim.unsqueeze"(%1) <{axis = 0 : i32}> : (tensor<5xi16>) -> tensor<1x5xi16>
    %5 = "mix.prim.slice"(%4) <{dim = 1 : i64, end = 9223372036854775807 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<1x5xi16>) -> tensor<1x5xi16>
    %6 = "mix.prim.lt"(%3, %5) : (tensor<5x1xi16>, tensor<1x5xi16>) -> tensor<5x5xi1>
    %7 = "mix.prim.constant"() <{value = dense<true> : tensor<5x0xi1>}> : () -> tensor<5x0xi1>
    %8 = "mix.prim.constant"() <{value = dense<true> : tensor<0x5xi1>}> : () -> tensor<0x5xi1>
    %9 = "mix.prim.constant"() <{value = dense<true> : tensor<0x0xi1>}> : () -> tensor<0x0xi1>
    %10 = "mix.prim.concat"(%6, %7) <{axis = 1 : i64}> : (tensor<5x5xi1>, tensor<5x0xi1>) -> tensor<5x5xi1>
    %11 = "mix.prim.concat"(%8, %9) <{axis = 1 : i64}> : (tensor<0x5xi1>, tensor<0x0xi1>) -> tensor<0x5xi1>
    %12 = "mix.prim.concat"(%10, %11) <{axis = 0 : i64}> : (tensor<5x5xi1>, tensor<0x5xi1>) -> tensor<5x5xi1>
    %13 = "mix.prim.unsqueeze"(%12) <{axis = 0 : i32}> : (tensor<5x5xi1>) -> tensor<1x5x5xi1>
    %14 = "mix.comp.weight"() <{param_loc = "transformer.h.0.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %15 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %16 = "mix.prim.pow"(%0, %15) : (tensor<5x5120xf16>, tensor<1xf16>) -> tensor<5x5120xf16>
    %17 = "mix.comp.mean"(%16) <{dims = [1 : i32], keepDim = true}> : (tensor<5x5120xf16>) -> tensor<5x1xf16>
    %18 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %19 = "mix.prim.add"(%17, %18) : (tensor<5x1xf16>, f16) -> tensor<5x1xf16>
    %20 = "mix.prim.rsqrt"(%19) : (tensor<5x1xf16>) -> tensor<5x1xf16>
    %21 = "mix.prim.mul"(%0, %20) : (tensor<5x5120xf16>, tensor<5x1xf16>) -> tensor<5x5120xf16>
    %22 = "mix.prim.mul"(%14, %21) : (tensor<5120xf16>, tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %23 = "mix.module.linear"(%22) <{dtype = f16, in_feature = 5120 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.0.self_attention.query"}> : (tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %24 = "mix.prim.reshape"(%23) <{shape = [5, 32, 160]}> : (tensor<5x5120xf16>) -> tensor<5x32x160xf16>
    %25 = "mix.module.linear"(%22) <{dtype = f16, in_feature = 5120 : i32, out_feature = 10240 : i32, params_loc = "transformer.h.0.self_attention.key_value"}> : (tensor<5x5120xf16>) -> tensor<5x10240xf16>
    %26 = "mix.prim.reshape"(%25) <{shape = [5, 32, 320]}> : (tensor<5x10240xf16>) -> tensor<5x32x320xf16>
    %27 = "mix.prim.slice"(%26) <{dim = 2 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x32x320xf16>) -> tensor<5x32x160xf16>
    %28 = "mix.prim.slice"(%26) <{dim = 2 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<5x32x320xf16>) -> tensor<5x32x160xf16>
    %29 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %30 = "mix.prim.convert"(%29) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %31 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %32 = "mix.prim.div"(%30, %31) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %33 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %34 = "mix.prim.pow"(%33, %32) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %35 = "mix.prim.reciprocal"(%34) : (tensor<80xf16>) -> tensor<80xf16>
    %36 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %37 = "mix.prim.mul"(%36, %35) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %38 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<5xf16>}> : () -> tensor<5xf16>
    %39 = "mix.prim.unsqueeze"(%38) <{axis = 1 : i32}> : (tensor<5xf16>) -> tensor<5x1xf16>
    %40 = "mix.prim.permute"(%39) <{dims = [0, 1]}> : (tensor<5x1xf16>) -> tensor<5x1xf16>
    %41 = "mix.prim.unsqueeze"(%37) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %42 = "mix.prim.permute"(%41) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %43 = "mix.prim.mul"(%40, %42) : (tensor<5x1xf16>, tensor<1x80xf16>) -> tensor<5x80xf16>
    %44 = "mix.prim.concat"(%43, %43) <{axis = 1 : i64}> : (tensor<5x80xf16>, tensor<5x80xf16>) -> tensor<5x160xf16>
    %45 = "mix.prim.cos"(%44) : (tensor<5x160xf16>) -> tensor<5x160xf16>
    %46 = "mix.prim.slice"(%45) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x160xf16>) -> tensor<5x160xf16>
    %47 = "mix.prim.unsqueeze"(%46) <{axis = 1 : i32}> : (tensor<5x160xf16>) -> tensor<5x1x160xf16>
    %48 = "mix.prim.slice"(%47) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x1x160xf16>) -> tensor<5x1x160xf16>
    %49 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %50 = "mix.prim.mul"(%48, %49) : (tensor<5x1x160xf16>, f16) -> tensor<5x1x160xf16>
    %51 = "mix.prim.sin"(%44) : (tensor<5x160xf16>) -> tensor<5x160xf16>
    %52 = "mix.prim.slice"(%51) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x160xf16>) -> tensor<5x160xf16>
    %53 = "mix.prim.unsqueeze"(%52) <{axis = 1 : i32}> : (tensor<5x160xf16>) -> tensor<5x1x160xf16>
    %54 = "mix.prim.slice"(%53) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x1x160xf16>) -> tensor<5x1x160xf16>
    %55 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %56 = "mix.prim.mul"(%54, %55) : (tensor<5x1x160xf16>, f16) -> tensor<5x1x160xf16>
    %57 = "mix.prim.mul"(%24, %50) : (tensor<5x32x160xf16>, tensor<5x1x160xf16>) -> tensor<5x32x160xf16>
    %58 = "mix.prim.slice"(%24) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x32x160xf16>) -> tensor<5x32x80xf16>
    %59 = "mix.prim.slice"(%24) <{dim = 2 : i64, end = 160 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<5x32x160xf16>) -> tensor<5x32x80xf16>
    %60 = "mix.prim.neg"(%59) : (tensor<5x32x80xf16>) -> tensor<5x32x80xf16>
    %61 = "mix.prim.concat"(%60, %58) <{axis = 2 : i64}> : (tensor<5x32x80xf16>, tensor<5x32x80xf16>) -> tensor<5x32x160xf16>
    %62 = "mix.prim.mul"(%61, %24) : (tensor<5x32x160xf16>, tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %63 = "mix.prim.add"(%57, %62) : (tensor<5x32x160xf16>, tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %64 = "mix.prim.mul"(%27, %50) : (tensor<5x32x160xf16>, tensor<5x1x160xf16>) -> tensor<5x32x160xf16>
    %65 = "mix.prim.slice"(%27) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x32x160xf16>) -> tensor<5x32x80xf16>
    %66 = "mix.prim.slice"(%27) <{dim = 2 : i64, end = 160 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<5x32x160xf16>) -> tensor<5x32x80xf16>
    %67 = "mix.prim.neg"(%66) : (tensor<5x32x80xf16>) -> tensor<5x32x80xf16>
    %68 = "mix.prim.concat"(%67, %65) <{axis = 2 : i64}> : (tensor<5x32x80xf16>, tensor<5x32x80xf16>) -> tensor<5x32x160xf16>
    %69 = "mix.prim.mul"(%68, %27) : (tensor<5x32x160xf16>, tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %70 = "mix.prim.add"(%64, %69) : (tensor<5x32x160xf16>, tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %71 = "mix.prim.reshape"(%63) <{shape = [5, 32, 160]}> : (tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %72 = "mix.prim.reshape"(%70) <{shape = [5, 32, 160]}> : (tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %73 = "mix.prim.transpose"(%71) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5x32x160xf16>) -> tensor<32x5x160xf16>
    %74 = "mix.prim.transpose"(%72) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5x32x160xf16>) -> tensor<32x5x160xf16>
    %75 = "mix.prim.transpose"(%74) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x5x160xf16>) -> tensor<32x160x5xf16>
    %76 = "mix.prim.batch_matmul"(%73, %75) : (tensor<32x5x160xf16>, tensor<32x160x5xf16>) -> tensor<32x5x5xf16>
    %77 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %78 = "mix.prim.mul"(%76, %77) : (tensor<32x5x5xf16>, f16) -> tensor<32x5x5xf16>
    %79 = "mix.prim.constant"() <{value = -6.550400e+04 : f16}> : () -> f16
    %80 = "mix.prim.masked_fill"(%78, %13, %79) : (tensor<32x5x5xf16>, tensor<1x5x5xi1>, f16) -> tensor<32x5x5xf16>
    %81 = "mix.prim.softmax"(%80) <{axis = -1 : si32}> : (tensor<32x5x5xf16>) -> tensor<32x5x5xf16>
    %82 = "mix.prim.transpose"(%28) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5x32x160xf16>) -> tensor<32x5x160xf16>
    %83 = "mix.prim.batch_matmul"(%81, %82) : (tensor<32x5x5xf16>, tensor<32x5x160xf16>) -> tensor<32x5x160xf16>
    %84 = "mix.prim.permute"(%83) <{dims = [1, 0, 2]}> : (tensor<32x5x160xf16>) -> tensor<5x32x160xf16>
    %85 = "mix.prim.reshape"(%84) <{shape = [5, 5120]}> : (tensor<5x32x160xf16>) -> tensor<5x5120xf16>
    %86 = "mix.module.linear"(%85) <{dtype = f16, has_bias, in_feature = 5120 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.0.self_attention.dense"}> : (tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %87 = "mix.prim.add"(%0, %86) : (tensor<5x5120xf16>, tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %88 = "mix.comp.weight"() <{param_loc = "transformer.h.0.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %89 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %90 = "mix.prim.pow"(%87, %89) : (tensor<5x5120xf16>, tensor<1xf16>) -> tensor<5x5120xf16>
    %91 = "mix.comp.mean"(%90) <{dims = [1 : i32], keepDim = true}> : (tensor<5x5120xf16>) -> tensor<5x1xf16>
    %92 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %93 = "mix.prim.add"(%91, %92) : (tensor<5x1xf16>, f16) -> tensor<5x1xf16>
    %94 = "mix.prim.rsqrt"(%93) : (tensor<5x1xf16>) -> tensor<5x1xf16>
    %95 = "mix.prim.mul"(%87, %94) : (tensor<5x5120xf16>, tensor<5x1xf16>) -> tensor<5x5120xf16>
    %96 = "mix.prim.mul"(%88, %95) : (tensor<5120xf16>, tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %97 = "mix.module.linear"(%96) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.0.mlp.gate_proj"}> : (tensor<5x5120xf16>) -> tensor<5x12288xf16>
    %98 = "mix.comp.silu"(%97) : (tensor<5x12288xf16>) -> tensor<5x12288xf16>
    %99 = "mix.module.linear"(%96) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.0.mlp.up_proj"}> : (tensor<5x5120xf16>) -> tensor<5x12288xf16>
    %100 = "mix.prim.mul"(%98, %99) : (tensor<5x12288xf16>, tensor<5x12288xf16>) -> tensor<5x12288xf16>
    %101 = "mix.module.linear"(%100) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.0.mlp.down_proj"}> : (tensor<5x12288xf16>) -> tensor<5x5120xf16>
    %102 = "mix.prim.add"(%101, %87) : (tensor<5x5120xf16>, tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %103 = "mix.comp.weight"() <{param_loc = "transformer.h.1.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %104 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %105 = "mix.prim.pow"(%102, %104) : (tensor<5x5120xf16>, tensor<1xf16>) -> tensor<5x5120xf16>
    %106 = "mix.comp.mean"(%105) <{dims = [1 : i32], keepDim = true}> : (tensor<5x5120xf16>) -> tensor<5x1xf16>
    %107 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %108 = "mix.prim.add"(%106, %107) : (tensor<5x1xf16>, f16) -> tensor<5x1xf16>
    %109 = "mix.prim.rsqrt"(%108) : (tensor<5x1xf16>) -> tensor<5x1xf16>
    %110 = "mix.prim.mul"(%102, %109) : (tensor<5x5120xf16>, tensor<5x1xf16>) -> tensor<5x5120xf16>
    %111 = "mix.prim.mul"(%103, %110) : (tensor<5120xf16>, tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %112 = "mix.module.linear"(%111) <{dtype = f16, in_feature = 5120 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.1.self_attention.query"}> : (tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %113 = "mix.prim.reshape"(%112) <{shape = [5, 32, 160]}> : (tensor<5x5120xf16>) -> tensor<5x32x160xf16>
    %114 = "mix.module.linear"(%111) <{dtype = f16, in_feature = 5120 : i32, out_feature = 10240 : i32, params_loc = "transformer.h.1.self_attention.key_value"}> : (tensor<5x5120xf16>) -> tensor<5x10240xf16>
    %115 = "mix.prim.reshape"(%114) <{shape = [5, 32, 320]}> : (tensor<5x10240xf16>) -> tensor<5x32x320xf16>
    %116 = "mix.prim.slice"(%115) <{dim = 2 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x32x320xf16>) -> tensor<5x32x160xf16>
    %117 = "mix.prim.slice"(%115) <{dim = 2 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<5x32x320xf16>) -> tensor<5x32x160xf16>
    %118 = "mix.prim.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150, 152, 154, 156, 158]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %119 = "mix.prim.convert"(%118) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %120 = "mix.prim.constant"() <{value = 1.600000e+02 : f16}> : () -> f16
    %121 = "mix.prim.div"(%119, %120) : (tensor<80xf16>, f16) -> tensor<80xf16>
    %122 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %123 = "mix.prim.pow"(%122, %121) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %124 = "mix.prim.reciprocal"(%123) : (tensor<80xf16>) -> tensor<80xf16>
    %125 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %126 = "mix.prim.mul"(%125, %124) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %127 = "mix.prim.constant"() <{value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<5xf16>}> : () -> tensor<5xf16>
    %128 = "mix.prim.unsqueeze"(%127) <{axis = 1 : i32}> : (tensor<5xf16>) -> tensor<5x1xf16>
    %129 = "mix.prim.permute"(%128) <{dims = [0, 1]}> : (tensor<5x1xf16>) -> tensor<5x1xf16>
    %130 = "mix.prim.unsqueeze"(%126) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %131 = "mix.prim.permute"(%130) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %132 = "mix.prim.mul"(%129, %131) : (tensor<5x1xf16>, tensor<1x80xf16>) -> tensor<5x80xf16>
    %133 = "mix.prim.concat"(%132, %132) <{axis = 1 : i64}> : (tensor<5x80xf16>, tensor<5x80xf16>) -> tensor<5x160xf16>
    %134 = "mix.prim.cos"(%133) : (tensor<5x160xf16>) -> tensor<5x160xf16>
    %135 = "mix.prim.slice"(%134) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x160xf16>) -> tensor<5x160xf16>
    %136 = "mix.prim.unsqueeze"(%135) <{axis = 1 : i32}> : (tensor<5x160xf16>) -> tensor<5x1x160xf16>
    %137 = "mix.prim.slice"(%136) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x1x160xf16>) -> tensor<5x1x160xf16>
    %138 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %139 = "mix.prim.mul"(%137, %138) : (tensor<5x1x160xf16>, f16) -> tensor<5x1x160xf16>
    %140 = "mix.prim.sin"(%133) : (tensor<5x160xf16>) -> tensor<5x160xf16>
    %141 = "mix.prim.slice"(%140) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x160xf16>) -> tensor<5x160xf16>
    %142 = "mix.prim.unsqueeze"(%141) <{axis = 1 : i32}> : (tensor<5x160xf16>) -> tensor<5x1x160xf16>
    %143 = "mix.prim.slice"(%142) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x1x160xf16>) -> tensor<5x1x160xf16>
    %144 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %145 = "mix.prim.mul"(%143, %144) : (tensor<5x1x160xf16>, f16) -> tensor<5x1x160xf16>
    %146 = "mix.prim.mul"(%113, %139) : (tensor<5x32x160xf16>, tensor<5x1x160xf16>) -> tensor<5x32x160xf16>
    %147 = "mix.prim.slice"(%113) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x32x160xf16>) -> tensor<5x32x80xf16>
    %148 = "mix.prim.slice"(%113) <{dim = 2 : i64, end = 160 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<5x32x160xf16>) -> tensor<5x32x80xf16>
    %149 = "mix.prim.neg"(%148) : (tensor<5x32x80xf16>) -> tensor<5x32x80xf16>
    %150 = "mix.prim.concat"(%149, %147) <{axis = 2 : i64}> : (tensor<5x32x80xf16>, tensor<5x32x80xf16>) -> tensor<5x32x160xf16>
    %151 = "mix.prim.mul"(%150, %113) : (tensor<5x32x160xf16>, tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %152 = "mix.prim.add"(%146, %151) : (tensor<5x32x160xf16>, tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %153 = "mix.prim.mul"(%116, %139) : (tensor<5x32x160xf16>, tensor<5x1x160xf16>) -> tensor<5x32x160xf16>
    %154 = "mix.prim.slice"(%116) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<5x32x160xf16>) -> tensor<5x32x80xf16>
    %155 = "mix.prim.slice"(%116) <{dim = 2 : i64, end = 160 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<5x32x160xf16>) -> tensor<5x32x80xf16>
    %156 = "mix.prim.neg"(%155) : (tensor<5x32x80xf16>) -> tensor<5x32x80xf16>
    %157 = "mix.prim.concat"(%156, %154) <{axis = 2 : i64}> : (tensor<5x32x80xf16>, tensor<5x32x80xf16>) -> tensor<5x32x160xf16>
    %158 = "mix.prim.mul"(%157, %116) : (tensor<5x32x160xf16>, tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %159 = "mix.prim.add"(%153, %158) : (tensor<5x32x160xf16>, tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %160 = "mix.prim.reshape"(%152) <{shape = [5, 32, 160]}> : (tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %161 = "mix.prim.reshape"(%159) <{shape = [5, 32, 160]}> : (tensor<5x32x160xf16>) -> tensor<5x32x160xf16>
    %162 = "mix.prim.transpose"(%160) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5x32x160xf16>) -> tensor<32x5x160xf16>
    %163 = "mix.prim.transpose"(%161) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5x32x160xf16>) -> tensor<32x5x160xf16>
    %164 = "mix.prim.transpose"(%163) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x5x160xf16>) -> tensor<32x160x5xf16>
    %165 = "mix.prim.batch_matmul"(%162, %164) : (tensor<32x5x160xf16>, tensor<32x160x5xf16>) -> tensor<32x5x5xf16>
    %166 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %167 = "mix.prim.mul"(%165, %166) : (tensor<32x5x5xf16>, f16) -> tensor<32x5x5xf16>
    %168 = "mix.prim.constant"() <{value = -6.550400e+04 : f16}> : () -> f16
    %169 = "mix.prim.masked_fill"(%167, %13, %168) : (tensor<32x5x5xf16>, tensor<1x5x5xi1>, f16) -> tensor<32x5x5xf16>
    %170 = "mix.prim.softmax"(%169) <{axis = -1 : si32}> : (tensor<32x5x5xf16>) -> tensor<32x5x5xf16>
    %171 = "mix.prim.transpose"(%117) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5x32x160xf16>) -> tensor<32x5x160xf16>
    %172 = "mix.prim.batch_matmul"(%170, %171) : (tensor<32x5x5xf16>, tensor<32x5x160xf16>) -> tensor<32x5x160xf16>
    %173 = "mix.prim.permute"(%172) <{dims = [1, 0, 2]}> : (tensor<32x5x160xf16>) -> tensor<5x32x160xf16>
    %174 = "mix.prim.reshape"(%173) <{shape = [5, 5120]}> : (tensor<5x32x160xf16>) -> tensor<5x5120xf16>
    %175 = "mix.module.linear"(%174) <{dtype = f16, has_bias, in_feature = 5120 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.1.self_attention.dense"}> : (tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %176 = "mix.prim.add"(%102, %175) : (tensor<5x5120xf16>, tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %177 = "mix.comp.weight"() <{param_loc = "transformer.h.1.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %178 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %179 = "mix.prim.pow"(%176, %178) : (tensor<5x5120xf16>, tensor<1xf16>) -> tensor<5x5120xf16>
    %180 = "mix.comp.mean"(%179) <{dims = [1 : i32], keepDim = true}> : (tensor<5x5120xf16>) -> tensor<5x1xf16>
    %181 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %182 = "mix.prim.add"(%180, %181) : (tensor<5x1xf16>, f16) -> tensor<5x1xf16>
    %183 = "mix.prim.rsqrt"(%182) : (tensor<5x1xf16>) -> tensor<5x1xf16>
    %184 = "mix.prim.mul"(%176, %183) : (tensor<5x5120xf16>, tensor<5x1xf16>) -> tensor<5x5120xf16>
    %185 = "mix.prim.mul"(%177, %184) : (tensor<5120xf16>, tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %186 = "mix.module.linear"(%185) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.1.mlp.gate_proj"}> : (tensor<5x5120xf16>) -> tensor<5x12288xf16>
    %187 = "mix.comp.silu"(%186) : (tensor<5x12288xf16>) -> tensor<5x12288xf16>
    %188 = "mix.module.linear"(%185) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.1.mlp.up_proj"}> : (tensor<5x5120xf16>) -> tensor<5x12288xf16>
    %189 = "mix.prim.mul"(%187, %188) : (tensor<5x12288xf16>, tensor<5x12288xf16>) -> tensor<5x12288xf16>
    %190 = "mix.module.linear"(%189) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.1.mlp.down_proj"}> : (tensor<5x12288xf16>) -> tensor<5x5120xf16>
    %191 = "mix.prim.add"(%190, %176) : (tensor<5x5120xf16>, tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %192 = "mix.comp.weight"() <{param_loc = "transformer.ln_f.weight"}> : () -> tensor<5120xf16>
    %193 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %194 = "mix.prim.pow"(%191, %193) : (tensor<5x5120xf16>, tensor<1xf16>) -> tensor<5x5120xf16>
    %195 = "mix.comp.mean"(%194) <{dims = [1 : i32], keepDim = true}> : (tensor<5x5120xf16>) -> tensor<5x1xf16>
    %196 = "mix.prim.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %197 = "mix.prim.add"(%195, %196) : (tensor<5x1xf16>, f16) -> tensor<5x1xf16>
    %198 = "mix.prim.rsqrt"(%197) : (tensor<5x1xf16>) -> tensor<5x1xf16>
    %199 = "mix.prim.mul"(%191, %198) : (tensor<5x5120xf16>, tensor<5x1xf16>) -> tensor<5x5120xf16>
    %200 = "mix.prim.mul"(%192, %199) : (tensor<5120xf16>, tensor<5x5120xf16>) -> tensor<5x5120xf16>
    %201 = "mix.module.linear"(%200) <{dtype = f16, in_feature = 5120 : i32, out_feature = 120000 : i32, params_loc = "lm_head"}> : (tensor<5x5120xf16>) -> tensor<5x120000xf16>
    return %201 : tensor<5x120000xf16>
  }
}
