"builtin.module"() ({
  "func.func"() <{function_type = (tensor<1x40xf16>) -> tensor<40x120000xf16>, sym_name = "Telechat", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<1x40xf16>):
    %0 = "mix.module.embedding"(%arg0) <{dtype = f16, embedding_dim = 5120 : i32, num_embeddings = 120000 : i32, params_loc = "transformer.word_embeddings.weight"}> : (tensor<1x40xf16>) -> tensor<1x40x5120xf16>
    %1 = "mix.prim.constant"() <{value = dense<true> : tensor<1x1x5x5xi1>}> : () -> tensor<1x1x5x5xi1>
    %2 = "mix.comp.weight"() <{param_loc = "transformer.h.0.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4 = "mix.prim.pow"(%0, %3) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %5 = "mix.comp.mean"(%4) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %6 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %7 = "mix.prim.add"(%5, %6) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %8 = "mix.prim.rsqrt"(%7) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %9 = "mix.prim.mul"(%0, %8) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %10 = "mix.prim.mul"(%2, %9) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %11 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %12 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %13 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %14 = "mix.comp.weight"() <{param_loc = "transformer.h.0.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %15 = "mix.prim.transpose"(%10) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %16 = "mix.prim.transpose"(%11) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %17 = "mix.prim.reshape"(%15) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %18 = "mix.prim.matmul"(%17, %16) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %19 = "mix.prim.reshape"(%18) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %20 = "mix.prim.reshape"(%19) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %21 = "mix.prim.transpose"(%12) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %22 = "mix.prim.reshape"(%15) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %23 = "mix.prim.matmul"(%22, %21) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %24 = "mix.prim.reshape"(%23) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %25 = "mix.prim.reshape"(%24) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %26 = "mix.prim.slice"(%25) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %27 = "mix.prim.slice"(%25) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %28 = "mix.prim.reshape"(%20) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %29 = "mix.prim.reshape"(%26) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %30 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %31 = "mix.prim.convert"(%30) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %32 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %33 = "mix.prim.div"(%31, %32) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %34 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %35 = "mix.prim.pow"(%34, %33) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %36 = "mix.prim.reciprocal"(%35) : (tensor<80xf16>) -> tensor<80xf16>
    %37 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %38 = "mix.prim.mul"(%37, %36) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %39 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %40 = "mix.prim.unsqueeze"(%39) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %41 = "mix.prim.permute"(%40) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %42 = "mix.prim.unsqueeze"(%38) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %43 = "mix.prim.permute"(%42) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %44 = "mix.prim.mul"(%41, %43) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %45 = "mix.prim.concat"(%44, %44) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %46 = "mix.prim.cos"(%45) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %47 = "mix.prim.slice"(%46) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %48 = "mix.prim.unsqueeze"(%47) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %49 = "mix.prim.slice"(%48) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %50 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %51 = "mix.prim.mul"(%49, %50) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %52 = "mix.prim.sin"(%45) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %53 = "mix.prim.slice"(%52) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %54 = "mix.prim.unsqueeze"(%53) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %55 = "mix.prim.slice"(%54) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %56 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %57 = "mix.prim.mul"(%55, %56) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %58 = "mix.prim.slice"(%51) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %59 = "mix.prim.slice"(%57) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %60 = "mix.prim.mul"(%28, %58) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %61 = "mix.prim.slice"(%28) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %62 = "mix.prim.slice"(%28) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %63 = "mix.prim.neg"(%62) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %64 = "mix.prim.concat"(%63, %61) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %65 = "mix.prim.mul"(%64, %59) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %66 = "mix.prim.add"(%60, %65) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %67 = "mix.prim.mul"(%29, %58) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %68 = "mix.prim.slice"(%29) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %69 = "mix.prim.slice"(%29) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %70 = "mix.prim.neg"(%69) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %71 = "mix.prim.concat"(%70, %68) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %72 = "mix.prim.mul"(%71, %59) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %73 = "mix.prim.add"(%67, %72) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %74 = "mix.prim.reshape"(%66) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %75 = "mix.prim.reshape"(%73) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %76 = "mix.prim.reshape"(%74) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %77 = "mix.prim.reshape"(%75) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %78 = "mix.prim.transpose"(%76) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %79 = "mix.prim.transpose"(%77) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %80 = "mix.prim.transpose"(%79) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %81 = "mix.prim.unsqueeze"(%78) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %82 = "mix.prim.permute"(%81) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %83 = "mix.prim.unsqueeze"(%80) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %84 = "mix.prim.permute"(%83) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %85 = "mix.prim.permute"(%82) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %86 = "mix.prim.reshape"(%85) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %87 = "mix.prim.permute"(%84) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %88 = "mix.prim.reshape"(%87) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %89 = "mix.prim.batch_matmul"(%86, %88) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %90 = "mix.prim.reshape"(%89) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %91 = "mix.prim.permute"(%90) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %92 = "mix.prim.reshape"(%91) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %93 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %94 = "mix.prim.mul"(%92, %93) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %95 = "mix.prim.reshape"(%94) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %96 = "mix.comp.masked_fill"(%95, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %97 = "mix.comp.softmax"(%96) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %98 = "mix.prim.reshape"(%97) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %99 = "mix.prim.reshape"(%27) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %100 = "mix.prim.transpose"(%99) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %101 = "mix.prim.batch_matmul"(%98, %100) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %102 = "mix.prim.reshape"(%101) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %103 = "mix.prim.permute"(%102) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %104 = "mix.prim.reshape"(%103) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %105 = "mix.prim.reshape"(%104) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %106 = "mix.prim.transpose"(%13) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %107 = "mix.prim.matmul"(%105, %106) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %108 = "mix.prim.add"(%107, %14) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %109 = "mix.prim.reshape"(%108) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %110 = "mix.prim.mul"(%0, %109) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %111 = "mix.comp.weight"() <{param_loc = "transformer.h.0.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %112 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %113 = "mix.prim.pow"(%110, %112) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %114 = "mix.comp.mean"(%113) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %115 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %116 = "mix.prim.add"(%114, %115) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %117 = "mix.prim.rsqrt"(%116) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %118 = "mix.prim.mul"(%110, %117) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %119 = "mix.prim.mul"(%111, %118) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %120 = "mix.module.linear"(%119) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.0.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %121 = "mix.comp.silu"(%120) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %122 = "mix.module.linear"(%119) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.0.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %123 = "mix.prim.mul"(%121, %122) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %124 = "mix.module.linear"(%123) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.0.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %125 = "mix.prim.add"(%124, %110) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %126 = "mix.comp.weight"() <{param_loc = "transformer.h.1.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %127 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %128 = "mix.prim.pow"(%125, %127) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %129 = "mix.comp.mean"(%128) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %130 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %131 = "mix.prim.add"(%129, %130) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %132 = "mix.prim.rsqrt"(%131) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %133 = "mix.prim.mul"(%125, %132) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %134 = "mix.prim.mul"(%126, %133) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %135 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %136 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %137 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %138 = "mix.comp.weight"() <{param_loc = "transformer.h.1.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %139 = "mix.prim.transpose"(%134) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %140 = "mix.prim.transpose"(%135) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %141 = "mix.prim.reshape"(%139) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %142 = "mix.prim.matmul"(%141, %140) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %143 = "mix.prim.reshape"(%142) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %144 = "mix.prim.reshape"(%143) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %145 = "mix.prim.transpose"(%136) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %146 = "mix.prim.reshape"(%139) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %147 = "mix.prim.matmul"(%146, %145) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %148 = "mix.prim.reshape"(%147) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %149 = "mix.prim.reshape"(%148) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %150 = "mix.prim.slice"(%149) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %151 = "mix.prim.slice"(%149) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %152 = "mix.prim.reshape"(%144) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %153 = "mix.prim.reshape"(%150) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %154 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %155 = "mix.prim.convert"(%154) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %156 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %157 = "mix.prim.div"(%155, %156) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %158 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %159 = "mix.prim.pow"(%158, %157) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %160 = "mix.prim.reciprocal"(%159) : (tensor<80xf16>) -> tensor<80xf16>
    %161 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %162 = "mix.prim.mul"(%161, %160) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %163 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %164 = "mix.prim.unsqueeze"(%163) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %165 = "mix.prim.permute"(%164) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %166 = "mix.prim.unsqueeze"(%162) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %167 = "mix.prim.permute"(%166) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %168 = "mix.prim.mul"(%165, %167) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %169 = "mix.prim.concat"(%168, %168) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %170 = "mix.prim.cos"(%169) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %171 = "mix.prim.slice"(%170) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %172 = "mix.prim.unsqueeze"(%171) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %173 = "mix.prim.slice"(%172) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %174 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %175 = "mix.prim.mul"(%173, %174) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %176 = "mix.prim.sin"(%169) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %177 = "mix.prim.slice"(%176) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %178 = "mix.prim.unsqueeze"(%177) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %179 = "mix.prim.slice"(%178) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %180 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %181 = "mix.prim.mul"(%179, %180) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %182 = "mix.prim.slice"(%175) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %183 = "mix.prim.slice"(%181) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %184 = "mix.prim.mul"(%152, %182) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %185 = "mix.prim.slice"(%152) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %186 = "mix.prim.slice"(%152) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %187 = "mix.prim.neg"(%186) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %188 = "mix.prim.concat"(%187, %185) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %189 = "mix.prim.mul"(%188, %183) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %190 = "mix.prim.add"(%184, %189) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %191 = "mix.prim.mul"(%153, %182) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %192 = "mix.prim.slice"(%153) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %193 = "mix.prim.slice"(%153) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %194 = "mix.prim.neg"(%193) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %195 = "mix.prim.concat"(%194, %192) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %196 = "mix.prim.mul"(%195, %183) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %197 = "mix.prim.add"(%191, %196) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %198 = "mix.prim.reshape"(%190) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %199 = "mix.prim.reshape"(%197) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %200 = "mix.prim.reshape"(%198) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %201 = "mix.prim.reshape"(%199) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %202 = "mix.prim.transpose"(%200) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %203 = "mix.prim.transpose"(%201) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %204 = "mix.prim.transpose"(%203) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %205 = "mix.prim.unsqueeze"(%202) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %206 = "mix.prim.permute"(%205) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %207 = "mix.prim.unsqueeze"(%204) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %208 = "mix.prim.permute"(%207) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %209 = "mix.prim.permute"(%206) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %210 = "mix.prim.reshape"(%209) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %211 = "mix.prim.permute"(%208) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %212 = "mix.prim.reshape"(%211) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %213 = "mix.prim.batch_matmul"(%210, %212) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %214 = "mix.prim.reshape"(%213) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %215 = "mix.prim.permute"(%214) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %216 = "mix.prim.reshape"(%215) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %217 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %218 = "mix.prim.mul"(%216, %217) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %219 = "mix.prim.reshape"(%218) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %220 = "mix.comp.masked_fill"(%219, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %221 = "mix.comp.softmax"(%220) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %222 = "mix.prim.reshape"(%221) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %223 = "mix.prim.reshape"(%151) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %224 = "mix.prim.transpose"(%223) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %225 = "mix.prim.batch_matmul"(%222, %224) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %226 = "mix.prim.reshape"(%225) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %227 = "mix.prim.permute"(%226) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %228 = "mix.prim.reshape"(%227) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %229 = "mix.prim.reshape"(%228) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %230 = "mix.prim.transpose"(%137) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %231 = "mix.prim.matmul"(%229, %230) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %232 = "mix.prim.add"(%231, %138) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %233 = "mix.prim.reshape"(%232) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %234 = "mix.prim.mul"(%125, %233) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %235 = "mix.comp.weight"() <{param_loc = "transformer.h.1.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %236 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %237 = "mix.prim.pow"(%234, %236) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %238 = "mix.comp.mean"(%237) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %239 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %240 = "mix.prim.add"(%238, %239) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %241 = "mix.prim.rsqrt"(%240) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %242 = "mix.prim.mul"(%234, %241) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %243 = "mix.prim.mul"(%235, %242) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %244 = "mix.module.linear"(%243) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.1.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %245 = "mix.comp.silu"(%244) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %246 = "mix.module.linear"(%243) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.1.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %247 = "mix.prim.mul"(%245, %246) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %248 = "mix.module.linear"(%247) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.1.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %249 = "mix.prim.add"(%248, %234) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %250 = "mix.comp.weight"() <{param_loc = "transformer.h.2.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %251 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %252 = "mix.prim.pow"(%249, %251) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %253 = "mix.comp.mean"(%252) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %254 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %255 = "mix.prim.add"(%253, %254) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %256 = "mix.prim.rsqrt"(%255) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %257 = "mix.prim.mul"(%249, %256) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %258 = "mix.prim.mul"(%250, %257) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %259 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %260 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %261 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %262 = "mix.comp.weight"() <{param_loc = "transformer.h.2.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %263 = "mix.prim.transpose"(%258) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %264 = "mix.prim.transpose"(%259) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %265 = "mix.prim.reshape"(%263) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %266 = "mix.prim.matmul"(%265, %264) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %267 = "mix.prim.reshape"(%266) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %268 = "mix.prim.reshape"(%267) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %269 = "mix.prim.transpose"(%260) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %270 = "mix.prim.reshape"(%263) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %271 = "mix.prim.matmul"(%270, %269) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %272 = "mix.prim.reshape"(%271) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %273 = "mix.prim.reshape"(%272) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %274 = "mix.prim.slice"(%273) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %275 = "mix.prim.slice"(%273) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %276 = "mix.prim.reshape"(%268) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %277 = "mix.prim.reshape"(%274) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %278 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %279 = "mix.prim.convert"(%278) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %280 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %281 = "mix.prim.div"(%279, %280) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %282 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %283 = "mix.prim.pow"(%282, %281) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %284 = "mix.prim.reciprocal"(%283) : (tensor<80xf16>) -> tensor<80xf16>
    %285 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %286 = "mix.prim.mul"(%285, %284) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %287 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %288 = "mix.prim.unsqueeze"(%287) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %289 = "mix.prim.permute"(%288) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %290 = "mix.prim.unsqueeze"(%286) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %291 = "mix.prim.permute"(%290) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %292 = "mix.prim.mul"(%289, %291) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %293 = "mix.prim.concat"(%292, %292) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %294 = "mix.prim.cos"(%293) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %295 = "mix.prim.slice"(%294) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %296 = "mix.prim.unsqueeze"(%295) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %297 = "mix.prim.slice"(%296) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %298 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %299 = "mix.prim.mul"(%297, %298) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %300 = "mix.prim.sin"(%293) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %301 = "mix.prim.slice"(%300) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %302 = "mix.prim.unsqueeze"(%301) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %303 = "mix.prim.slice"(%302) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %304 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %305 = "mix.prim.mul"(%303, %304) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %306 = "mix.prim.slice"(%299) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %307 = "mix.prim.slice"(%305) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %308 = "mix.prim.mul"(%276, %306) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %309 = "mix.prim.slice"(%276) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %310 = "mix.prim.slice"(%276) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %311 = "mix.prim.neg"(%310) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %312 = "mix.prim.concat"(%311, %309) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %313 = "mix.prim.mul"(%312, %307) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %314 = "mix.prim.add"(%308, %313) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %315 = "mix.prim.mul"(%277, %306) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %316 = "mix.prim.slice"(%277) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %317 = "mix.prim.slice"(%277) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %318 = "mix.prim.neg"(%317) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %319 = "mix.prim.concat"(%318, %316) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %320 = "mix.prim.mul"(%319, %307) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %321 = "mix.prim.add"(%315, %320) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %322 = "mix.prim.reshape"(%314) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %323 = "mix.prim.reshape"(%321) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %324 = "mix.prim.reshape"(%322) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %325 = "mix.prim.reshape"(%323) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %326 = "mix.prim.transpose"(%324) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %327 = "mix.prim.transpose"(%325) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %328 = "mix.prim.transpose"(%327) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %329 = "mix.prim.unsqueeze"(%326) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %330 = "mix.prim.permute"(%329) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %331 = "mix.prim.unsqueeze"(%328) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %332 = "mix.prim.permute"(%331) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %333 = "mix.prim.permute"(%330) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %334 = "mix.prim.reshape"(%333) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %335 = "mix.prim.permute"(%332) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %336 = "mix.prim.reshape"(%335) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %337 = "mix.prim.batch_matmul"(%334, %336) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %338 = "mix.prim.reshape"(%337) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %339 = "mix.prim.permute"(%338) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %340 = "mix.prim.reshape"(%339) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %341 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %342 = "mix.prim.mul"(%340, %341) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %343 = "mix.prim.reshape"(%342) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %344 = "mix.comp.masked_fill"(%343, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %345 = "mix.comp.softmax"(%344) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %346 = "mix.prim.reshape"(%345) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %347 = "mix.prim.reshape"(%275) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %348 = "mix.prim.transpose"(%347) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %349 = "mix.prim.batch_matmul"(%346, %348) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %350 = "mix.prim.reshape"(%349) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %351 = "mix.prim.permute"(%350) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %352 = "mix.prim.reshape"(%351) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %353 = "mix.prim.reshape"(%352) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %354 = "mix.prim.transpose"(%261) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %355 = "mix.prim.matmul"(%353, %354) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %356 = "mix.prim.add"(%355, %262) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %357 = "mix.prim.reshape"(%356) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %358 = "mix.prim.mul"(%249, %357) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %359 = "mix.comp.weight"() <{param_loc = "transformer.h.2.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %360 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %361 = "mix.prim.pow"(%358, %360) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %362 = "mix.comp.mean"(%361) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %363 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %364 = "mix.prim.add"(%362, %363) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %365 = "mix.prim.rsqrt"(%364) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %366 = "mix.prim.mul"(%358, %365) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %367 = "mix.prim.mul"(%359, %366) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %368 = "mix.module.linear"(%367) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.2.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %369 = "mix.comp.silu"(%368) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %370 = "mix.module.linear"(%367) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.2.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %371 = "mix.prim.mul"(%369, %370) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %372 = "mix.module.linear"(%371) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.2.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %373 = "mix.prim.add"(%372, %358) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %374 = "mix.comp.weight"() <{param_loc = "transformer.h.3.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %375 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %376 = "mix.prim.pow"(%373, %375) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %377 = "mix.comp.mean"(%376) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %378 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %379 = "mix.prim.add"(%377, %378) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %380 = "mix.prim.rsqrt"(%379) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %381 = "mix.prim.mul"(%373, %380) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %382 = "mix.prim.mul"(%374, %381) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %383 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %384 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %385 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %386 = "mix.comp.weight"() <{param_loc = "transformer.h.3.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %387 = "mix.prim.transpose"(%382) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %388 = "mix.prim.transpose"(%383) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %389 = "mix.prim.reshape"(%387) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %390 = "mix.prim.matmul"(%389, %388) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %391 = "mix.prim.reshape"(%390) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %392 = "mix.prim.reshape"(%391) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %393 = "mix.prim.transpose"(%384) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %394 = "mix.prim.reshape"(%387) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %395 = "mix.prim.matmul"(%394, %393) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %396 = "mix.prim.reshape"(%395) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %397 = "mix.prim.reshape"(%396) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %398 = "mix.prim.slice"(%397) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %399 = "mix.prim.slice"(%397) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %400 = "mix.prim.reshape"(%392) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %401 = "mix.prim.reshape"(%398) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %402 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %403 = "mix.prim.convert"(%402) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %404 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %405 = "mix.prim.div"(%403, %404) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %406 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %407 = "mix.prim.pow"(%406, %405) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %408 = "mix.prim.reciprocal"(%407) : (tensor<80xf16>) -> tensor<80xf16>
    %409 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %410 = "mix.prim.mul"(%409, %408) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %411 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %412 = "mix.prim.unsqueeze"(%411) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %413 = "mix.prim.permute"(%412) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %414 = "mix.prim.unsqueeze"(%410) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %415 = "mix.prim.permute"(%414) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %416 = "mix.prim.mul"(%413, %415) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %417 = "mix.prim.concat"(%416, %416) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %418 = "mix.prim.cos"(%417) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %419 = "mix.prim.slice"(%418) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %420 = "mix.prim.unsqueeze"(%419) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %421 = "mix.prim.slice"(%420) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %422 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %423 = "mix.prim.mul"(%421, %422) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %424 = "mix.prim.sin"(%417) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %425 = "mix.prim.slice"(%424) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %426 = "mix.prim.unsqueeze"(%425) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %427 = "mix.prim.slice"(%426) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %428 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %429 = "mix.prim.mul"(%427, %428) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %430 = "mix.prim.slice"(%423) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %431 = "mix.prim.slice"(%429) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %432 = "mix.prim.mul"(%400, %430) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %433 = "mix.prim.slice"(%400) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %434 = "mix.prim.slice"(%400) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %435 = "mix.prim.neg"(%434) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %436 = "mix.prim.concat"(%435, %433) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %437 = "mix.prim.mul"(%436, %431) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %438 = "mix.prim.add"(%432, %437) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %439 = "mix.prim.mul"(%401, %430) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %440 = "mix.prim.slice"(%401) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %441 = "mix.prim.slice"(%401) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %442 = "mix.prim.neg"(%441) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %443 = "mix.prim.concat"(%442, %440) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %444 = "mix.prim.mul"(%443, %431) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %445 = "mix.prim.add"(%439, %444) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %446 = "mix.prim.reshape"(%438) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %447 = "mix.prim.reshape"(%445) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %448 = "mix.prim.reshape"(%446) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %449 = "mix.prim.reshape"(%447) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %450 = "mix.prim.transpose"(%448) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %451 = "mix.prim.transpose"(%449) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %452 = "mix.prim.transpose"(%451) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %453 = "mix.prim.unsqueeze"(%450) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %454 = "mix.prim.permute"(%453) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %455 = "mix.prim.unsqueeze"(%452) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %456 = "mix.prim.permute"(%455) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %457 = "mix.prim.permute"(%454) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %458 = "mix.prim.reshape"(%457) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %459 = "mix.prim.permute"(%456) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %460 = "mix.prim.reshape"(%459) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %461 = "mix.prim.batch_matmul"(%458, %460) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %462 = "mix.prim.reshape"(%461) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %463 = "mix.prim.permute"(%462) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %464 = "mix.prim.reshape"(%463) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %465 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %466 = "mix.prim.mul"(%464, %465) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %467 = "mix.prim.reshape"(%466) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %468 = "mix.comp.masked_fill"(%467, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %469 = "mix.comp.softmax"(%468) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %470 = "mix.prim.reshape"(%469) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %471 = "mix.prim.reshape"(%399) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %472 = "mix.prim.transpose"(%471) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %473 = "mix.prim.batch_matmul"(%470, %472) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %474 = "mix.prim.reshape"(%473) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %475 = "mix.prim.permute"(%474) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %476 = "mix.prim.reshape"(%475) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %477 = "mix.prim.reshape"(%476) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %478 = "mix.prim.transpose"(%385) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %479 = "mix.prim.matmul"(%477, %478) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %480 = "mix.prim.add"(%479, %386) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %481 = "mix.prim.reshape"(%480) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %482 = "mix.prim.mul"(%373, %481) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %483 = "mix.comp.weight"() <{param_loc = "transformer.h.3.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %484 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %485 = "mix.prim.pow"(%482, %484) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %486 = "mix.comp.mean"(%485) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %487 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %488 = "mix.prim.add"(%486, %487) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %489 = "mix.prim.rsqrt"(%488) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %490 = "mix.prim.mul"(%482, %489) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %491 = "mix.prim.mul"(%483, %490) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %492 = "mix.module.linear"(%491) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.3.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %493 = "mix.comp.silu"(%492) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %494 = "mix.module.linear"(%491) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.3.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %495 = "mix.prim.mul"(%493, %494) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %496 = "mix.module.linear"(%495) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.3.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %497 = "mix.prim.add"(%496, %482) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %498 = "mix.comp.weight"() <{param_loc = "transformer.h.4.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %499 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %500 = "mix.prim.pow"(%497, %499) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %501 = "mix.comp.mean"(%500) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %502 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %503 = "mix.prim.add"(%501, %502) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %504 = "mix.prim.rsqrt"(%503) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %505 = "mix.prim.mul"(%497, %504) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %506 = "mix.prim.mul"(%498, %505) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %507 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %508 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %509 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %510 = "mix.comp.weight"() <{param_loc = "transformer.h.4.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %511 = "mix.prim.transpose"(%506) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %512 = "mix.prim.transpose"(%507) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %513 = "mix.prim.reshape"(%511) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %514 = "mix.prim.matmul"(%513, %512) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %515 = "mix.prim.reshape"(%514) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %516 = "mix.prim.reshape"(%515) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %517 = "mix.prim.transpose"(%508) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %518 = "mix.prim.reshape"(%511) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %519 = "mix.prim.matmul"(%518, %517) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %520 = "mix.prim.reshape"(%519) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %521 = "mix.prim.reshape"(%520) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %522 = "mix.prim.slice"(%521) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %523 = "mix.prim.slice"(%521) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %524 = "mix.prim.reshape"(%516) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %525 = "mix.prim.reshape"(%522) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %526 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %527 = "mix.prim.convert"(%526) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %528 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %529 = "mix.prim.div"(%527, %528) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %530 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %531 = "mix.prim.pow"(%530, %529) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %532 = "mix.prim.reciprocal"(%531) : (tensor<80xf16>) -> tensor<80xf16>
    %533 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %534 = "mix.prim.mul"(%533, %532) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %535 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %536 = "mix.prim.unsqueeze"(%535) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %537 = "mix.prim.permute"(%536) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %538 = "mix.prim.unsqueeze"(%534) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %539 = "mix.prim.permute"(%538) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %540 = "mix.prim.mul"(%537, %539) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %541 = "mix.prim.concat"(%540, %540) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %542 = "mix.prim.cos"(%541) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %543 = "mix.prim.slice"(%542) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %544 = "mix.prim.unsqueeze"(%543) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %545 = "mix.prim.slice"(%544) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %546 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %547 = "mix.prim.mul"(%545, %546) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %548 = "mix.prim.sin"(%541) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %549 = "mix.prim.slice"(%548) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %550 = "mix.prim.unsqueeze"(%549) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %551 = "mix.prim.slice"(%550) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %552 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %553 = "mix.prim.mul"(%551, %552) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %554 = "mix.prim.slice"(%547) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %555 = "mix.prim.slice"(%553) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %556 = "mix.prim.mul"(%524, %554) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %557 = "mix.prim.slice"(%524) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %558 = "mix.prim.slice"(%524) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %559 = "mix.prim.neg"(%558) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %560 = "mix.prim.concat"(%559, %557) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %561 = "mix.prim.mul"(%560, %555) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %562 = "mix.prim.add"(%556, %561) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %563 = "mix.prim.mul"(%525, %554) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %564 = "mix.prim.slice"(%525) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %565 = "mix.prim.slice"(%525) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %566 = "mix.prim.neg"(%565) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %567 = "mix.prim.concat"(%566, %564) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %568 = "mix.prim.mul"(%567, %555) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %569 = "mix.prim.add"(%563, %568) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %570 = "mix.prim.reshape"(%562) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %571 = "mix.prim.reshape"(%569) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %572 = "mix.prim.reshape"(%570) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %573 = "mix.prim.reshape"(%571) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %574 = "mix.prim.transpose"(%572) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %575 = "mix.prim.transpose"(%573) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %576 = "mix.prim.transpose"(%575) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %577 = "mix.prim.unsqueeze"(%574) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %578 = "mix.prim.permute"(%577) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %579 = "mix.prim.unsqueeze"(%576) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %580 = "mix.prim.permute"(%579) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %581 = "mix.prim.permute"(%578) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %582 = "mix.prim.reshape"(%581) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %583 = "mix.prim.permute"(%580) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %584 = "mix.prim.reshape"(%583) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %585 = "mix.prim.batch_matmul"(%582, %584) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %586 = "mix.prim.reshape"(%585) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %587 = "mix.prim.permute"(%586) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %588 = "mix.prim.reshape"(%587) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %589 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %590 = "mix.prim.mul"(%588, %589) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %591 = "mix.prim.reshape"(%590) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %592 = "mix.comp.masked_fill"(%591, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %593 = "mix.comp.softmax"(%592) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %594 = "mix.prim.reshape"(%593) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %595 = "mix.prim.reshape"(%523) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %596 = "mix.prim.transpose"(%595) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %597 = "mix.prim.batch_matmul"(%594, %596) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %598 = "mix.prim.reshape"(%597) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %599 = "mix.prim.permute"(%598) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %600 = "mix.prim.reshape"(%599) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %601 = "mix.prim.reshape"(%600) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %602 = "mix.prim.transpose"(%509) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %603 = "mix.prim.matmul"(%601, %602) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %604 = "mix.prim.add"(%603, %510) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %605 = "mix.prim.reshape"(%604) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %606 = "mix.prim.mul"(%497, %605) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %607 = "mix.comp.weight"() <{param_loc = "transformer.h.4.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %608 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %609 = "mix.prim.pow"(%606, %608) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %610 = "mix.comp.mean"(%609) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %611 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %612 = "mix.prim.add"(%610, %611) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %613 = "mix.prim.rsqrt"(%612) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %614 = "mix.prim.mul"(%606, %613) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %615 = "mix.prim.mul"(%607, %614) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %616 = "mix.module.linear"(%615) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.4.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %617 = "mix.comp.silu"(%616) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %618 = "mix.module.linear"(%615) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.4.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %619 = "mix.prim.mul"(%617, %618) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %620 = "mix.module.linear"(%619) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.4.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %621 = "mix.prim.add"(%620, %606) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %622 = "mix.comp.weight"() <{param_loc = "transformer.h.5.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %623 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %624 = "mix.prim.pow"(%621, %623) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %625 = "mix.comp.mean"(%624) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %626 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %627 = "mix.prim.add"(%625, %626) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %628 = "mix.prim.rsqrt"(%627) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %629 = "mix.prim.mul"(%621, %628) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %630 = "mix.prim.mul"(%622, %629) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %631 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %632 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %633 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %634 = "mix.comp.weight"() <{param_loc = "transformer.h.5.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %635 = "mix.prim.transpose"(%630) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %636 = "mix.prim.transpose"(%631) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %637 = "mix.prim.reshape"(%635) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %638 = "mix.prim.matmul"(%637, %636) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %639 = "mix.prim.reshape"(%638) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %640 = "mix.prim.reshape"(%639) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %641 = "mix.prim.transpose"(%632) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %642 = "mix.prim.reshape"(%635) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %643 = "mix.prim.matmul"(%642, %641) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %644 = "mix.prim.reshape"(%643) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %645 = "mix.prim.reshape"(%644) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %646 = "mix.prim.slice"(%645) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %647 = "mix.prim.slice"(%645) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %648 = "mix.prim.reshape"(%640) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %649 = "mix.prim.reshape"(%646) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %650 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %651 = "mix.prim.convert"(%650) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %652 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %653 = "mix.prim.div"(%651, %652) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %654 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %655 = "mix.prim.pow"(%654, %653) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %656 = "mix.prim.reciprocal"(%655) : (tensor<80xf16>) -> tensor<80xf16>
    %657 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %658 = "mix.prim.mul"(%657, %656) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %659 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %660 = "mix.prim.unsqueeze"(%659) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %661 = "mix.prim.permute"(%660) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %662 = "mix.prim.unsqueeze"(%658) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %663 = "mix.prim.permute"(%662) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %664 = "mix.prim.mul"(%661, %663) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %665 = "mix.prim.concat"(%664, %664) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %666 = "mix.prim.cos"(%665) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %667 = "mix.prim.slice"(%666) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %668 = "mix.prim.unsqueeze"(%667) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %669 = "mix.prim.slice"(%668) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %670 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %671 = "mix.prim.mul"(%669, %670) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %672 = "mix.prim.sin"(%665) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %673 = "mix.prim.slice"(%672) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %674 = "mix.prim.unsqueeze"(%673) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %675 = "mix.prim.slice"(%674) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %676 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %677 = "mix.prim.mul"(%675, %676) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %678 = "mix.prim.slice"(%671) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %679 = "mix.prim.slice"(%677) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %680 = "mix.prim.mul"(%648, %678) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %681 = "mix.prim.slice"(%648) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %682 = "mix.prim.slice"(%648) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %683 = "mix.prim.neg"(%682) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %684 = "mix.prim.concat"(%683, %681) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %685 = "mix.prim.mul"(%684, %679) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %686 = "mix.prim.add"(%680, %685) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %687 = "mix.prim.mul"(%649, %678) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %688 = "mix.prim.slice"(%649) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %689 = "mix.prim.slice"(%649) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %690 = "mix.prim.neg"(%689) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %691 = "mix.prim.concat"(%690, %688) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %692 = "mix.prim.mul"(%691, %679) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %693 = "mix.prim.add"(%687, %692) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %694 = "mix.prim.reshape"(%686) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %695 = "mix.prim.reshape"(%693) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %696 = "mix.prim.reshape"(%694) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %697 = "mix.prim.reshape"(%695) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %698 = "mix.prim.transpose"(%696) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %699 = "mix.prim.transpose"(%697) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %700 = "mix.prim.transpose"(%699) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %701 = "mix.prim.unsqueeze"(%698) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %702 = "mix.prim.permute"(%701) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %703 = "mix.prim.unsqueeze"(%700) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %704 = "mix.prim.permute"(%703) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %705 = "mix.prim.permute"(%702) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %706 = "mix.prim.reshape"(%705) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %707 = "mix.prim.permute"(%704) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %708 = "mix.prim.reshape"(%707) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %709 = "mix.prim.batch_matmul"(%706, %708) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %710 = "mix.prim.reshape"(%709) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %711 = "mix.prim.permute"(%710) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %712 = "mix.prim.reshape"(%711) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %713 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %714 = "mix.prim.mul"(%712, %713) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %715 = "mix.prim.reshape"(%714) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %716 = "mix.comp.masked_fill"(%715, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %717 = "mix.comp.softmax"(%716) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %718 = "mix.prim.reshape"(%717) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %719 = "mix.prim.reshape"(%647) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %720 = "mix.prim.transpose"(%719) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %721 = "mix.prim.batch_matmul"(%718, %720) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %722 = "mix.prim.reshape"(%721) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %723 = "mix.prim.permute"(%722) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %724 = "mix.prim.reshape"(%723) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %725 = "mix.prim.reshape"(%724) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %726 = "mix.prim.transpose"(%633) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %727 = "mix.prim.matmul"(%725, %726) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %728 = "mix.prim.add"(%727, %634) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %729 = "mix.prim.reshape"(%728) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %730 = "mix.prim.mul"(%621, %729) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %731 = "mix.comp.weight"() <{param_loc = "transformer.h.5.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %732 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %733 = "mix.prim.pow"(%730, %732) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %734 = "mix.comp.mean"(%733) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %735 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %736 = "mix.prim.add"(%734, %735) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %737 = "mix.prim.rsqrt"(%736) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %738 = "mix.prim.mul"(%730, %737) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %739 = "mix.prim.mul"(%731, %738) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %740 = "mix.module.linear"(%739) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.5.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %741 = "mix.comp.silu"(%740) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %742 = "mix.module.linear"(%739) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.5.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %743 = "mix.prim.mul"(%741, %742) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %744 = "mix.module.linear"(%743) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.5.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %745 = "mix.prim.add"(%744, %730) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %746 = "mix.comp.weight"() <{param_loc = "transformer.h.6.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %747 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %748 = "mix.prim.pow"(%745, %747) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %749 = "mix.comp.mean"(%748) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %750 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %751 = "mix.prim.add"(%749, %750) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %752 = "mix.prim.rsqrt"(%751) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %753 = "mix.prim.mul"(%745, %752) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %754 = "mix.prim.mul"(%746, %753) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %755 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %756 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %757 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %758 = "mix.comp.weight"() <{param_loc = "transformer.h.6.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %759 = "mix.prim.transpose"(%754) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %760 = "mix.prim.transpose"(%755) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %761 = "mix.prim.reshape"(%759) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %762 = "mix.prim.matmul"(%761, %760) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %763 = "mix.prim.reshape"(%762) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %764 = "mix.prim.reshape"(%763) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %765 = "mix.prim.transpose"(%756) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %766 = "mix.prim.reshape"(%759) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %767 = "mix.prim.matmul"(%766, %765) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %768 = "mix.prim.reshape"(%767) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %769 = "mix.prim.reshape"(%768) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %770 = "mix.prim.slice"(%769) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %771 = "mix.prim.slice"(%769) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %772 = "mix.prim.reshape"(%764) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %773 = "mix.prim.reshape"(%770) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %774 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %775 = "mix.prim.convert"(%774) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %776 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %777 = "mix.prim.div"(%775, %776) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %778 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %779 = "mix.prim.pow"(%778, %777) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %780 = "mix.prim.reciprocal"(%779) : (tensor<80xf16>) -> tensor<80xf16>
    %781 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %782 = "mix.prim.mul"(%781, %780) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %783 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %784 = "mix.prim.unsqueeze"(%783) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %785 = "mix.prim.permute"(%784) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %786 = "mix.prim.unsqueeze"(%782) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %787 = "mix.prim.permute"(%786) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %788 = "mix.prim.mul"(%785, %787) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %789 = "mix.prim.concat"(%788, %788) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %790 = "mix.prim.cos"(%789) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %791 = "mix.prim.slice"(%790) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %792 = "mix.prim.unsqueeze"(%791) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %793 = "mix.prim.slice"(%792) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %794 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %795 = "mix.prim.mul"(%793, %794) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %796 = "mix.prim.sin"(%789) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %797 = "mix.prim.slice"(%796) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %798 = "mix.prim.unsqueeze"(%797) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %799 = "mix.prim.slice"(%798) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %800 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %801 = "mix.prim.mul"(%799, %800) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %802 = "mix.prim.slice"(%795) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %803 = "mix.prim.slice"(%801) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %804 = "mix.prim.mul"(%772, %802) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %805 = "mix.prim.slice"(%772) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %806 = "mix.prim.slice"(%772) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %807 = "mix.prim.neg"(%806) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %808 = "mix.prim.concat"(%807, %805) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %809 = "mix.prim.mul"(%808, %803) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %810 = "mix.prim.add"(%804, %809) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %811 = "mix.prim.mul"(%773, %802) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %812 = "mix.prim.slice"(%773) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %813 = "mix.prim.slice"(%773) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %814 = "mix.prim.neg"(%813) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %815 = "mix.prim.concat"(%814, %812) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %816 = "mix.prim.mul"(%815, %803) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %817 = "mix.prim.add"(%811, %816) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %818 = "mix.prim.reshape"(%810) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %819 = "mix.prim.reshape"(%817) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %820 = "mix.prim.reshape"(%818) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %821 = "mix.prim.reshape"(%819) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %822 = "mix.prim.transpose"(%820) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %823 = "mix.prim.transpose"(%821) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %824 = "mix.prim.transpose"(%823) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %825 = "mix.prim.unsqueeze"(%822) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %826 = "mix.prim.permute"(%825) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %827 = "mix.prim.unsqueeze"(%824) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %828 = "mix.prim.permute"(%827) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %829 = "mix.prim.permute"(%826) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %830 = "mix.prim.reshape"(%829) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %831 = "mix.prim.permute"(%828) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %832 = "mix.prim.reshape"(%831) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %833 = "mix.prim.batch_matmul"(%830, %832) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %834 = "mix.prim.reshape"(%833) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %835 = "mix.prim.permute"(%834) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %836 = "mix.prim.reshape"(%835) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %837 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %838 = "mix.prim.mul"(%836, %837) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %839 = "mix.prim.reshape"(%838) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %840 = "mix.comp.masked_fill"(%839, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %841 = "mix.comp.softmax"(%840) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %842 = "mix.prim.reshape"(%841) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %843 = "mix.prim.reshape"(%771) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %844 = "mix.prim.transpose"(%843) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %845 = "mix.prim.batch_matmul"(%842, %844) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %846 = "mix.prim.reshape"(%845) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %847 = "mix.prim.permute"(%846) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %848 = "mix.prim.reshape"(%847) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %849 = "mix.prim.reshape"(%848) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %850 = "mix.prim.transpose"(%757) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %851 = "mix.prim.matmul"(%849, %850) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %852 = "mix.prim.add"(%851, %758) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %853 = "mix.prim.reshape"(%852) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %854 = "mix.prim.mul"(%745, %853) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %855 = "mix.comp.weight"() <{param_loc = "transformer.h.6.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %856 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %857 = "mix.prim.pow"(%854, %856) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %858 = "mix.comp.mean"(%857) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %859 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %860 = "mix.prim.add"(%858, %859) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %861 = "mix.prim.rsqrt"(%860) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %862 = "mix.prim.mul"(%854, %861) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %863 = "mix.prim.mul"(%855, %862) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %864 = "mix.module.linear"(%863) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.6.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %865 = "mix.comp.silu"(%864) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %866 = "mix.module.linear"(%863) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.6.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %867 = "mix.prim.mul"(%865, %866) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %868 = "mix.module.linear"(%867) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.6.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %869 = "mix.prim.add"(%868, %854) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %870 = "mix.comp.weight"() <{param_loc = "transformer.h.7.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %871 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %872 = "mix.prim.pow"(%869, %871) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %873 = "mix.comp.mean"(%872) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %874 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %875 = "mix.prim.add"(%873, %874) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %876 = "mix.prim.rsqrt"(%875) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %877 = "mix.prim.mul"(%869, %876) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %878 = "mix.prim.mul"(%870, %877) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %879 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %880 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %881 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %882 = "mix.comp.weight"() <{param_loc = "transformer.h.7.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %883 = "mix.prim.transpose"(%878) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %884 = "mix.prim.transpose"(%879) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %885 = "mix.prim.reshape"(%883) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %886 = "mix.prim.matmul"(%885, %884) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %887 = "mix.prim.reshape"(%886) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %888 = "mix.prim.reshape"(%887) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %889 = "mix.prim.transpose"(%880) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %890 = "mix.prim.reshape"(%883) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %891 = "mix.prim.matmul"(%890, %889) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %892 = "mix.prim.reshape"(%891) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %893 = "mix.prim.reshape"(%892) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %894 = "mix.prim.slice"(%893) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %895 = "mix.prim.slice"(%893) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %896 = "mix.prim.reshape"(%888) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %897 = "mix.prim.reshape"(%894) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %898 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %899 = "mix.prim.convert"(%898) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %900 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %901 = "mix.prim.div"(%899, %900) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %902 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %903 = "mix.prim.pow"(%902, %901) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %904 = "mix.prim.reciprocal"(%903) : (tensor<80xf16>) -> tensor<80xf16>
    %905 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %906 = "mix.prim.mul"(%905, %904) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %907 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %908 = "mix.prim.unsqueeze"(%907) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %909 = "mix.prim.permute"(%908) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %910 = "mix.prim.unsqueeze"(%906) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %911 = "mix.prim.permute"(%910) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %912 = "mix.prim.mul"(%909, %911) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %913 = "mix.prim.concat"(%912, %912) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %914 = "mix.prim.cos"(%913) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %915 = "mix.prim.slice"(%914) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %916 = "mix.prim.unsqueeze"(%915) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %917 = "mix.prim.slice"(%916) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %918 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %919 = "mix.prim.mul"(%917, %918) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %920 = "mix.prim.sin"(%913) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %921 = "mix.prim.slice"(%920) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %922 = "mix.prim.unsqueeze"(%921) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %923 = "mix.prim.slice"(%922) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %924 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %925 = "mix.prim.mul"(%923, %924) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %926 = "mix.prim.slice"(%919) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %927 = "mix.prim.slice"(%925) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %928 = "mix.prim.mul"(%896, %926) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %929 = "mix.prim.slice"(%896) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %930 = "mix.prim.slice"(%896) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %931 = "mix.prim.neg"(%930) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %932 = "mix.prim.concat"(%931, %929) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %933 = "mix.prim.mul"(%932, %927) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %934 = "mix.prim.add"(%928, %933) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %935 = "mix.prim.mul"(%897, %926) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %936 = "mix.prim.slice"(%897) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %937 = "mix.prim.slice"(%897) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %938 = "mix.prim.neg"(%937) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %939 = "mix.prim.concat"(%938, %936) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %940 = "mix.prim.mul"(%939, %927) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %941 = "mix.prim.add"(%935, %940) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %942 = "mix.prim.reshape"(%934) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %943 = "mix.prim.reshape"(%941) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %944 = "mix.prim.reshape"(%942) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %945 = "mix.prim.reshape"(%943) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %946 = "mix.prim.transpose"(%944) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %947 = "mix.prim.transpose"(%945) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %948 = "mix.prim.transpose"(%947) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %949 = "mix.prim.unsqueeze"(%946) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %950 = "mix.prim.permute"(%949) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %951 = "mix.prim.unsqueeze"(%948) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %952 = "mix.prim.permute"(%951) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %953 = "mix.prim.permute"(%950) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %954 = "mix.prim.reshape"(%953) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %955 = "mix.prim.permute"(%952) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %956 = "mix.prim.reshape"(%955) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %957 = "mix.prim.batch_matmul"(%954, %956) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %958 = "mix.prim.reshape"(%957) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %959 = "mix.prim.permute"(%958) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %960 = "mix.prim.reshape"(%959) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %961 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %962 = "mix.prim.mul"(%960, %961) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %963 = "mix.prim.reshape"(%962) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %964 = "mix.comp.masked_fill"(%963, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %965 = "mix.comp.softmax"(%964) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %966 = "mix.prim.reshape"(%965) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %967 = "mix.prim.reshape"(%895) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %968 = "mix.prim.transpose"(%967) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %969 = "mix.prim.batch_matmul"(%966, %968) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %970 = "mix.prim.reshape"(%969) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %971 = "mix.prim.permute"(%970) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %972 = "mix.prim.reshape"(%971) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %973 = "mix.prim.reshape"(%972) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %974 = "mix.prim.transpose"(%881) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %975 = "mix.prim.matmul"(%973, %974) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %976 = "mix.prim.add"(%975, %882) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %977 = "mix.prim.reshape"(%976) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %978 = "mix.prim.mul"(%869, %977) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %979 = "mix.comp.weight"() <{param_loc = "transformer.h.7.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %980 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %981 = "mix.prim.pow"(%978, %980) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %982 = "mix.comp.mean"(%981) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %983 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %984 = "mix.prim.add"(%982, %983) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %985 = "mix.prim.rsqrt"(%984) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %986 = "mix.prim.mul"(%978, %985) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %987 = "mix.prim.mul"(%979, %986) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %988 = "mix.module.linear"(%987) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.7.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %989 = "mix.comp.silu"(%988) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %990 = "mix.module.linear"(%987) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.7.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %991 = "mix.prim.mul"(%989, %990) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %992 = "mix.module.linear"(%991) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.7.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %993 = "mix.prim.add"(%992, %978) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %994 = "mix.comp.weight"() <{param_loc = "transformer.h.8.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %995 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %996 = "mix.prim.pow"(%993, %995) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %997 = "mix.comp.mean"(%996) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %998 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %999 = "mix.prim.add"(%997, %998) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1000 = "mix.prim.rsqrt"(%999) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1001 = "mix.prim.mul"(%993, %1000) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1002 = "mix.prim.mul"(%994, %1001) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1003 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1004 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1005 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1006 = "mix.comp.weight"() <{param_loc = "transformer.h.8.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1007 = "mix.prim.transpose"(%1002) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1008 = "mix.prim.transpose"(%1003) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1009 = "mix.prim.reshape"(%1007) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1010 = "mix.prim.matmul"(%1009, %1008) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1011 = "mix.prim.reshape"(%1010) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1012 = "mix.prim.reshape"(%1011) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1013 = "mix.prim.transpose"(%1004) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1014 = "mix.prim.reshape"(%1007) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1015 = "mix.prim.matmul"(%1014, %1013) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1016 = "mix.prim.reshape"(%1015) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1017 = "mix.prim.reshape"(%1016) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1018 = "mix.prim.slice"(%1017) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1019 = "mix.prim.slice"(%1017) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1020 = "mix.prim.reshape"(%1012) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1021 = "mix.prim.reshape"(%1018) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1022 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1023 = "mix.prim.convert"(%1022) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1024 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1025 = "mix.prim.div"(%1023, %1024) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1026 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1027 = "mix.prim.pow"(%1026, %1025) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1028 = "mix.prim.reciprocal"(%1027) : (tensor<80xf16>) -> tensor<80xf16>
    %1029 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1030 = "mix.prim.mul"(%1029, %1028) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1031 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1032 = "mix.prim.unsqueeze"(%1031) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1033 = "mix.prim.permute"(%1032) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1034 = "mix.prim.unsqueeze"(%1030) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1035 = "mix.prim.permute"(%1034) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1036 = "mix.prim.mul"(%1033, %1035) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1037 = "mix.prim.concat"(%1036, %1036) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1038 = "mix.prim.cos"(%1037) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1039 = "mix.prim.slice"(%1038) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1040 = "mix.prim.unsqueeze"(%1039) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1041 = "mix.prim.slice"(%1040) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1042 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1043 = "mix.prim.mul"(%1041, %1042) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1044 = "mix.prim.sin"(%1037) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1045 = "mix.prim.slice"(%1044) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1046 = "mix.prim.unsqueeze"(%1045) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1047 = "mix.prim.slice"(%1046) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1048 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1049 = "mix.prim.mul"(%1047, %1048) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1050 = "mix.prim.slice"(%1043) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1051 = "mix.prim.slice"(%1049) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1052 = "mix.prim.mul"(%1020, %1050) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1053 = "mix.prim.slice"(%1020) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1054 = "mix.prim.slice"(%1020) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1055 = "mix.prim.neg"(%1054) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1056 = "mix.prim.concat"(%1055, %1053) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1057 = "mix.prim.mul"(%1056, %1051) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1058 = "mix.prim.add"(%1052, %1057) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1059 = "mix.prim.mul"(%1021, %1050) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1060 = "mix.prim.slice"(%1021) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1061 = "mix.prim.slice"(%1021) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1062 = "mix.prim.neg"(%1061) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1063 = "mix.prim.concat"(%1062, %1060) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1064 = "mix.prim.mul"(%1063, %1051) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1065 = "mix.prim.add"(%1059, %1064) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1066 = "mix.prim.reshape"(%1058) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1067 = "mix.prim.reshape"(%1065) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1068 = "mix.prim.reshape"(%1066) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1069 = "mix.prim.reshape"(%1067) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1070 = "mix.prim.transpose"(%1068) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1071 = "mix.prim.transpose"(%1069) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1072 = "mix.prim.transpose"(%1071) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1073 = "mix.prim.unsqueeze"(%1070) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1074 = "mix.prim.permute"(%1073) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1075 = "mix.prim.unsqueeze"(%1072) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1076 = "mix.prim.permute"(%1075) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1077 = "mix.prim.permute"(%1074) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1078 = "mix.prim.reshape"(%1077) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1079 = "mix.prim.permute"(%1076) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1080 = "mix.prim.reshape"(%1079) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1081 = "mix.prim.batch_matmul"(%1078, %1080) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1082 = "mix.prim.reshape"(%1081) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1083 = "mix.prim.permute"(%1082) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1084 = "mix.prim.reshape"(%1083) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1085 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1086 = "mix.prim.mul"(%1084, %1085) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1087 = "mix.prim.reshape"(%1086) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1088 = "mix.comp.masked_fill"(%1087, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %1089 = "mix.comp.softmax"(%1088) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1090 = "mix.prim.reshape"(%1089) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1091 = "mix.prim.reshape"(%1019) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1092 = "mix.prim.transpose"(%1091) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1093 = "mix.prim.batch_matmul"(%1090, %1092) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1094 = "mix.prim.reshape"(%1093) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1095 = "mix.prim.permute"(%1094) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1096 = "mix.prim.reshape"(%1095) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1097 = "mix.prim.reshape"(%1096) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1098 = "mix.prim.transpose"(%1005) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1099 = "mix.prim.matmul"(%1097, %1098) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1100 = "mix.prim.add"(%1099, %1006) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1101 = "mix.prim.reshape"(%1100) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1102 = "mix.prim.mul"(%993, %1101) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1103 = "mix.comp.weight"() <{param_loc = "transformer.h.8.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1104 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1105 = "mix.prim.pow"(%1102, %1104) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1106 = "mix.comp.mean"(%1105) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1107 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1108 = "mix.prim.add"(%1106, %1107) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1109 = "mix.prim.rsqrt"(%1108) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1110 = "mix.prim.mul"(%1102, %1109) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1111 = "mix.prim.mul"(%1103, %1110) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1112 = "mix.module.linear"(%1111) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.8.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1113 = "mix.comp.silu"(%1112) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1114 = "mix.module.linear"(%1111) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.8.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1115 = "mix.prim.mul"(%1113, %1114) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1116 = "mix.module.linear"(%1115) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.8.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1117 = "mix.prim.add"(%1116, %1102) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1118 = "mix.comp.weight"() <{param_loc = "transformer.h.9.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1119 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1120 = "mix.prim.pow"(%1117, %1119) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1121 = "mix.comp.mean"(%1120) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1122 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1123 = "mix.prim.add"(%1121, %1122) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1124 = "mix.prim.rsqrt"(%1123) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1125 = "mix.prim.mul"(%1117, %1124) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1126 = "mix.prim.mul"(%1118, %1125) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1127 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1128 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1129 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1130 = "mix.comp.weight"() <{param_loc = "transformer.h.9.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1131 = "mix.prim.transpose"(%1126) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1132 = "mix.prim.transpose"(%1127) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1133 = "mix.prim.reshape"(%1131) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1134 = "mix.prim.matmul"(%1133, %1132) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1135 = "mix.prim.reshape"(%1134) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1136 = "mix.prim.reshape"(%1135) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1137 = "mix.prim.transpose"(%1128) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1138 = "mix.prim.reshape"(%1131) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1139 = "mix.prim.matmul"(%1138, %1137) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1140 = "mix.prim.reshape"(%1139) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1141 = "mix.prim.reshape"(%1140) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1142 = "mix.prim.slice"(%1141) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1143 = "mix.prim.slice"(%1141) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1144 = "mix.prim.reshape"(%1136) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1145 = "mix.prim.reshape"(%1142) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1146 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1147 = "mix.prim.convert"(%1146) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1148 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1149 = "mix.prim.div"(%1147, %1148) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1150 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1151 = "mix.prim.pow"(%1150, %1149) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1152 = "mix.prim.reciprocal"(%1151) : (tensor<80xf16>) -> tensor<80xf16>
    %1153 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1154 = "mix.prim.mul"(%1153, %1152) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1155 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1156 = "mix.prim.unsqueeze"(%1155) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1157 = "mix.prim.permute"(%1156) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1158 = "mix.prim.unsqueeze"(%1154) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1159 = "mix.prim.permute"(%1158) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1160 = "mix.prim.mul"(%1157, %1159) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1161 = "mix.prim.concat"(%1160, %1160) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1162 = "mix.prim.cos"(%1161) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1163 = "mix.prim.slice"(%1162) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1164 = "mix.prim.unsqueeze"(%1163) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1165 = "mix.prim.slice"(%1164) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1166 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1167 = "mix.prim.mul"(%1165, %1166) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1168 = "mix.prim.sin"(%1161) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1169 = "mix.prim.slice"(%1168) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1170 = "mix.prim.unsqueeze"(%1169) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1171 = "mix.prim.slice"(%1170) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1172 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1173 = "mix.prim.mul"(%1171, %1172) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1174 = "mix.prim.slice"(%1167) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1175 = "mix.prim.slice"(%1173) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1176 = "mix.prim.mul"(%1144, %1174) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1177 = "mix.prim.slice"(%1144) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1178 = "mix.prim.slice"(%1144) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1179 = "mix.prim.neg"(%1178) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1180 = "mix.prim.concat"(%1179, %1177) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1181 = "mix.prim.mul"(%1180, %1175) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1182 = "mix.prim.add"(%1176, %1181) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1183 = "mix.prim.mul"(%1145, %1174) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1184 = "mix.prim.slice"(%1145) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1185 = "mix.prim.slice"(%1145) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1186 = "mix.prim.neg"(%1185) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1187 = "mix.prim.concat"(%1186, %1184) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1188 = "mix.prim.mul"(%1187, %1175) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1189 = "mix.prim.add"(%1183, %1188) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1190 = "mix.prim.reshape"(%1182) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1191 = "mix.prim.reshape"(%1189) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1192 = "mix.prim.reshape"(%1190) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1193 = "mix.prim.reshape"(%1191) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1194 = "mix.prim.transpose"(%1192) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1195 = "mix.prim.transpose"(%1193) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1196 = "mix.prim.transpose"(%1195) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1197 = "mix.prim.unsqueeze"(%1194) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1198 = "mix.prim.permute"(%1197) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1199 = "mix.prim.unsqueeze"(%1196) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1200 = "mix.prim.permute"(%1199) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1201 = "mix.prim.permute"(%1198) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1202 = "mix.prim.reshape"(%1201) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1203 = "mix.prim.permute"(%1200) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1204 = "mix.prim.reshape"(%1203) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1205 = "mix.prim.batch_matmul"(%1202, %1204) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1206 = "mix.prim.reshape"(%1205) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1207 = "mix.prim.permute"(%1206) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1208 = "mix.prim.reshape"(%1207) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1209 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1210 = "mix.prim.mul"(%1208, %1209) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1211 = "mix.prim.reshape"(%1210) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1212 = "mix.comp.masked_fill"(%1211, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %1213 = "mix.comp.softmax"(%1212) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1214 = "mix.prim.reshape"(%1213) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1215 = "mix.prim.reshape"(%1143) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1216 = "mix.prim.transpose"(%1215) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1217 = "mix.prim.batch_matmul"(%1214, %1216) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1218 = "mix.prim.reshape"(%1217) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1219 = "mix.prim.permute"(%1218) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1220 = "mix.prim.reshape"(%1219) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1221 = "mix.prim.reshape"(%1220) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1222 = "mix.prim.transpose"(%1129) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1223 = "mix.prim.matmul"(%1221, %1222) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1224 = "mix.prim.add"(%1223, %1130) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1225 = "mix.prim.reshape"(%1224) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1226 = "mix.prim.mul"(%1117, %1225) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1227 = "mix.comp.weight"() <{param_loc = "transformer.h.9.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1228 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1229 = "mix.prim.pow"(%1226, %1228) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1230 = "mix.comp.mean"(%1229) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1231 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1232 = "mix.prim.add"(%1230, %1231) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1233 = "mix.prim.rsqrt"(%1232) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1234 = "mix.prim.mul"(%1226, %1233) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1235 = "mix.prim.mul"(%1227, %1234) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1236 = "mix.module.linear"(%1235) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.9.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1237 = "mix.comp.silu"(%1236) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1238 = "mix.module.linear"(%1235) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.9.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1239 = "mix.prim.mul"(%1237, %1238) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1240 = "mix.module.linear"(%1239) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.9.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1241 = "mix.prim.add"(%1240, %1226) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1242 = "mix.comp.weight"() <{param_loc = "transformer.h.10.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1243 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1244 = "mix.prim.pow"(%1241, %1243) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1245 = "mix.comp.mean"(%1244) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1246 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1247 = "mix.prim.add"(%1245, %1246) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1248 = "mix.prim.rsqrt"(%1247) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1249 = "mix.prim.mul"(%1241, %1248) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1250 = "mix.prim.mul"(%1242, %1249) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1251 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1252 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1253 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1254 = "mix.comp.weight"() <{param_loc = "transformer.h.10.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1255 = "mix.prim.transpose"(%1250) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1256 = "mix.prim.transpose"(%1251) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1257 = "mix.prim.reshape"(%1255) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1258 = "mix.prim.matmul"(%1257, %1256) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1259 = "mix.prim.reshape"(%1258) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1260 = "mix.prim.reshape"(%1259) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1261 = "mix.prim.transpose"(%1252) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1262 = "mix.prim.reshape"(%1255) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1263 = "mix.prim.matmul"(%1262, %1261) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1264 = "mix.prim.reshape"(%1263) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1265 = "mix.prim.reshape"(%1264) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1266 = "mix.prim.slice"(%1265) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1267 = "mix.prim.slice"(%1265) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1268 = "mix.prim.reshape"(%1260) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1269 = "mix.prim.reshape"(%1266) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1270 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1271 = "mix.prim.convert"(%1270) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1272 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1273 = "mix.prim.div"(%1271, %1272) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1274 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1275 = "mix.prim.pow"(%1274, %1273) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1276 = "mix.prim.reciprocal"(%1275) : (tensor<80xf16>) -> tensor<80xf16>
    %1277 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1278 = "mix.prim.mul"(%1277, %1276) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1279 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1280 = "mix.prim.unsqueeze"(%1279) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1281 = "mix.prim.permute"(%1280) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1282 = "mix.prim.unsqueeze"(%1278) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1283 = "mix.prim.permute"(%1282) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1284 = "mix.prim.mul"(%1281, %1283) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1285 = "mix.prim.concat"(%1284, %1284) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1286 = "mix.prim.cos"(%1285) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1287 = "mix.prim.slice"(%1286) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1288 = "mix.prim.unsqueeze"(%1287) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1289 = "mix.prim.slice"(%1288) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1290 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1291 = "mix.prim.mul"(%1289, %1290) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1292 = "mix.prim.sin"(%1285) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1293 = "mix.prim.slice"(%1292) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1294 = "mix.prim.unsqueeze"(%1293) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1295 = "mix.prim.slice"(%1294) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1296 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1297 = "mix.prim.mul"(%1295, %1296) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1298 = "mix.prim.slice"(%1291) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1299 = "mix.prim.slice"(%1297) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1300 = "mix.prim.mul"(%1268, %1298) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1301 = "mix.prim.slice"(%1268) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1302 = "mix.prim.slice"(%1268) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1303 = "mix.prim.neg"(%1302) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1304 = "mix.prim.concat"(%1303, %1301) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1305 = "mix.prim.mul"(%1304, %1299) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1306 = "mix.prim.add"(%1300, %1305) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1307 = "mix.prim.mul"(%1269, %1298) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1308 = "mix.prim.slice"(%1269) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1309 = "mix.prim.slice"(%1269) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1310 = "mix.prim.neg"(%1309) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1311 = "mix.prim.concat"(%1310, %1308) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1312 = "mix.prim.mul"(%1311, %1299) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1313 = "mix.prim.add"(%1307, %1312) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1314 = "mix.prim.reshape"(%1306) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1315 = "mix.prim.reshape"(%1313) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1316 = "mix.prim.reshape"(%1314) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1317 = "mix.prim.reshape"(%1315) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1318 = "mix.prim.transpose"(%1316) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1319 = "mix.prim.transpose"(%1317) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1320 = "mix.prim.transpose"(%1319) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1321 = "mix.prim.unsqueeze"(%1318) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1322 = "mix.prim.permute"(%1321) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1323 = "mix.prim.unsqueeze"(%1320) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1324 = "mix.prim.permute"(%1323) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1325 = "mix.prim.permute"(%1322) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1326 = "mix.prim.reshape"(%1325) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1327 = "mix.prim.permute"(%1324) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1328 = "mix.prim.reshape"(%1327) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1329 = "mix.prim.batch_matmul"(%1326, %1328) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1330 = "mix.prim.reshape"(%1329) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1331 = "mix.prim.permute"(%1330) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1332 = "mix.prim.reshape"(%1331) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1333 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1334 = "mix.prim.mul"(%1332, %1333) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1335 = "mix.prim.reshape"(%1334) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1336 = "mix.comp.masked_fill"(%1335, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %1337 = "mix.comp.softmax"(%1336) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1338 = "mix.prim.reshape"(%1337) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1339 = "mix.prim.reshape"(%1267) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1340 = "mix.prim.transpose"(%1339) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1341 = "mix.prim.batch_matmul"(%1338, %1340) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1342 = "mix.prim.reshape"(%1341) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1343 = "mix.prim.permute"(%1342) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1344 = "mix.prim.reshape"(%1343) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1345 = "mix.prim.reshape"(%1344) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1346 = "mix.prim.transpose"(%1253) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1347 = "mix.prim.matmul"(%1345, %1346) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1348 = "mix.prim.add"(%1347, %1254) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1349 = "mix.prim.reshape"(%1348) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1350 = "mix.prim.mul"(%1241, %1349) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1351 = "mix.comp.weight"() <{param_loc = "transformer.h.10.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1352 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1353 = "mix.prim.pow"(%1350, %1352) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1354 = "mix.comp.mean"(%1353) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1355 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1356 = "mix.prim.add"(%1354, %1355) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1357 = "mix.prim.rsqrt"(%1356) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1358 = "mix.prim.mul"(%1350, %1357) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1359 = "mix.prim.mul"(%1351, %1358) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1360 = "mix.module.linear"(%1359) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.10.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1361 = "mix.comp.silu"(%1360) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1362 = "mix.module.linear"(%1359) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.10.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1363 = "mix.prim.mul"(%1361, %1362) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1364 = "mix.module.linear"(%1363) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.10.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1365 = "mix.prim.add"(%1364, %1350) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1366 = "mix.comp.weight"() <{param_loc = "transformer.h.11.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1367 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1368 = "mix.prim.pow"(%1365, %1367) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1369 = "mix.comp.mean"(%1368) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1370 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1371 = "mix.prim.add"(%1369, %1370) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1372 = "mix.prim.rsqrt"(%1371) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1373 = "mix.prim.mul"(%1365, %1372) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1374 = "mix.prim.mul"(%1366, %1373) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1375 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1376 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1377 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1378 = "mix.comp.weight"() <{param_loc = "transformer.h.11.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1379 = "mix.prim.transpose"(%1374) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1380 = "mix.prim.transpose"(%1375) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1381 = "mix.prim.reshape"(%1379) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1382 = "mix.prim.matmul"(%1381, %1380) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1383 = "mix.prim.reshape"(%1382) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1384 = "mix.prim.reshape"(%1383) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1385 = "mix.prim.transpose"(%1376) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1386 = "mix.prim.reshape"(%1379) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1387 = "mix.prim.matmul"(%1386, %1385) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1388 = "mix.prim.reshape"(%1387) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1389 = "mix.prim.reshape"(%1388) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1390 = "mix.prim.slice"(%1389) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1391 = "mix.prim.slice"(%1389) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1392 = "mix.prim.reshape"(%1384) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1393 = "mix.prim.reshape"(%1390) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1394 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1395 = "mix.prim.convert"(%1394) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1396 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1397 = "mix.prim.div"(%1395, %1396) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1398 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1399 = "mix.prim.pow"(%1398, %1397) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1400 = "mix.prim.reciprocal"(%1399) : (tensor<80xf16>) -> tensor<80xf16>
    %1401 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1402 = "mix.prim.mul"(%1401, %1400) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1403 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1404 = "mix.prim.unsqueeze"(%1403) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1405 = "mix.prim.permute"(%1404) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1406 = "mix.prim.unsqueeze"(%1402) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1407 = "mix.prim.permute"(%1406) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1408 = "mix.prim.mul"(%1405, %1407) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1409 = "mix.prim.concat"(%1408, %1408) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1410 = "mix.prim.cos"(%1409) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1411 = "mix.prim.slice"(%1410) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1412 = "mix.prim.unsqueeze"(%1411) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1413 = "mix.prim.slice"(%1412) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1414 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1415 = "mix.prim.mul"(%1413, %1414) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1416 = "mix.prim.sin"(%1409) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1417 = "mix.prim.slice"(%1416) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1418 = "mix.prim.unsqueeze"(%1417) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1419 = "mix.prim.slice"(%1418) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1420 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1421 = "mix.prim.mul"(%1419, %1420) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1422 = "mix.prim.slice"(%1415) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1423 = "mix.prim.slice"(%1421) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1424 = "mix.prim.mul"(%1392, %1422) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1425 = "mix.prim.slice"(%1392) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1426 = "mix.prim.slice"(%1392) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1427 = "mix.prim.neg"(%1426) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1428 = "mix.prim.concat"(%1427, %1425) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1429 = "mix.prim.mul"(%1428, %1423) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1430 = "mix.prim.add"(%1424, %1429) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1431 = "mix.prim.mul"(%1393, %1422) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1432 = "mix.prim.slice"(%1393) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1433 = "mix.prim.slice"(%1393) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1434 = "mix.prim.neg"(%1433) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1435 = "mix.prim.concat"(%1434, %1432) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1436 = "mix.prim.mul"(%1435, %1423) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1437 = "mix.prim.add"(%1431, %1436) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1438 = "mix.prim.reshape"(%1430) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1439 = "mix.prim.reshape"(%1437) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1440 = "mix.prim.reshape"(%1438) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1441 = "mix.prim.reshape"(%1439) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1442 = "mix.prim.transpose"(%1440) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1443 = "mix.prim.transpose"(%1441) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1444 = "mix.prim.transpose"(%1443) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1445 = "mix.prim.unsqueeze"(%1442) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1446 = "mix.prim.permute"(%1445) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1447 = "mix.prim.unsqueeze"(%1444) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1448 = "mix.prim.permute"(%1447) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1449 = "mix.prim.permute"(%1446) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1450 = "mix.prim.reshape"(%1449) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1451 = "mix.prim.permute"(%1448) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1452 = "mix.prim.reshape"(%1451) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1453 = "mix.prim.batch_matmul"(%1450, %1452) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1454 = "mix.prim.reshape"(%1453) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1455 = "mix.prim.permute"(%1454) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1456 = "mix.prim.reshape"(%1455) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1457 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1458 = "mix.prim.mul"(%1456, %1457) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1459 = "mix.prim.reshape"(%1458) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1460 = "mix.comp.masked_fill"(%1459, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %1461 = "mix.comp.softmax"(%1460) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1462 = "mix.prim.reshape"(%1461) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1463 = "mix.prim.reshape"(%1391) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1464 = "mix.prim.transpose"(%1463) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1465 = "mix.prim.batch_matmul"(%1462, %1464) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1466 = "mix.prim.reshape"(%1465) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1467 = "mix.prim.permute"(%1466) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1468 = "mix.prim.reshape"(%1467) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1469 = "mix.prim.reshape"(%1468) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1470 = "mix.prim.transpose"(%1377) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1471 = "mix.prim.matmul"(%1469, %1470) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1472 = "mix.prim.add"(%1471, %1378) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1473 = "mix.prim.reshape"(%1472) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1474 = "mix.prim.mul"(%1365, %1473) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1475 = "mix.comp.weight"() <{param_loc = "transformer.h.11.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1476 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1477 = "mix.prim.pow"(%1474, %1476) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1478 = "mix.comp.mean"(%1477) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1479 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1480 = "mix.prim.add"(%1478, %1479) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1481 = "mix.prim.rsqrt"(%1480) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1482 = "mix.prim.mul"(%1474, %1481) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1483 = "mix.prim.mul"(%1475, %1482) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1484 = "mix.module.linear"(%1483) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.11.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1485 = "mix.comp.silu"(%1484) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1486 = "mix.module.linear"(%1483) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.11.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1487 = "mix.prim.mul"(%1485, %1486) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1488 = "mix.module.linear"(%1487) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.11.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1489 = "mix.prim.add"(%1488, %1474) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1490 = "mix.comp.weight"() <{param_loc = "transformer.h.12.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1491 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1492 = "mix.prim.pow"(%1489, %1491) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1493 = "mix.comp.mean"(%1492) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1494 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1495 = "mix.prim.add"(%1493, %1494) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1496 = "mix.prim.rsqrt"(%1495) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1497 = "mix.prim.mul"(%1489, %1496) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1498 = "mix.prim.mul"(%1490, %1497) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1499 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1500 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1501 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1502 = "mix.comp.weight"() <{param_loc = "transformer.h.12.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1503 = "mix.prim.transpose"(%1498) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1504 = "mix.prim.transpose"(%1499) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1505 = "mix.prim.reshape"(%1503) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1506 = "mix.prim.matmul"(%1505, %1504) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1507 = "mix.prim.reshape"(%1506) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1508 = "mix.prim.reshape"(%1507) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1509 = "mix.prim.transpose"(%1500) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1510 = "mix.prim.reshape"(%1503) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1511 = "mix.prim.matmul"(%1510, %1509) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1512 = "mix.prim.reshape"(%1511) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1513 = "mix.prim.reshape"(%1512) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1514 = "mix.prim.slice"(%1513) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1515 = "mix.prim.slice"(%1513) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1516 = "mix.prim.reshape"(%1508) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1517 = "mix.prim.reshape"(%1514) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1518 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1519 = "mix.prim.convert"(%1518) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1520 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1521 = "mix.prim.div"(%1519, %1520) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1522 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1523 = "mix.prim.pow"(%1522, %1521) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1524 = "mix.prim.reciprocal"(%1523) : (tensor<80xf16>) -> tensor<80xf16>
    %1525 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1526 = "mix.prim.mul"(%1525, %1524) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1527 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1528 = "mix.prim.unsqueeze"(%1527) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1529 = "mix.prim.permute"(%1528) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1530 = "mix.prim.unsqueeze"(%1526) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1531 = "mix.prim.permute"(%1530) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1532 = "mix.prim.mul"(%1529, %1531) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1533 = "mix.prim.concat"(%1532, %1532) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1534 = "mix.prim.cos"(%1533) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1535 = "mix.prim.slice"(%1534) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1536 = "mix.prim.unsqueeze"(%1535) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1537 = "mix.prim.slice"(%1536) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1538 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1539 = "mix.prim.mul"(%1537, %1538) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1540 = "mix.prim.sin"(%1533) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1541 = "mix.prim.slice"(%1540) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1542 = "mix.prim.unsqueeze"(%1541) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1543 = "mix.prim.slice"(%1542) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1544 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1545 = "mix.prim.mul"(%1543, %1544) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1546 = "mix.prim.slice"(%1539) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1547 = "mix.prim.slice"(%1545) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1548 = "mix.prim.mul"(%1516, %1546) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1549 = "mix.prim.slice"(%1516) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1550 = "mix.prim.slice"(%1516) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1551 = "mix.prim.neg"(%1550) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1552 = "mix.prim.concat"(%1551, %1549) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1553 = "mix.prim.mul"(%1552, %1547) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1554 = "mix.prim.add"(%1548, %1553) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1555 = "mix.prim.mul"(%1517, %1546) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1556 = "mix.prim.slice"(%1517) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1557 = "mix.prim.slice"(%1517) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1558 = "mix.prim.neg"(%1557) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1559 = "mix.prim.concat"(%1558, %1556) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1560 = "mix.prim.mul"(%1559, %1547) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1561 = "mix.prim.add"(%1555, %1560) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1562 = "mix.prim.reshape"(%1554) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1563 = "mix.prim.reshape"(%1561) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1564 = "mix.prim.reshape"(%1562) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1565 = "mix.prim.reshape"(%1563) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1566 = "mix.prim.transpose"(%1564) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1567 = "mix.prim.transpose"(%1565) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1568 = "mix.prim.transpose"(%1567) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1569 = "mix.prim.unsqueeze"(%1566) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1570 = "mix.prim.permute"(%1569) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1571 = "mix.prim.unsqueeze"(%1568) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1572 = "mix.prim.permute"(%1571) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1573 = "mix.prim.permute"(%1570) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1574 = "mix.prim.reshape"(%1573) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1575 = "mix.prim.permute"(%1572) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1576 = "mix.prim.reshape"(%1575) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1577 = "mix.prim.batch_matmul"(%1574, %1576) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1578 = "mix.prim.reshape"(%1577) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1579 = "mix.prim.permute"(%1578) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1580 = "mix.prim.reshape"(%1579) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1581 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1582 = "mix.prim.mul"(%1580, %1581) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1583 = "mix.prim.reshape"(%1582) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1584 = "mix.comp.masked_fill"(%1583, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %1585 = "mix.comp.softmax"(%1584) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1586 = "mix.prim.reshape"(%1585) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1587 = "mix.prim.reshape"(%1515) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1588 = "mix.prim.transpose"(%1587) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1589 = "mix.prim.batch_matmul"(%1586, %1588) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1590 = "mix.prim.reshape"(%1589) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1591 = "mix.prim.permute"(%1590) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1592 = "mix.prim.reshape"(%1591) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1593 = "mix.prim.reshape"(%1592) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1594 = "mix.prim.transpose"(%1501) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1595 = "mix.prim.matmul"(%1593, %1594) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1596 = "mix.prim.add"(%1595, %1502) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1597 = "mix.prim.reshape"(%1596) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1598 = "mix.prim.mul"(%1489, %1597) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1599 = "mix.comp.weight"() <{param_loc = "transformer.h.12.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1600 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1601 = "mix.prim.pow"(%1598, %1600) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1602 = "mix.comp.mean"(%1601) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1603 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1604 = "mix.prim.add"(%1602, %1603) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1605 = "mix.prim.rsqrt"(%1604) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1606 = "mix.prim.mul"(%1598, %1605) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1607 = "mix.prim.mul"(%1599, %1606) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1608 = "mix.module.linear"(%1607) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.12.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1609 = "mix.comp.silu"(%1608) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1610 = "mix.module.linear"(%1607) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.12.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1611 = "mix.prim.mul"(%1609, %1610) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1612 = "mix.module.linear"(%1611) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.12.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1613 = "mix.prim.add"(%1612, %1598) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1614 = "mix.comp.weight"() <{param_loc = "transformer.h.13.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1615 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1616 = "mix.prim.pow"(%1613, %1615) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1617 = "mix.comp.mean"(%1616) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1618 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1619 = "mix.prim.add"(%1617, %1618) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1620 = "mix.prim.rsqrt"(%1619) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1621 = "mix.prim.mul"(%1613, %1620) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1622 = "mix.prim.mul"(%1614, %1621) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1623 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1624 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1625 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1626 = "mix.comp.weight"() <{param_loc = "transformer.h.13.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1627 = "mix.prim.transpose"(%1622) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1628 = "mix.prim.transpose"(%1623) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1629 = "mix.prim.reshape"(%1627) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1630 = "mix.prim.matmul"(%1629, %1628) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1631 = "mix.prim.reshape"(%1630) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1632 = "mix.prim.reshape"(%1631) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1633 = "mix.prim.transpose"(%1624) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1634 = "mix.prim.reshape"(%1627) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1635 = "mix.prim.matmul"(%1634, %1633) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1636 = "mix.prim.reshape"(%1635) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1637 = "mix.prim.reshape"(%1636) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1638 = "mix.prim.slice"(%1637) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1639 = "mix.prim.slice"(%1637) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1640 = "mix.prim.reshape"(%1632) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1641 = "mix.prim.reshape"(%1638) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1642 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1643 = "mix.prim.convert"(%1642) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1644 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1645 = "mix.prim.div"(%1643, %1644) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1646 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1647 = "mix.prim.pow"(%1646, %1645) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1648 = "mix.prim.reciprocal"(%1647) : (tensor<80xf16>) -> tensor<80xf16>
    %1649 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1650 = "mix.prim.mul"(%1649, %1648) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1651 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1652 = "mix.prim.unsqueeze"(%1651) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1653 = "mix.prim.permute"(%1652) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1654 = "mix.prim.unsqueeze"(%1650) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1655 = "mix.prim.permute"(%1654) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1656 = "mix.prim.mul"(%1653, %1655) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1657 = "mix.prim.concat"(%1656, %1656) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1658 = "mix.prim.cos"(%1657) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1659 = "mix.prim.slice"(%1658) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1660 = "mix.prim.unsqueeze"(%1659) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1661 = "mix.prim.slice"(%1660) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1662 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1663 = "mix.prim.mul"(%1661, %1662) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1664 = "mix.prim.sin"(%1657) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1665 = "mix.prim.slice"(%1664) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1666 = "mix.prim.unsqueeze"(%1665) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1667 = "mix.prim.slice"(%1666) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1668 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1669 = "mix.prim.mul"(%1667, %1668) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1670 = "mix.prim.slice"(%1663) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1671 = "mix.prim.slice"(%1669) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1672 = "mix.prim.mul"(%1640, %1670) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1673 = "mix.prim.slice"(%1640) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1674 = "mix.prim.slice"(%1640) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1675 = "mix.prim.neg"(%1674) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1676 = "mix.prim.concat"(%1675, %1673) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1677 = "mix.prim.mul"(%1676, %1671) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1678 = "mix.prim.add"(%1672, %1677) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1679 = "mix.prim.mul"(%1641, %1670) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1680 = "mix.prim.slice"(%1641) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1681 = "mix.prim.slice"(%1641) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1682 = "mix.prim.neg"(%1681) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1683 = "mix.prim.concat"(%1682, %1680) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1684 = "mix.prim.mul"(%1683, %1671) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1685 = "mix.prim.add"(%1679, %1684) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1686 = "mix.prim.reshape"(%1678) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1687 = "mix.prim.reshape"(%1685) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1688 = "mix.prim.reshape"(%1686) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1689 = "mix.prim.reshape"(%1687) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1690 = "mix.prim.transpose"(%1688) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1691 = "mix.prim.transpose"(%1689) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1692 = "mix.prim.transpose"(%1691) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1693 = "mix.prim.unsqueeze"(%1690) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1694 = "mix.prim.permute"(%1693) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1695 = "mix.prim.unsqueeze"(%1692) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1696 = "mix.prim.permute"(%1695) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1697 = "mix.prim.permute"(%1694) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1698 = "mix.prim.reshape"(%1697) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1699 = "mix.prim.permute"(%1696) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1700 = "mix.prim.reshape"(%1699) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1701 = "mix.prim.batch_matmul"(%1698, %1700) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1702 = "mix.prim.reshape"(%1701) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1703 = "mix.prim.permute"(%1702) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1704 = "mix.prim.reshape"(%1703) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1705 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1706 = "mix.prim.mul"(%1704, %1705) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1707 = "mix.prim.reshape"(%1706) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1708 = "mix.comp.masked_fill"(%1707, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %1709 = "mix.comp.softmax"(%1708) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1710 = "mix.prim.reshape"(%1709) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1711 = "mix.prim.reshape"(%1639) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1712 = "mix.prim.transpose"(%1711) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1713 = "mix.prim.batch_matmul"(%1710, %1712) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1714 = "mix.prim.reshape"(%1713) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1715 = "mix.prim.permute"(%1714) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1716 = "mix.prim.reshape"(%1715) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1717 = "mix.prim.reshape"(%1716) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1718 = "mix.prim.transpose"(%1625) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1719 = "mix.prim.matmul"(%1717, %1718) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1720 = "mix.prim.add"(%1719, %1626) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1721 = "mix.prim.reshape"(%1720) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1722 = "mix.prim.mul"(%1613, %1721) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1723 = "mix.comp.weight"() <{param_loc = "transformer.h.13.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1724 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1725 = "mix.prim.pow"(%1722, %1724) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1726 = "mix.comp.mean"(%1725) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1727 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1728 = "mix.prim.add"(%1726, %1727) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1729 = "mix.prim.rsqrt"(%1728) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1730 = "mix.prim.mul"(%1722, %1729) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1731 = "mix.prim.mul"(%1723, %1730) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1732 = "mix.module.linear"(%1731) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.13.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1733 = "mix.comp.silu"(%1732) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1734 = "mix.module.linear"(%1731) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.13.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1735 = "mix.prim.mul"(%1733, %1734) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1736 = "mix.module.linear"(%1735) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.13.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1737 = "mix.prim.add"(%1736, %1722) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1738 = "mix.comp.weight"() <{param_loc = "transformer.h.14.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1739 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1740 = "mix.prim.pow"(%1737, %1739) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1741 = "mix.comp.mean"(%1740) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1742 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1743 = "mix.prim.add"(%1741, %1742) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1744 = "mix.prim.rsqrt"(%1743) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1745 = "mix.prim.mul"(%1737, %1744) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1746 = "mix.prim.mul"(%1738, %1745) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1747 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1748 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1749 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1750 = "mix.comp.weight"() <{param_loc = "transformer.h.14.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1751 = "mix.prim.transpose"(%1746) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1752 = "mix.prim.transpose"(%1747) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1753 = "mix.prim.reshape"(%1751) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1754 = "mix.prim.matmul"(%1753, %1752) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1755 = "mix.prim.reshape"(%1754) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1756 = "mix.prim.reshape"(%1755) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1757 = "mix.prim.transpose"(%1748) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1758 = "mix.prim.reshape"(%1751) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1759 = "mix.prim.matmul"(%1758, %1757) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1760 = "mix.prim.reshape"(%1759) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1761 = "mix.prim.reshape"(%1760) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1762 = "mix.prim.slice"(%1761) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1763 = "mix.prim.slice"(%1761) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1764 = "mix.prim.reshape"(%1756) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1765 = "mix.prim.reshape"(%1762) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1766 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1767 = "mix.prim.convert"(%1766) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1768 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1769 = "mix.prim.div"(%1767, %1768) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1770 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1771 = "mix.prim.pow"(%1770, %1769) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1772 = "mix.prim.reciprocal"(%1771) : (tensor<80xf16>) -> tensor<80xf16>
    %1773 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1774 = "mix.prim.mul"(%1773, %1772) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1775 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1776 = "mix.prim.unsqueeze"(%1775) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1777 = "mix.prim.permute"(%1776) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1778 = "mix.prim.unsqueeze"(%1774) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1779 = "mix.prim.permute"(%1778) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1780 = "mix.prim.mul"(%1777, %1779) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1781 = "mix.prim.concat"(%1780, %1780) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1782 = "mix.prim.cos"(%1781) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1783 = "mix.prim.slice"(%1782) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1784 = "mix.prim.unsqueeze"(%1783) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1785 = "mix.prim.slice"(%1784) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1786 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1787 = "mix.prim.mul"(%1785, %1786) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1788 = "mix.prim.sin"(%1781) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1789 = "mix.prim.slice"(%1788) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1790 = "mix.prim.unsqueeze"(%1789) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1791 = "mix.prim.slice"(%1790) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1792 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1793 = "mix.prim.mul"(%1791, %1792) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1794 = "mix.prim.slice"(%1787) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1795 = "mix.prim.slice"(%1793) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1796 = "mix.prim.mul"(%1764, %1794) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1797 = "mix.prim.slice"(%1764) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1798 = "mix.prim.slice"(%1764) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1799 = "mix.prim.neg"(%1798) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1800 = "mix.prim.concat"(%1799, %1797) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1801 = "mix.prim.mul"(%1800, %1795) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1802 = "mix.prim.add"(%1796, %1801) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1803 = "mix.prim.mul"(%1765, %1794) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1804 = "mix.prim.slice"(%1765) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1805 = "mix.prim.slice"(%1765) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1806 = "mix.prim.neg"(%1805) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1807 = "mix.prim.concat"(%1806, %1804) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1808 = "mix.prim.mul"(%1807, %1795) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1809 = "mix.prim.add"(%1803, %1808) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1810 = "mix.prim.reshape"(%1802) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1811 = "mix.prim.reshape"(%1809) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1812 = "mix.prim.reshape"(%1810) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1813 = "mix.prim.reshape"(%1811) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1814 = "mix.prim.transpose"(%1812) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1815 = "mix.prim.transpose"(%1813) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1816 = "mix.prim.transpose"(%1815) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1817 = "mix.prim.unsqueeze"(%1814) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1818 = "mix.prim.permute"(%1817) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1819 = "mix.prim.unsqueeze"(%1816) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1820 = "mix.prim.permute"(%1819) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1821 = "mix.prim.permute"(%1818) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1822 = "mix.prim.reshape"(%1821) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1823 = "mix.prim.permute"(%1820) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1824 = "mix.prim.reshape"(%1823) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1825 = "mix.prim.batch_matmul"(%1822, %1824) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1826 = "mix.prim.reshape"(%1825) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1827 = "mix.prim.permute"(%1826) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1828 = "mix.prim.reshape"(%1827) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1829 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1830 = "mix.prim.mul"(%1828, %1829) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1831 = "mix.prim.reshape"(%1830) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1832 = "mix.comp.masked_fill"(%1831, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %1833 = "mix.comp.softmax"(%1832) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1834 = "mix.prim.reshape"(%1833) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1835 = "mix.prim.reshape"(%1763) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1836 = "mix.prim.transpose"(%1835) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1837 = "mix.prim.batch_matmul"(%1834, %1836) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1838 = "mix.prim.reshape"(%1837) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1839 = "mix.prim.permute"(%1838) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1840 = "mix.prim.reshape"(%1839) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1841 = "mix.prim.reshape"(%1840) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1842 = "mix.prim.transpose"(%1749) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1843 = "mix.prim.matmul"(%1841, %1842) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1844 = "mix.prim.add"(%1843, %1750) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1845 = "mix.prim.reshape"(%1844) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1846 = "mix.prim.mul"(%1737, %1845) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1847 = "mix.comp.weight"() <{param_loc = "transformer.h.14.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1848 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1849 = "mix.prim.pow"(%1846, %1848) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1850 = "mix.comp.mean"(%1849) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1851 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1852 = "mix.prim.add"(%1850, %1851) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1853 = "mix.prim.rsqrt"(%1852) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1854 = "mix.prim.mul"(%1846, %1853) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1855 = "mix.prim.mul"(%1847, %1854) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1856 = "mix.module.linear"(%1855) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.14.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1857 = "mix.comp.silu"(%1856) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1858 = "mix.module.linear"(%1855) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.14.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1859 = "mix.prim.mul"(%1857, %1858) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1860 = "mix.module.linear"(%1859) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.14.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1861 = "mix.prim.add"(%1860, %1846) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1862 = "mix.comp.weight"() <{param_loc = "transformer.h.15.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1863 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1864 = "mix.prim.pow"(%1861, %1863) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1865 = "mix.comp.mean"(%1864) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1866 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1867 = "mix.prim.add"(%1865, %1866) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1868 = "mix.prim.rsqrt"(%1867) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1869 = "mix.prim.mul"(%1861, %1868) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1870 = "mix.prim.mul"(%1862, %1869) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1871 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1872 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1873 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1874 = "mix.comp.weight"() <{param_loc = "transformer.h.15.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1875 = "mix.prim.transpose"(%1870) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %1876 = "mix.prim.transpose"(%1871) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1877 = "mix.prim.reshape"(%1875) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1878 = "mix.prim.matmul"(%1877, %1876) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1879 = "mix.prim.reshape"(%1878) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %1880 = "mix.prim.reshape"(%1879) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %1881 = "mix.prim.transpose"(%1872) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %1882 = "mix.prim.reshape"(%1875) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %1883 = "mix.prim.matmul"(%1882, %1881) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %1884 = "mix.prim.reshape"(%1883) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %1885 = "mix.prim.reshape"(%1884) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %1886 = "mix.prim.slice"(%1885) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1887 = "mix.prim.slice"(%1885) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %1888 = "mix.prim.reshape"(%1880) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1889 = "mix.prim.reshape"(%1886) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1890 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %1891 = "mix.prim.convert"(%1890) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %1892 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %1893 = "mix.prim.div"(%1891, %1892) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %1894 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %1895 = "mix.prim.pow"(%1894, %1893) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1896 = "mix.prim.reciprocal"(%1895) : (tensor<80xf16>) -> tensor<80xf16>
    %1897 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1898 = "mix.prim.mul"(%1897, %1896) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %1899 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %1900 = "mix.prim.unsqueeze"(%1899) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %1901 = "mix.prim.permute"(%1900) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %1902 = "mix.prim.unsqueeze"(%1898) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %1903 = "mix.prim.permute"(%1902) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %1904 = "mix.prim.mul"(%1901, %1903) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %1905 = "mix.prim.concat"(%1904, %1904) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %1906 = "mix.prim.cos"(%1905) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1907 = "mix.prim.slice"(%1906) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1908 = "mix.prim.unsqueeze"(%1907) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1909 = "mix.prim.slice"(%1908) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1910 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1911 = "mix.prim.mul"(%1909, %1910) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1912 = "mix.prim.sin"(%1905) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1913 = "mix.prim.slice"(%1912) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %1914 = "mix.prim.unsqueeze"(%1913) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %1915 = "mix.prim.slice"(%1914) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1916 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %1917 = "mix.prim.mul"(%1915, %1916) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %1918 = "mix.prim.slice"(%1911) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1919 = "mix.prim.slice"(%1917) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %1920 = "mix.prim.mul"(%1888, %1918) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1921 = "mix.prim.slice"(%1888) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1922 = "mix.prim.slice"(%1888) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1923 = "mix.prim.neg"(%1922) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1924 = "mix.prim.concat"(%1923, %1921) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1925 = "mix.prim.mul"(%1924, %1919) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1926 = "mix.prim.add"(%1920, %1925) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1927 = "mix.prim.mul"(%1889, %1918) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1928 = "mix.prim.slice"(%1889) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1929 = "mix.prim.slice"(%1889) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %1930 = "mix.prim.neg"(%1929) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %1931 = "mix.prim.concat"(%1930, %1928) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %1932 = "mix.prim.mul"(%1931, %1919) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %1933 = "mix.prim.add"(%1927, %1932) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %1934 = "mix.prim.reshape"(%1926) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1935 = "mix.prim.reshape"(%1933) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %1936 = "mix.prim.reshape"(%1934) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1937 = "mix.prim.reshape"(%1935) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1938 = "mix.prim.transpose"(%1936) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1939 = "mix.prim.transpose"(%1937) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1940 = "mix.prim.transpose"(%1939) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %1941 = "mix.prim.unsqueeze"(%1938) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %1942 = "mix.prim.permute"(%1941) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %1943 = "mix.prim.unsqueeze"(%1940) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %1944 = "mix.prim.permute"(%1943) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %1945 = "mix.prim.permute"(%1942) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %1946 = "mix.prim.reshape"(%1945) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %1947 = "mix.prim.permute"(%1944) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %1948 = "mix.prim.reshape"(%1947) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %1949 = "mix.prim.batch_matmul"(%1946, %1948) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %1950 = "mix.prim.reshape"(%1949) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %1951 = "mix.prim.permute"(%1950) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %1952 = "mix.prim.reshape"(%1951) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %1953 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %1954 = "mix.prim.mul"(%1952, %1953) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %1955 = "mix.prim.reshape"(%1954) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1956 = "mix.comp.masked_fill"(%1955, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %1957 = "mix.comp.softmax"(%1956) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %1958 = "mix.prim.reshape"(%1957) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %1959 = "mix.prim.reshape"(%1887) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %1960 = "mix.prim.transpose"(%1959) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %1961 = "mix.prim.batch_matmul"(%1958, %1960) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %1962 = "mix.prim.reshape"(%1961) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %1963 = "mix.prim.permute"(%1962) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %1964 = "mix.prim.reshape"(%1963) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %1965 = "mix.prim.reshape"(%1964) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %1966 = "mix.prim.transpose"(%1873) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %1967 = "mix.prim.matmul"(%1965, %1966) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %1968 = "mix.prim.add"(%1967, %1874) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %1969 = "mix.prim.reshape"(%1968) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %1970 = "mix.prim.mul"(%1861, %1969) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1971 = "mix.comp.weight"() <{param_loc = "transformer.h.15.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %1972 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1973 = "mix.prim.pow"(%1970, %1972) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1974 = "mix.comp.mean"(%1973) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1975 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1976 = "mix.prim.add"(%1974, %1975) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1977 = "mix.prim.rsqrt"(%1976) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1978 = "mix.prim.mul"(%1970, %1977) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1979 = "mix.prim.mul"(%1971, %1978) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1980 = "mix.module.linear"(%1979) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.15.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1981 = "mix.comp.silu"(%1980) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1982 = "mix.module.linear"(%1979) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.15.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %1983 = "mix.prim.mul"(%1981, %1982) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %1984 = "mix.module.linear"(%1983) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.15.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %1985 = "mix.prim.add"(%1984, %1970) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1986 = "mix.comp.weight"() <{param_loc = "transformer.h.16.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %1987 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %1988 = "mix.prim.pow"(%1985, %1987) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %1989 = "mix.comp.mean"(%1988) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %1990 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %1991 = "mix.prim.add"(%1989, %1990) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %1992 = "mix.prim.rsqrt"(%1991) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %1993 = "mix.prim.mul"(%1985, %1992) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %1994 = "mix.prim.mul"(%1986, %1993) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %1995 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %1996 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %1997 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %1998 = "mix.comp.weight"() <{param_loc = "transformer.h.16.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %1999 = "mix.prim.transpose"(%1994) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2000 = "mix.prim.transpose"(%1995) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2001 = "mix.prim.reshape"(%1999) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2002 = "mix.prim.matmul"(%2001, %2000) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2003 = "mix.prim.reshape"(%2002) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2004 = "mix.prim.reshape"(%2003) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2005 = "mix.prim.transpose"(%1996) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2006 = "mix.prim.reshape"(%1999) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2007 = "mix.prim.matmul"(%2006, %2005) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2008 = "mix.prim.reshape"(%2007) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2009 = "mix.prim.reshape"(%2008) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2010 = "mix.prim.slice"(%2009) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2011 = "mix.prim.slice"(%2009) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2012 = "mix.prim.reshape"(%2004) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2013 = "mix.prim.reshape"(%2010) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2014 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2015 = "mix.prim.convert"(%2014) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2016 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2017 = "mix.prim.div"(%2015, %2016) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2018 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2019 = "mix.prim.pow"(%2018, %2017) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2020 = "mix.prim.reciprocal"(%2019) : (tensor<80xf16>) -> tensor<80xf16>
    %2021 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2022 = "mix.prim.mul"(%2021, %2020) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2023 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2024 = "mix.prim.unsqueeze"(%2023) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2025 = "mix.prim.permute"(%2024) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2026 = "mix.prim.unsqueeze"(%2022) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2027 = "mix.prim.permute"(%2026) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2028 = "mix.prim.mul"(%2025, %2027) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2029 = "mix.prim.concat"(%2028, %2028) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2030 = "mix.prim.cos"(%2029) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2031 = "mix.prim.slice"(%2030) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2032 = "mix.prim.unsqueeze"(%2031) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2033 = "mix.prim.slice"(%2032) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2034 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2035 = "mix.prim.mul"(%2033, %2034) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2036 = "mix.prim.sin"(%2029) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2037 = "mix.prim.slice"(%2036) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2038 = "mix.prim.unsqueeze"(%2037) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2039 = "mix.prim.slice"(%2038) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2040 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2041 = "mix.prim.mul"(%2039, %2040) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2042 = "mix.prim.slice"(%2035) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2043 = "mix.prim.slice"(%2041) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2044 = "mix.prim.mul"(%2012, %2042) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2045 = "mix.prim.slice"(%2012) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2046 = "mix.prim.slice"(%2012) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2047 = "mix.prim.neg"(%2046) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2048 = "mix.prim.concat"(%2047, %2045) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2049 = "mix.prim.mul"(%2048, %2043) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2050 = "mix.prim.add"(%2044, %2049) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2051 = "mix.prim.mul"(%2013, %2042) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2052 = "mix.prim.slice"(%2013) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2053 = "mix.prim.slice"(%2013) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2054 = "mix.prim.neg"(%2053) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2055 = "mix.prim.concat"(%2054, %2052) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2056 = "mix.prim.mul"(%2055, %2043) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2057 = "mix.prim.add"(%2051, %2056) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2058 = "mix.prim.reshape"(%2050) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2059 = "mix.prim.reshape"(%2057) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2060 = "mix.prim.reshape"(%2058) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2061 = "mix.prim.reshape"(%2059) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2062 = "mix.prim.transpose"(%2060) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2063 = "mix.prim.transpose"(%2061) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2064 = "mix.prim.transpose"(%2063) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2065 = "mix.prim.unsqueeze"(%2062) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2066 = "mix.prim.permute"(%2065) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2067 = "mix.prim.unsqueeze"(%2064) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2068 = "mix.prim.permute"(%2067) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2069 = "mix.prim.permute"(%2066) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2070 = "mix.prim.reshape"(%2069) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2071 = "mix.prim.permute"(%2068) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2072 = "mix.prim.reshape"(%2071) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2073 = "mix.prim.batch_matmul"(%2070, %2072) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2074 = "mix.prim.reshape"(%2073) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2075 = "mix.prim.permute"(%2074) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2076 = "mix.prim.reshape"(%2075) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2077 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2078 = "mix.prim.mul"(%2076, %2077) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2079 = "mix.prim.reshape"(%2078) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2080 = "mix.comp.masked_fill"(%2079, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %2081 = "mix.comp.softmax"(%2080) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2082 = "mix.prim.reshape"(%2081) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2083 = "mix.prim.reshape"(%2011) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2084 = "mix.prim.transpose"(%2083) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2085 = "mix.prim.batch_matmul"(%2082, %2084) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2086 = "mix.prim.reshape"(%2085) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2087 = "mix.prim.permute"(%2086) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2088 = "mix.prim.reshape"(%2087) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2089 = "mix.prim.reshape"(%2088) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2090 = "mix.prim.transpose"(%1997) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2091 = "mix.prim.matmul"(%2089, %2090) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2092 = "mix.prim.add"(%2091, %1998) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2093 = "mix.prim.reshape"(%2092) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2094 = "mix.prim.mul"(%1985, %2093) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2095 = "mix.comp.weight"() <{param_loc = "transformer.h.16.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2096 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2097 = "mix.prim.pow"(%2094, %2096) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2098 = "mix.comp.mean"(%2097) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2099 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2100 = "mix.prim.add"(%2098, %2099) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2101 = "mix.prim.rsqrt"(%2100) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2102 = "mix.prim.mul"(%2094, %2101) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2103 = "mix.prim.mul"(%2095, %2102) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2104 = "mix.module.linear"(%2103) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.16.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2105 = "mix.comp.silu"(%2104) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2106 = "mix.module.linear"(%2103) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.16.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2107 = "mix.prim.mul"(%2105, %2106) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2108 = "mix.module.linear"(%2107) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.16.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2109 = "mix.prim.add"(%2108, %2094) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2110 = "mix.comp.weight"() <{param_loc = "transformer.h.17.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2111 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2112 = "mix.prim.pow"(%2109, %2111) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2113 = "mix.comp.mean"(%2112) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2114 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2115 = "mix.prim.add"(%2113, %2114) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2116 = "mix.prim.rsqrt"(%2115) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2117 = "mix.prim.mul"(%2109, %2116) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2118 = "mix.prim.mul"(%2110, %2117) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2119 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2120 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2121 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2122 = "mix.comp.weight"() <{param_loc = "transformer.h.17.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2123 = "mix.prim.transpose"(%2118) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2124 = "mix.prim.transpose"(%2119) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2125 = "mix.prim.reshape"(%2123) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2126 = "mix.prim.matmul"(%2125, %2124) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2127 = "mix.prim.reshape"(%2126) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2128 = "mix.prim.reshape"(%2127) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2129 = "mix.prim.transpose"(%2120) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2130 = "mix.prim.reshape"(%2123) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2131 = "mix.prim.matmul"(%2130, %2129) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2132 = "mix.prim.reshape"(%2131) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2133 = "mix.prim.reshape"(%2132) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2134 = "mix.prim.slice"(%2133) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2135 = "mix.prim.slice"(%2133) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2136 = "mix.prim.reshape"(%2128) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2137 = "mix.prim.reshape"(%2134) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2138 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2139 = "mix.prim.convert"(%2138) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2140 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2141 = "mix.prim.div"(%2139, %2140) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2142 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2143 = "mix.prim.pow"(%2142, %2141) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2144 = "mix.prim.reciprocal"(%2143) : (tensor<80xf16>) -> tensor<80xf16>
    %2145 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2146 = "mix.prim.mul"(%2145, %2144) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2147 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2148 = "mix.prim.unsqueeze"(%2147) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2149 = "mix.prim.permute"(%2148) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2150 = "mix.prim.unsqueeze"(%2146) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2151 = "mix.prim.permute"(%2150) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2152 = "mix.prim.mul"(%2149, %2151) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2153 = "mix.prim.concat"(%2152, %2152) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2154 = "mix.prim.cos"(%2153) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2155 = "mix.prim.slice"(%2154) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2156 = "mix.prim.unsqueeze"(%2155) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2157 = "mix.prim.slice"(%2156) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2158 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2159 = "mix.prim.mul"(%2157, %2158) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2160 = "mix.prim.sin"(%2153) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2161 = "mix.prim.slice"(%2160) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2162 = "mix.prim.unsqueeze"(%2161) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2163 = "mix.prim.slice"(%2162) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2164 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2165 = "mix.prim.mul"(%2163, %2164) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2166 = "mix.prim.slice"(%2159) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2167 = "mix.prim.slice"(%2165) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2168 = "mix.prim.mul"(%2136, %2166) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2169 = "mix.prim.slice"(%2136) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2170 = "mix.prim.slice"(%2136) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2171 = "mix.prim.neg"(%2170) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2172 = "mix.prim.concat"(%2171, %2169) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2173 = "mix.prim.mul"(%2172, %2167) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2174 = "mix.prim.add"(%2168, %2173) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2175 = "mix.prim.mul"(%2137, %2166) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2176 = "mix.prim.slice"(%2137) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2177 = "mix.prim.slice"(%2137) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2178 = "mix.prim.neg"(%2177) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2179 = "mix.prim.concat"(%2178, %2176) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2180 = "mix.prim.mul"(%2179, %2167) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2181 = "mix.prim.add"(%2175, %2180) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2182 = "mix.prim.reshape"(%2174) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2183 = "mix.prim.reshape"(%2181) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2184 = "mix.prim.reshape"(%2182) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2185 = "mix.prim.reshape"(%2183) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2186 = "mix.prim.transpose"(%2184) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2187 = "mix.prim.transpose"(%2185) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2188 = "mix.prim.transpose"(%2187) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2189 = "mix.prim.unsqueeze"(%2186) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2190 = "mix.prim.permute"(%2189) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2191 = "mix.prim.unsqueeze"(%2188) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2192 = "mix.prim.permute"(%2191) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2193 = "mix.prim.permute"(%2190) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2194 = "mix.prim.reshape"(%2193) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2195 = "mix.prim.permute"(%2192) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2196 = "mix.prim.reshape"(%2195) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2197 = "mix.prim.batch_matmul"(%2194, %2196) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2198 = "mix.prim.reshape"(%2197) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2199 = "mix.prim.permute"(%2198) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2200 = "mix.prim.reshape"(%2199) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2201 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2202 = "mix.prim.mul"(%2200, %2201) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2203 = "mix.prim.reshape"(%2202) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2204 = "mix.comp.masked_fill"(%2203, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %2205 = "mix.comp.softmax"(%2204) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2206 = "mix.prim.reshape"(%2205) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2207 = "mix.prim.reshape"(%2135) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2208 = "mix.prim.transpose"(%2207) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2209 = "mix.prim.batch_matmul"(%2206, %2208) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2210 = "mix.prim.reshape"(%2209) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2211 = "mix.prim.permute"(%2210) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2212 = "mix.prim.reshape"(%2211) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2213 = "mix.prim.reshape"(%2212) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2214 = "mix.prim.transpose"(%2121) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2215 = "mix.prim.matmul"(%2213, %2214) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2216 = "mix.prim.add"(%2215, %2122) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2217 = "mix.prim.reshape"(%2216) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2218 = "mix.prim.mul"(%2109, %2217) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2219 = "mix.comp.weight"() <{param_loc = "transformer.h.17.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2220 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2221 = "mix.prim.pow"(%2218, %2220) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2222 = "mix.comp.mean"(%2221) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2223 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2224 = "mix.prim.add"(%2222, %2223) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2225 = "mix.prim.rsqrt"(%2224) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2226 = "mix.prim.mul"(%2218, %2225) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2227 = "mix.prim.mul"(%2219, %2226) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2228 = "mix.module.linear"(%2227) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.17.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2229 = "mix.comp.silu"(%2228) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2230 = "mix.module.linear"(%2227) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.17.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2231 = "mix.prim.mul"(%2229, %2230) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2232 = "mix.module.linear"(%2231) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.17.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2233 = "mix.prim.add"(%2232, %2218) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2234 = "mix.comp.weight"() <{param_loc = "transformer.h.18.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2235 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2236 = "mix.prim.pow"(%2233, %2235) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2237 = "mix.comp.mean"(%2236) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2238 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2239 = "mix.prim.add"(%2237, %2238) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2240 = "mix.prim.rsqrt"(%2239) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2241 = "mix.prim.mul"(%2233, %2240) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2242 = "mix.prim.mul"(%2234, %2241) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2243 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2244 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2245 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2246 = "mix.comp.weight"() <{param_loc = "transformer.h.18.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2247 = "mix.prim.transpose"(%2242) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2248 = "mix.prim.transpose"(%2243) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2249 = "mix.prim.reshape"(%2247) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2250 = "mix.prim.matmul"(%2249, %2248) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2251 = "mix.prim.reshape"(%2250) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2252 = "mix.prim.reshape"(%2251) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2253 = "mix.prim.transpose"(%2244) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2254 = "mix.prim.reshape"(%2247) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2255 = "mix.prim.matmul"(%2254, %2253) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2256 = "mix.prim.reshape"(%2255) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2257 = "mix.prim.reshape"(%2256) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2258 = "mix.prim.slice"(%2257) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2259 = "mix.prim.slice"(%2257) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2260 = "mix.prim.reshape"(%2252) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2261 = "mix.prim.reshape"(%2258) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2262 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2263 = "mix.prim.convert"(%2262) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2264 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2265 = "mix.prim.div"(%2263, %2264) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2266 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2267 = "mix.prim.pow"(%2266, %2265) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2268 = "mix.prim.reciprocal"(%2267) : (tensor<80xf16>) -> tensor<80xf16>
    %2269 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2270 = "mix.prim.mul"(%2269, %2268) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2271 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2272 = "mix.prim.unsqueeze"(%2271) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2273 = "mix.prim.permute"(%2272) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2274 = "mix.prim.unsqueeze"(%2270) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2275 = "mix.prim.permute"(%2274) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2276 = "mix.prim.mul"(%2273, %2275) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2277 = "mix.prim.concat"(%2276, %2276) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2278 = "mix.prim.cos"(%2277) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2279 = "mix.prim.slice"(%2278) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2280 = "mix.prim.unsqueeze"(%2279) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2281 = "mix.prim.slice"(%2280) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2282 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2283 = "mix.prim.mul"(%2281, %2282) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2284 = "mix.prim.sin"(%2277) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2285 = "mix.prim.slice"(%2284) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2286 = "mix.prim.unsqueeze"(%2285) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2287 = "mix.prim.slice"(%2286) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2288 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2289 = "mix.prim.mul"(%2287, %2288) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2290 = "mix.prim.slice"(%2283) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2291 = "mix.prim.slice"(%2289) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2292 = "mix.prim.mul"(%2260, %2290) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2293 = "mix.prim.slice"(%2260) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2294 = "mix.prim.slice"(%2260) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2295 = "mix.prim.neg"(%2294) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2296 = "mix.prim.concat"(%2295, %2293) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2297 = "mix.prim.mul"(%2296, %2291) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2298 = "mix.prim.add"(%2292, %2297) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2299 = "mix.prim.mul"(%2261, %2290) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2300 = "mix.prim.slice"(%2261) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2301 = "mix.prim.slice"(%2261) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2302 = "mix.prim.neg"(%2301) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2303 = "mix.prim.concat"(%2302, %2300) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2304 = "mix.prim.mul"(%2303, %2291) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2305 = "mix.prim.add"(%2299, %2304) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2306 = "mix.prim.reshape"(%2298) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2307 = "mix.prim.reshape"(%2305) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2308 = "mix.prim.reshape"(%2306) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2309 = "mix.prim.reshape"(%2307) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2310 = "mix.prim.transpose"(%2308) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2311 = "mix.prim.transpose"(%2309) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2312 = "mix.prim.transpose"(%2311) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2313 = "mix.prim.unsqueeze"(%2310) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2314 = "mix.prim.permute"(%2313) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2315 = "mix.prim.unsqueeze"(%2312) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2316 = "mix.prim.permute"(%2315) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2317 = "mix.prim.permute"(%2314) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2318 = "mix.prim.reshape"(%2317) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2319 = "mix.prim.permute"(%2316) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2320 = "mix.prim.reshape"(%2319) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2321 = "mix.prim.batch_matmul"(%2318, %2320) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2322 = "mix.prim.reshape"(%2321) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2323 = "mix.prim.permute"(%2322) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2324 = "mix.prim.reshape"(%2323) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2325 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2326 = "mix.prim.mul"(%2324, %2325) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2327 = "mix.prim.reshape"(%2326) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2328 = "mix.comp.masked_fill"(%2327, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %2329 = "mix.comp.softmax"(%2328) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2330 = "mix.prim.reshape"(%2329) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2331 = "mix.prim.reshape"(%2259) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2332 = "mix.prim.transpose"(%2331) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2333 = "mix.prim.batch_matmul"(%2330, %2332) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2334 = "mix.prim.reshape"(%2333) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2335 = "mix.prim.permute"(%2334) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2336 = "mix.prim.reshape"(%2335) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2337 = "mix.prim.reshape"(%2336) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2338 = "mix.prim.transpose"(%2245) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2339 = "mix.prim.matmul"(%2337, %2338) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2340 = "mix.prim.add"(%2339, %2246) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2341 = "mix.prim.reshape"(%2340) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2342 = "mix.prim.mul"(%2233, %2341) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2343 = "mix.comp.weight"() <{param_loc = "transformer.h.18.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2344 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2345 = "mix.prim.pow"(%2342, %2344) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2346 = "mix.comp.mean"(%2345) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2347 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2348 = "mix.prim.add"(%2346, %2347) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2349 = "mix.prim.rsqrt"(%2348) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2350 = "mix.prim.mul"(%2342, %2349) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2351 = "mix.prim.mul"(%2343, %2350) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2352 = "mix.module.linear"(%2351) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.18.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2353 = "mix.comp.silu"(%2352) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2354 = "mix.module.linear"(%2351) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.18.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2355 = "mix.prim.mul"(%2353, %2354) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2356 = "mix.module.linear"(%2355) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.18.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2357 = "mix.prim.add"(%2356, %2342) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2358 = "mix.comp.weight"() <{param_loc = "transformer.h.19.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2359 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2360 = "mix.prim.pow"(%2357, %2359) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2361 = "mix.comp.mean"(%2360) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2362 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2363 = "mix.prim.add"(%2361, %2362) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2364 = "mix.prim.rsqrt"(%2363) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2365 = "mix.prim.mul"(%2357, %2364) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2366 = "mix.prim.mul"(%2358, %2365) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2367 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2368 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2369 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2370 = "mix.comp.weight"() <{param_loc = "transformer.h.19.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2371 = "mix.prim.transpose"(%2366) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2372 = "mix.prim.transpose"(%2367) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2373 = "mix.prim.reshape"(%2371) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2374 = "mix.prim.matmul"(%2373, %2372) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2375 = "mix.prim.reshape"(%2374) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2376 = "mix.prim.reshape"(%2375) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2377 = "mix.prim.transpose"(%2368) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2378 = "mix.prim.reshape"(%2371) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2379 = "mix.prim.matmul"(%2378, %2377) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2380 = "mix.prim.reshape"(%2379) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2381 = "mix.prim.reshape"(%2380) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2382 = "mix.prim.slice"(%2381) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2383 = "mix.prim.slice"(%2381) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2384 = "mix.prim.reshape"(%2376) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2385 = "mix.prim.reshape"(%2382) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2386 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2387 = "mix.prim.convert"(%2386) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2388 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2389 = "mix.prim.div"(%2387, %2388) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2390 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2391 = "mix.prim.pow"(%2390, %2389) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2392 = "mix.prim.reciprocal"(%2391) : (tensor<80xf16>) -> tensor<80xf16>
    %2393 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2394 = "mix.prim.mul"(%2393, %2392) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2395 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2396 = "mix.prim.unsqueeze"(%2395) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2397 = "mix.prim.permute"(%2396) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2398 = "mix.prim.unsqueeze"(%2394) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2399 = "mix.prim.permute"(%2398) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2400 = "mix.prim.mul"(%2397, %2399) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2401 = "mix.prim.concat"(%2400, %2400) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2402 = "mix.prim.cos"(%2401) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2403 = "mix.prim.slice"(%2402) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2404 = "mix.prim.unsqueeze"(%2403) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2405 = "mix.prim.slice"(%2404) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2406 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2407 = "mix.prim.mul"(%2405, %2406) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2408 = "mix.prim.sin"(%2401) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2409 = "mix.prim.slice"(%2408) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2410 = "mix.prim.unsqueeze"(%2409) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2411 = "mix.prim.slice"(%2410) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2412 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2413 = "mix.prim.mul"(%2411, %2412) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2414 = "mix.prim.slice"(%2407) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2415 = "mix.prim.slice"(%2413) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2416 = "mix.prim.mul"(%2384, %2414) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2417 = "mix.prim.slice"(%2384) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2418 = "mix.prim.slice"(%2384) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2419 = "mix.prim.neg"(%2418) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2420 = "mix.prim.concat"(%2419, %2417) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2421 = "mix.prim.mul"(%2420, %2415) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2422 = "mix.prim.add"(%2416, %2421) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2423 = "mix.prim.mul"(%2385, %2414) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2424 = "mix.prim.slice"(%2385) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2425 = "mix.prim.slice"(%2385) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2426 = "mix.prim.neg"(%2425) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2427 = "mix.prim.concat"(%2426, %2424) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2428 = "mix.prim.mul"(%2427, %2415) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2429 = "mix.prim.add"(%2423, %2428) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2430 = "mix.prim.reshape"(%2422) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2431 = "mix.prim.reshape"(%2429) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2432 = "mix.prim.reshape"(%2430) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2433 = "mix.prim.reshape"(%2431) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2434 = "mix.prim.transpose"(%2432) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2435 = "mix.prim.transpose"(%2433) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2436 = "mix.prim.transpose"(%2435) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2437 = "mix.prim.unsqueeze"(%2434) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2438 = "mix.prim.permute"(%2437) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2439 = "mix.prim.unsqueeze"(%2436) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2440 = "mix.prim.permute"(%2439) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2441 = "mix.prim.permute"(%2438) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2442 = "mix.prim.reshape"(%2441) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2443 = "mix.prim.permute"(%2440) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2444 = "mix.prim.reshape"(%2443) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2445 = "mix.prim.batch_matmul"(%2442, %2444) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2446 = "mix.prim.reshape"(%2445) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2447 = "mix.prim.permute"(%2446) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2448 = "mix.prim.reshape"(%2447) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2449 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2450 = "mix.prim.mul"(%2448, %2449) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2451 = "mix.prim.reshape"(%2450) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2452 = "mix.comp.masked_fill"(%2451, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %2453 = "mix.comp.softmax"(%2452) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2454 = "mix.prim.reshape"(%2453) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2455 = "mix.prim.reshape"(%2383) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2456 = "mix.prim.transpose"(%2455) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2457 = "mix.prim.batch_matmul"(%2454, %2456) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2458 = "mix.prim.reshape"(%2457) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2459 = "mix.prim.permute"(%2458) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2460 = "mix.prim.reshape"(%2459) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2461 = "mix.prim.reshape"(%2460) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2462 = "mix.prim.transpose"(%2369) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2463 = "mix.prim.matmul"(%2461, %2462) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2464 = "mix.prim.add"(%2463, %2370) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2465 = "mix.prim.reshape"(%2464) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2466 = "mix.prim.mul"(%2357, %2465) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2467 = "mix.comp.weight"() <{param_loc = "transformer.h.19.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2468 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2469 = "mix.prim.pow"(%2466, %2468) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2470 = "mix.comp.mean"(%2469) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2471 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2472 = "mix.prim.add"(%2470, %2471) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2473 = "mix.prim.rsqrt"(%2472) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2474 = "mix.prim.mul"(%2466, %2473) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2475 = "mix.prim.mul"(%2467, %2474) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2476 = "mix.module.linear"(%2475) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.19.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2477 = "mix.comp.silu"(%2476) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2478 = "mix.module.linear"(%2475) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.19.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2479 = "mix.prim.mul"(%2477, %2478) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2480 = "mix.module.linear"(%2479) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.19.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2481 = "mix.prim.add"(%2480, %2466) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2482 = "mix.comp.weight"() <{param_loc = "transformer.h.20.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2483 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2484 = "mix.prim.pow"(%2481, %2483) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2485 = "mix.comp.mean"(%2484) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2486 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2487 = "mix.prim.add"(%2485, %2486) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2488 = "mix.prim.rsqrt"(%2487) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2489 = "mix.prim.mul"(%2481, %2488) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2490 = "mix.prim.mul"(%2482, %2489) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2491 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2492 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2493 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2494 = "mix.comp.weight"() <{param_loc = "transformer.h.20.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2495 = "mix.prim.transpose"(%2490) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2496 = "mix.prim.transpose"(%2491) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2497 = "mix.prim.reshape"(%2495) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2498 = "mix.prim.matmul"(%2497, %2496) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2499 = "mix.prim.reshape"(%2498) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2500 = "mix.prim.reshape"(%2499) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2501 = "mix.prim.transpose"(%2492) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2502 = "mix.prim.reshape"(%2495) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2503 = "mix.prim.matmul"(%2502, %2501) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2504 = "mix.prim.reshape"(%2503) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2505 = "mix.prim.reshape"(%2504) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2506 = "mix.prim.slice"(%2505) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2507 = "mix.prim.slice"(%2505) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2508 = "mix.prim.reshape"(%2500) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2509 = "mix.prim.reshape"(%2506) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2510 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2511 = "mix.prim.convert"(%2510) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2512 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2513 = "mix.prim.div"(%2511, %2512) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2514 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2515 = "mix.prim.pow"(%2514, %2513) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2516 = "mix.prim.reciprocal"(%2515) : (tensor<80xf16>) -> tensor<80xf16>
    %2517 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2518 = "mix.prim.mul"(%2517, %2516) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2519 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2520 = "mix.prim.unsqueeze"(%2519) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2521 = "mix.prim.permute"(%2520) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2522 = "mix.prim.unsqueeze"(%2518) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2523 = "mix.prim.permute"(%2522) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2524 = "mix.prim.mul"(%2521, %2523) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2525 = "mix.prim.concat"(%2524, %2524) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2526 = "mix.prim.cos"(%2525) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2527 = "mix.prim.slice"(%2526) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2528 = "mix.prim.unsqueeze"(%2527) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2529 = "mix.prim.slice"(%2528) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2530 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2531 = "mix.prim.mul"(%2529, %2530) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2532 = "mix.prim.sin"(%2525) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2533 = "mix.prim.slice"(%2532) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2534 = "mix.prim.unsqueeze"(%2533) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2535 = "mix.prim.slice"(%2534) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2536 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2537 = "mix.prim.mul"(%2535, %2536) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2538 = "mix.prim.slice"(%2531) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2539 = "mix.prim.slice"(%2537) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2540 = "mix.prim.mul"(%2508, %2538) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2541 = "mix.prim.slice"(%2508) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2542 = "mix.prim.slice"(%2508) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2543 = "mix.prim.neg"(%2542) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2544 = "mix.prim.concat"(%2543, %2541) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2545 = "mix.prim.mul"(%2544, %2539) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2546 = "mix.prim.add"(%2540, %2545) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2547 = "mix.prim.mul"(%2509, %2538) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2548 = "mix.prim.slice"(%2509) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2549 = "mix.prim.slice"(%2509) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2550 = "mix.prim.neg"(%2549) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2551 = "mix.prim.concat"(%2550, %2548) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2552 = "mix.prim.mul"(%2551, %2539) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2553 = "mix.prim.add"(%2547, %2552) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2554 = "mix.prim.reshape"(%2546) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2555 = "mix.prim.reshape"(%2553) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2556 = "mix.prim.reshape"(%2554) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2557 = "mix.prim.reshape"(%2555) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2558 = "mix.prim.transpose"(%2556) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2559 = "mix.prim.transpose"(%2557) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2560 = "mix.prim.transpose"(%2559) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2561 = "mix.prim.unsqueeze"(%2558) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2562 = "mix.prim.permute"(%2561) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2563 = "mix.prim.unsqueeze"(%2560) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2564 = "mix.prim.permute"(%2563) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2565 = "mix.prim.permute"(%2562) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2566 = "mix.prim.reshape"(%2565) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2567 = "mix.prim.permute"(%2564) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2568 = "mix.prim.reshape"(%2567) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2569 = "mix.prim.batch_matmul"(%2566, %2568) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2570 = "mix.prim.reshape"(%2569) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2571 = "mix.prim.permute"(%2570) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2572 = "mix.prim.reshape"(%2571) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2573 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2574 = "mix.prim.mul"(%2572, %2573) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2575 = "mix.prim.reshape"(%2574) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2576 = "mix.comp.masked_fill"(%2575, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %2577 = "mix.comp.softmax"(%2576) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2578 = "mix.prim.reshape"(%2577) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2579 = "mix.prim.reshape"(%2507) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2580 = "mix.prim.transpose"(%2579) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2581 = "mix.prim.batch_matmul"(%2578, %2580) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2582 = "mix.prim.reshape"(%2581) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2583 = "mix.prim.permute"(%2582) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2584 = "mix.prim.reshape"(%2583) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2585 = "mix.prim.reshape"(%2584) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2586 = "mix.prim.transpose"(%2493) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2587 = "mix.prim.matmul"(%2585, %2586) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2588 = "mix.prim.add"(%2587, %2494) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2589 = "mix.prim.reshape"(%2588) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2590 = "mix.prim.mul"(%2481, %2589) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2591 = "mix.comp.weight"() <{param_loc = "transformer.h.20.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2592 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2593 = "mix.prim.pow"(%2590, %2592) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2594 = "mix.comp.mean"(%2593) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2595 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2596 = "mix.prim.add"(%2594, %2595) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2597 = "mix.prim.rsqrt"(%2596) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2598 = "mix.prim.mul"(%2590, %2597) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2599 = "mix.prim.mul"(%2591, %2598) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2600 = "mix.module.linear"(%2599) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.20.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2601 = "mix.comp.silu"(%2600) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2602 = "mix.module.linear"(%2599) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.20.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2603 = "mix.prim.mul"(%2601, %2602) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2604 = "mix.module.linear"(%2603) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.20.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2605 = "mix.prim.add"(%2604, %2590) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2606 = "mix.comp.weight"() <{param_loc = "transformer.h.21.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2607 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2608 = "mix.prim.pow"(%2605, %2607) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2609 = "mix.comp.mean"(%2608) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2610 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2611 = "mix.prim.add"(%2609, %2610) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2612 = "mix.prim.rsqrt"(%2611) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2613 = "mix.prim.mul"(%2605, %2612) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2614 = "mix.prim.mul"(%2606, %2613) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2615 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2616 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2617 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2618 = "mix.comp.weight"() <{param_loc = "transformer.h.21.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2619 = "mix.prim.transpose"(%2614) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2620 = "mix.prim.transpose"(%2615) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2621 = "mix.prim.reshape"(%2619) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2622 = "mix.prim.matmul"(%2621, %2620) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2623 = "mix.prim.reshape"(%2622) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2624 = "mix.prim.reshape"(%2623) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2625 = "mix.prim.transpose"(%2616) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2626 = "mix.prim.reshape"(%2619) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2627 = "mix.prim.matmul"(%2626, %2625) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2628 = "mix.prim.reshape"(%2627) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2629 = "mix.prim.reshape"(%2628) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2630 = "mix.prim.slice"(%2629) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2631 = "mix.prim.slice"(%2629) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2632 = "mix.prim.reshape"(%2624) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2633 = "mix.prim.reshape"(%2630) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2634 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2635 = "mix.prim.convert"(%2634) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2636 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2637 = "mix.prim.div"(%2635, %2636) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2638 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2639 = "mix.prim.pow"(%2638, %2637) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2640 = "mix.prim.reciprocal"(%2639) : (tensor<80xf16>) -> tensor<80xf16>
    %2641 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2642 = "mix.prim.mul"(%2641, %2640) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2643 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2644 = "mix.prim.unsqueeze"(%2643) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2645 = "mix.prim.permute"(%2644) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2646 = "mix.prim.unsqueeze"(%2642) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2647 = "mix.prim.permute"(%2646) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2648 = "mix.prim.mul"(%2645, %2647) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2649 = "mix.prim.concat"(%2648, %2648) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2650 = "mix.prim.cos"(%2649) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2651 = "mix.prim.slice"(%2650) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2652 = "mix.prim.unsqueeze"(%2651) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2653 = "mix.prim.slice"(%2652) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2654 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2655 = "mix.prim.mul"(%2653, %2654) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2656 = "mix.prim.sin"(%2649) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2657 = "mix.prim.slice"(%2656) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2658 = "mix.prim.unsqueeze"(%2657) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2659 = "mix.prim.slice"(%2658) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2660 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2661 = "mix.prim.mul"(%2659, %2660) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2662 = "mix.prim.slice"(%2655) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2663 = "mix.prim.slice"(%2661) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2664 = "mix.prim.mul"(%2632, %2662) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2665 = "mix.prim.slice"(%2632) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2666 = "mix.prim.slice"(%2632) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2667 = "mix.prim.neg"(%2666) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2668 = "mix.prim.concat"(%2667, %2665) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2669 = "mix.prim.mul"(%2668, %2663) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2670 = "mix.prim.add"(%2664, %2669) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2671 = "mix.prim.mul"(%2633, %2662) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2672 = "mix.prim.slice"(%2633) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2673 = "mix.prim.slice"(%2633) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2674 = "mix.prim.neg"(%2673) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2675 = "mix.prim.concat"(%2674, %2672) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2676 = "mix.prim.mul"(%2675, %2663) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2677 = "mix.prim.add"(%2671, %2676) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2678 = "mix.prim.reshape"(%2670) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2679 = "mix.prim.reshape"(%2677) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2680 = "mix.prim.reshape"(%2678) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2681 = "mix.prim.reshape"(%2679) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2682 = "mix.prim.transpose"(%2680) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2683 = "mix.prim.transpose"(%2681) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2684 = "mix.prim.transpose"(%2683) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2685 = "mix.prim.unsqueeze"(%2682) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2686 = "mix.prim.permute"(%2685) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2687 = "mix.prim.unsqueeze"(%2684) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2688 = "mix.prim.permute"(%2687) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2689 = "mix.prim.permute"(%2686) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2690 = "mix.prim.reshape"(%2689) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2691 = "mix.prim.permute"(%2688) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2692 = "mix.prim.reshape"(%2691) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2693 = "mix.prim.batch_matmul"(%2690, %2692) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2694 = "mix.prim.reshape"(%2693) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2695 = "mix.prim.permute"(%2694) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2696 = "mix.prim.reshape"(%2695) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2697 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2698 = "mix.prim.mul"(%2696, %2697) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2699 = "mix.prim.reshape"(%2698) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2700 = "mix.comp.masked_fill"(%2699, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %2701 = "mix.comp.softmax"(%2700) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2702 = "mix.prim.reshape"(%2701) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2703 = "mix.prim.reshape"(%2631) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2704 = "mix.prim.transpose"(%2703) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2705 = "mix.prim.batch_matmul"(%2702, %2704) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2706 = "mix.prim.reshape"(%2705) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2707 = "mix.prim.permute"(%2706) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2708 = "mix.prim.reshape"(%2707) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2709 = "mix.prim.reshape"(%2708) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2710 = "mix.prim.transpose"(%2617) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2711 = "mix.prim.matmul"(%2709, %2710) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2712 = "mix.prim.add"(%2711, %2618) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2713 = "mix.prim.reshape"(%2712) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2714 = "mix.prim.mul"(%2605, %2713) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2715 = "mix.comp.weight"() <{param_loc = "transformer.h.21.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2716 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2717 = "mix.prim.pow"(%2714, %2716) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2718 = "mix.comp.mean"(%2717) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2719 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2720 = "mix.prim.add"(%2718, %2719) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2721 = "mix.prim.rsqrt"(%2720) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2722 = "mix.prim.mul"(%2714, %2721) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2723 = "mix.prim.mul"(%2715, %2722) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2724 = "mix.module.linear"(%2723) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.21.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2725 = "mix.comp.silu"(%2724) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2726 = "mix.module.linear"(%2723) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.21.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2727 = "mix.prim.mul"(%2725, %2726) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2728 = "mix.module.linear"(%2727) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.21.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2729 = "mix.prim.add"(%2728, %2714) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2730 = "mix.comp.weight"() <{param_loc = "transformer.h.22.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2731 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2732 = "mix.prim.pow"(%2729, %2731) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2733 = "mix.comp.mean"(%2732) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2734 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2735 = "mix.prim.add"(%2733, %2734) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2736 = "mix.prim.rsqrt"(%2735) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2737 = "mix.prim.mul"(%2729, %2736) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2738 = "mix.prim.mul"(%2730, %2737) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2739 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2740 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2741 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2742 = "mix.comp.weight"() <{param_loc = "transformer.h.22.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2743 = "mix.prim.transpose"(%2738) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2744 = "mix.prim.transpose"(%2739) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2745 = "mix.prim.reshape"(%2743) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2746 = "mix.prim.matmul"(%2745, %2744) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2747 = "mix.prim.reshape"(%2746) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2748 = "mix.prim.reshape"(%2747) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2749 = "mix.prim.transpose"(%2740) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2750 = "mix.prim.reshape"(%2743) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2751 = "mix.prim.matmul"(%2750, %2749) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2752 = "mix.prim.reshape"(%2751) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2753 = "mix.prim.reshape"(%2752) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2754 = "mix.prim.slice"(%2753) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2755 = "mix.prim.slice"(%2753) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2756 = "mix.prim.reshape"(%2748) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2757 = "mix.prim.reshape"(%2754) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2758 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2759 = "mix.prim.convert"(%2758) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2760 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2761 = "mix.prim.div"(%2759, %2760) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2762 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2763 = "mix.prim.pow"(%2762, %2761) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2764 = "mix.prim.reciprocal"(%2763) : (tensor<80xf16>) -> tensor<80xf16>
    %2765 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2766 = "mix.prim.mul"(%2765, %2764) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2767 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2768 = "mix.prim.unsqueeze"(%2767) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2769 = "mix.prim.permute"(%2768) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2770 = "mix.prim.unsqueeze"(%2766) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2771 = "mix.prim.permute"(%2770) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2772 = "mix.prim.mul"(%2769, %2771) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2773 = "mix.prim.concat"(%2772, %2772) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2774 = "mix.prim.cos"(%2773) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2775 = "mix.prim.slice"(%2774) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2776 = "mix.prim.unsqueeze"(%2775) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2777 = "mix.prim.slice"(%2776) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2778 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2779 = "mix.prim.mul"(%2777, %2778) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2780 = "mix.prim.sin"(%2773) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2781 = "mix.prim.slice"(%2780) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2782 = "mix.prim.unsqueeze"(%2781) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2783 = "mix.prim.slice"(%2782) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2784 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2785 = "mix.prim.mul"(%2783, %2784) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2786 = "mix.prim.slice"(%2779) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2787 = "mix.prim.slice"(%2785) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2788 = "mix.prim.mul"(%2756, %2786) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2789 = "mix.prim.slice"(%2756) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2790 = "mix.prim.slice"(%2756) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2791 = "mix.prim.neg"(%2790) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2792 = "mix.prim.concat"(%2791, %2789) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2793 = "mix.prim.mul"(%2792, %2787) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2794 = "mix.prim.add"(%2788, %2793) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2795 = "mix.prim.mul"(%2757, %2786) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2796 = "mix.prim.slice"(%2757) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2797 = "mix.prim.slice"(%2757) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2798 = "mix.prim.neg"(%2797) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2799 = "mix.prim.concat"(%2798, %2796) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2800 = "mix.prim.mul"(%2799, %2787) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2801 = "mix.prim.add"(%2795, %2800) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2802 = "mix.prim.reshape"(%2794) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2803 = "mix.prim.reshape"(%2801) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2804 = "mix.prim.reshape"(%2802) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2805 = "mix.prim.reshape"(%2803) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2806 = "mix.prim.transpose"(%2804) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2807 = "mix.prim.transpose"(%2805) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2808 = "mix.prim.transpose"(%2807) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2809 = "mix.prim.unsqueeze"(%2806) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2810 = "mix.prim.permute"(%2809) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2811 = "mix.prim.unsqueeze"(%2808) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2812 = "mix.prim.permute"(%2811) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2813 = "mix.prim.permute"(%2810) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2814 = "mix.prim.reshape"(%2813) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2815 = "mix.prim.permute"(%2812) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2816 = "mix.prim.reshape"(%2815) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2817 = "mix.prim.batch_matmul"(%2814, %2816) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2818 = "mix.prim.reshape"(%2817) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2819 = "mix.prim.permute"(%2818) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2820 = "mix.prim.reshape"(%2819) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2821 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2822 = "mix.prim.mul"(%2820, %2821) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2823 = "mix.prim.reshape"(%2822) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2824 = "mix.comp.masked_fill"(%2823, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %2825 = "mix.comp.softmax"(%2824) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2826 = "mix.prim.reshape"(%2825) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2827 = "mix.prim.reshape"(%2755) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2828 = "mix.prim.transpose"(%2827) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2829 = "mix.prim.batch_matmul"(%2826, %2828) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2830 = "mix.prim.reshape"(%2829) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2831 = "mix.prim.permute"(%2830) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2832 = "mix.prim.reshape"(%2831) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2833 = "mix.prim.reshape"(%2832) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2834 = "mix.prim.transpose"(%2741) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2835 = "mix.prim.matmul"(%2833, %2834) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2836 = "mix.prim.add"(%2835, %2742) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2837 = "mix.prim.reshape"(%2836) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2838 = "mix.prim.mul"(%2729, %2837) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2839 = "mix.comp.weight"() <{param_loc = "transformer.h.22.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2840 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2841 = "mix.prim.pow"(%2838, %2840) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2842 = "mix.comp.mean"(%2841) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2843 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2844 = "mix.prim.add"(%2842, %2843) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2845 = "mix.prim.rsqrt"(%2844) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2846 = "mix.prim.mul"(%2838, %2845) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2847 = "mix.prim.mul"(%2839, %2846) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2848 = "mix.module.linear"(%2847) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.22.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2849 = "mix.comp.silu"(%2848) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2850 = "mix.module.linear"(%2847) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.22.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2851 = "mix.prim.mul"(%2849, %2850) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2852 = "mix.module.linear"(%2851) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.22.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2853 = "mix.prim.add"(%2852, %2838) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2854 = "mix.comp.weight"() <{param_loc = "transformer.h.23.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2855 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2856 = "mix.prim.pow"(%2853, %2855) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2857 = "mix.comp.mean"(%2856) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2858 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2859 = "mix.prim.add"(%2857, %2858) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2860 = "mix.prim.rsqrt"(%2859) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2861 = "mix.prim.mul"(%2853, %2860) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2862 = "mix.prim.mul"(%2854, %2861) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2863 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2864 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2865 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2866 = "mix.comp.weight"() <{param_loc = "transformer.h.23.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2867 = "mix.prim.transpose"(%2862) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2868 = "mix.prim.transpose"(%2863) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2869 = "mix.prim.reshape"(%2867) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2870 = "mix.prim.matmul"(%2869, %2868) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2871 = "mix.prim.reshape"(%2870) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2872 = "mix.prim.reshape"(%2871) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2873 = "mix.prim.transpose"(%2864) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2874 = "mix.prim.reshape"(%2867) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2875 = "mix.prim.matmul"(%2874, %2873) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %2876 = "mix.prim.reshape"(%2875) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %2877 = "mix.prim.reshape"(%2876) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %2878 = "mix.prim.slice"(%2877) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2879 = "mix.prim.slice"(%2877) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %2880 = "mix.prim.reshape"(%2872) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2881 = "mix.prim.reshape"(%2878) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2882 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %2883 = "mix.prim.convert"(%2882) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %2884 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %2885 = "mix.prim.div"(%2883, %2884) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %2886 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %2887 = "mix.prim.pow"(%2886, %2885) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2888 = "mix.prim.reciprocal"(%2887) : (tensor<80xf16>) -> tensor<80xf16>
    %2889 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2890 = "mix.prim.mul"(%2889, %2888) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %2891 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %2892 = "mix.prim.unsqueeze"(%2891) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %2893 = "mix.prim.permute"(%2892) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %2894 = "mix.prim.unsqueeze"(%2890) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %2895 = "mix.prim.permute"(%2894) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %2896 = "mix.prim.mul"(%2893, %2895) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %2897 = "mix.prim.concat"(%2896, %2896) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %2898 = "mix.prim.cos"(%2897) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2899 = "mix.prim.slice"(%2898) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2900 = "mix.prim.unsqueeze"(%2899) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2901 = "mix.prim.slice"(%2900) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2902 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2903 = "mix.prim.mul"(%2901, %2902) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2904 = "mix.prim.sin"(%2897) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2905 = "mix.prim.slice"(%2904) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %2906 = "mix.prim.unsqueeze"(%2905) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %2907 = "mix.prim.slice"(%2906) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2908 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %2909 = "mix.prim.mul"(%2907, %2908) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %2910 = "mix.prim.slice"(%2903) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2911 = "mix.prim.slice"(%2909) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %2912 = "mix.prim.mul"(%2880, %2910) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2913 = "mix.prim.slice"(%2880) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2914 = "mix.prim.slice"(%2880) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2915 = "mix.prim.neg"(%2914) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2916 = "mix.prim.concat"(%2915, %2913) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2917 = "mix.prim.mul"(%2916, %2911) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2918 = "mix.prim.add"(%2912, %2917) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2919 = "mix.prim.mul"(%2881, %2910) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2920 = "mix.prim.slice"(%2881) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2921 = "mix.prim.slice"(%2881) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %2922 = "mix.prim.neg"(%2921) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %2923 = "mix.prim.concat"(%2922, %2920) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %2924 = "mix.prim.mul"(%2923, %2911) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %2925 = "mix.prim.add"(%2919, %2924) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %2926 = "mix.prim.reshape"(%2918) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2927 = "mix.prim.reshape"(%2925) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %2928 = "mix.prim.reshape"(%2926) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2929 = "mix.prim.reshape"(%2927) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2930 = "mix.prim.transpose"(%2928) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2931 = "mix.prim.transpose"(%2929) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2932 = "mix.prim.transpose"(%2931) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %2933 = "mix.prim.unsqueeze"(%2930) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %2934 = "mix.prim.permute"(%2933) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %2935 = "mix.prim.unsqueeze"(%2932) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %2936 = "mix.prim.permute"(%2935) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %2937 = "mix.prim.permute"(%2934) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %2938 = "mix.prim.reshape"(%2937) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %2939 = "mix.prim.permute"(%2936) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %2940 = "mix.prim.reshape"(%2939) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %2941 = "mix.prim.batch_matmul"(%2938, %2940) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %2942 = "mix.prim.reshape"(%2941) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %2943 = "mix.prim.permute"(%2942) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %2944 = "mix.prim.reshape"(%2943) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %2945 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %2946 = "mix.prim.mul"(%2944, %2945) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %2947 = "mix.prim.reshape"(%2946) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2948 = "mix.comp.masked_fill"(%2947, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %2949 = "mix.comp.softmax"(%2948) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %2950 = "mix.prim.reshape"(%2949) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %2951 = "mix.prim.reshape"(%2879) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %2952 = "mix.prim.transpose"(%2951) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %2953 = "mix.prim.batch_matmul"(%2950, %2952) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %2954 = "mix.prim.reshape"(%2953) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %2955 = "mix.prim.permute"(%2954) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %2956 = "mix.prim.reshape"(%2955) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %2957 = "mix.prim.reshape"(%2956) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %2958 = "mix.prim.transpose"(%2865) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2959 = "mix.prim.matmul"(%2957, %2958) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2960 = "mix.prim.add"(%2959, %2866) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %2961 = "mix.prim.reshape"(%2960) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %2962 = "mix.prim.mul"(%2853, %2961) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2963 = "mix.comp.weight"() <{param_loc = "transformer.h.23.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %2964 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2965 = "mix.prim.pow"(%2962, %2964) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2966 = "mix.comp.mean"(%2965) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2967 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2968 = "mix.prim.add"(%2966, %2967) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2969 = "mix.prim.rsqrt"(%2968) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2970 = "mix.prim.mul"(%2962, %2969) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2971 = "mix.prim.mul"(%2963, %2970) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2972 = "mix.module.linear"(%2971) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.23.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2973 = "mix.comp.silu"(%2972) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2974 = "mix.module.linear"(%2971) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.23.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %2975 = "mix.prim.mul"(%2973, %2974) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %2976 = "mix.module.linear"(%2975) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.23.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %2977 = "mix.prim.add"(%2976, %2962) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2978 = "mix.comp.weight"() <{param_loc = "transformer.h.24.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %2979 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %2980 = "mix.prim.pow"(%2977, %2979) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %2981 = "mix.comp.mean"(%2980) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %2982 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %2983 = "mix.prim.add"(%2981, %2982) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %2984 = "mix.prim.rsqrt"(%2983) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %2985 = "mix.prim.mul"(%2977, %2984) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %2986 = "mix.prim.mul"(%2978, %2985) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %2987 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %2988 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %2989 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %2990 = "mix.comp.weight"() <{param_loc = "transformer.h.24.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %2991 = "mix.prim.transpose"(%2986) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %2992 = "mix.prim.transpose"(%2987) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %2993 = "mix.prim.reshape"(%2991) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2994 = "mix.prim.matmul"(%2993, %2992) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %2995 = "mix.prim.reshape"(%2994) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %2996 = "mix.prim.reshape"(%2995) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %2997 = "mix.prim.transpose"(%2988) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %2998 = "mix.prim.reshape"(%2991) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %2999 = "mix.prim.matmul"(%2998, %2997) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3000 = "mix.prim.reshape"(%2999) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3001 = "mix.prim.reshape"(%3000) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3002 = "mix.prim.slice"(%3001) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3003 = "mix.prim.slice"(%3001) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3004 = "mix.prim.reshape"(%2996) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3005 = "mix.prim.reshape"(%3002) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3006 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3007 = "mix.prim.convert"(%3006) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3008 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3009 = "mix.prim.div"(%3007, %3008) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3010 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3011 = "mix.prim.pow"(%3010, %3009) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3012 = "mix.prim.reciprocal"(%3011) : (tensor<80xf16>) -> tensor<80xf16>
    %3013 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3014 = "mix.prim.mul"(%3013, %3012) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3015 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3016 = "mix.prim.unsqueeze"(%3015) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3017 = "mix.prim.permute"(%3016) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3018 = "mix.prim.unsqueeze"(%3014) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3019 = "mix.prim.permute"(%3018) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3020 = "mix.prim.mul"(%3017, %3019) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3021 = "mix.prim.concat"(%3020, %3020) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3022 = "mix.prim.cos"(%3021) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3023 = "mix.prim.slice"(%3022) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3024 = "mix.prim.unsqueeze"(%3023) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3025 = "mix.prim.slice"(%3024) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3026 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3027 = "mix.prim.mul"(%3025, %3026) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3028 = "mix.prim.sin"(%3021) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3029 = "mix.prim.slice"(%3028) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3030 = "mix.prim.unsqueeze"(%3029) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3031 = "mix.prim.slice"(%3030) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3032 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3033 = "mix.prim.mul"(%3031, %3032) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3034 = "mix.prim.slice"(%3027) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3035 = "mix.prim.slice"(%3033) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3036 = "mix.prim.mul"(%3004, %3034) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3037 = "mix.prim.slice"(%3004) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3038 = "mix.prim.slice"(%3004) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3039 = "mix.prim.neg"(%3038) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3040 = "mix.prim.concat"(%3039, %3037) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3041 = "mix.prim.mul"(%3040, %3035) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3042 = "mix.prim.add"(%3036, %3041) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3043 = "mix.prim.mul"(%3005, %3034) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3044 = "mix.prim.slice"(%3005) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3045 = "mix.prim.slice"(%3005) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3046 = "mix.prim.neg"(%3045) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3047 = "mix.prim.concat"(%3046, %3044) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3048 = "mix.prim.mul"(%3047, %3035) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3049 = "mix.prim.add"(%3043, %3048) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3050 = "mix.prim.reshape"(%3042) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3051 = "mix.prim.reshape"(%3049) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3052 = "mix.prim.reshape"(%3050) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3053 = "mix.prim.reshape"(%3051) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3054 = "mix.prim.transpose"(%3052) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3055 = "mix.prim.transpose"(%3053) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3056 = "mix.prim.transpose"(%3055) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3057 = "mix.prim.unsqueeze"(%3054) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3058 = "mix.prim.permute"(%3057) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3059 = "mix.prim.unsqueeze"(%3056) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3060 = "mix.prim.permute"(%3059) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3061 = "mix.prim.permute"(%3058) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3062 = "mix.prim.reshape"(%3061) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3063 = "mix.prim.permute"(%3060) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3064 = "mix.prim.reshape"(%3063) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3065 = "mix.prim.batch_matmul"(%3062, %3064) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3066 = "mix.prim.reshape"(%3065) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3067 = "mix.prim.permute"(%3066) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3068 = "mix.prim.reshape"(%3067) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3069 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3070 = "mix.prim.mul"(%3068, %3069) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3071 = "mix.prim.reshape"(%3070) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3072 = "mix.comp.masked_fill"(%3071, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %3073 = "mix.comp.softmax"(%3072) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3074 = "mix.prim.reshape"(%3073) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3075 = "mix.prim.reshape"(%3003) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3076 = "mix.prim.transpose"(%3075) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3077 = "mix.prim.batch_matmul"(%3074, %3076) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3078 = "mix.prim.reshape"(%3077) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3079 = "mix.prim.permute"(%3078) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3080 = "mix.prim.reshape"(%3079) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3081 = "mix.prim.reshape"(%3080) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3082 = "mix.prim.transpose"(%2989) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3083 = "mix.prim.matmul"(%3081, %3082) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3084 = "mix.prim.add"(%3083, %2990) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3085 = "mix.prim.reshape"(%3084) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3086 = "mix.prim.mul"(%2977, %3085) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3087 = "mix.comp.weight"() <{param_loc = "transformer.h.24.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3088 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3089 = "mix.prim.pow"(%3086, %3088) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3090 = "mix.comp.mean"(%3089) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3091 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3092 = "mix.prim.add"(%3090, %3091) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3093 = "mix.prim.rsqrt"(%3092) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3094 = "mix.prim.mul"(%3086, %3093) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3095 = "mix.prim.mul"(%3087, %3094) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3096 = "mix.module.linear"(%3095) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.24.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3097 = "mix.comp.silu"(%3096) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3098 = "mix.module.linear"(%3095) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.24.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3099 = "mix.prim.mul"(%3097, %3098) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3100 = "mix.module.linear"(%3099) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.24.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3101 = "mix.prim.add"(%3100, %3086) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3102 = "mix.comp.weight"() <{param_loc = "transformer.h.25.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3103 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3104 = "mix.prim.pow"(%3101, %3103) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3105 = "mix.comp.mean"(%3104) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3106 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3107 = "mix.prim.add"(%3105, %3106) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3108 = "mix.prim.rsqrt"(%3107) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3109 = "mix.prim.mul"(%3101, %3108) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3110 = "mix.prim.mul"(%3102, %3109) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3111 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3112 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3113 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3114 = "mix.comp.weight"() <{param_loc = "transformer.h.25.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3115 = "mix.prim.transpose"(%3110) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3116 = "mix.prim.transpose"(%3111) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3117 = "mix.prim.reshape"(%3115) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3118 = "mix.prim.matmul"(%3117, %3116) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3119 = "mix.prim.reshape"(%3118) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3120 = "mix.prim.reshape"(%3119) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3121 = "mix.prim.transpose"(%3112) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3122 = "mix.prim.reshape"(%3115) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3123 = "mix.prim.matmul"(%3122, %3121) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3124 = "mix.prim.reshape"(%3123) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3125 = "mix.prim.reshape"(%3124) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3126 = "mix.prim.slice"(%3125) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3127 = "mix.prim.slice"(%3125) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3128 = "mix.prim.reshape"(%3120) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3129 = "mix.prim.reshape"(%3126) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3130 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3131 = "mix.prim.convert"(%3130) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3132 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3133 = "mix.prim.div"(%3131, %3132) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3134 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3135 = "mix.prim.pow"(%3134, %3133) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3136 = "mix.prim.reciprocal"(%3135) : (tensor<80xf16>) -> tensor<80xf16>
    %3137 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3138 = "mix.prim.mul"(%3137, %3136) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3139 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3140 = "mix.prim.unsqueeze"(%3139) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3141 = "mix.prim.permute"(%3140) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3142 = "mix.prim.unsqueeze"(%3138) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3143 = "mix.prim.permute"(%3142) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3144 = "mix.prim.mul"(%3141, %3143) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3145 = "mix.prim.concat"(%3144, %3144) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3146 = "mix.prim.cos"(%3145) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3147 = "mix.prim.slice"(%3146) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3148 = "mix.prim.unsqueeze"(%3147) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3149 = "mix.prim.slice"(%3148) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3150 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3151 = "mix.prim.mul"(%3149, %3150) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3152 = "mix.prim.sin"(%3145) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3153 = "mix.prim.slice"(%3152) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3154 = "mix.prim.unsqueeze"(%3153) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3155 = "mix.prim.slice"(%3154) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3156 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3157 = "mix.prim.mul"(%3155, %3156) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3158 = "mix.prim.slice"(%3151) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3159 = "mix.prim.slice"(%3157) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3160 = "mix.prim.mul"(%3128, %3158) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3161 = "mix.prim.slice"(%3128) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3162 = "mix.prim.slice"(%3128) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3163 = "mix.prim.neg"(%3162) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3164 = "mix.prim.concat"(%3163, %3161) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3165 = "mix.prim.mul"(%3164, %3159) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3166 = "mix.prim.add"(%3160, %3165) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3167 = "mix.prim.mul"(%3129, %3158) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3168 = "mix.prim.slice"(%3129) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3169 = "mix.prim.slice"(%3129) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3170 = "mix.prim.neg"(%3169) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3171 = "mix.prim.concat"(%3170, %3168) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3172 = "mix.prim.mul"(%3171, %3159) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3173 = "mix.prim.add"(%3167, %3172) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3174 = "mix.prim.reshape"(%3166) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3175 = "mix.prim.reshape"(%3173) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3176 = "mix.prim.reshape"(%3174) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3177 = "mix.prim.reshape"(%3175) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3178 = "mix.prim.transpose"(%3176) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3179 = "mix.prim.transpose"(%3177) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3180 = "mix.prim.transpose"(%3179) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3181 = "mix.prim.unsqueeze"(%3178) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3182 = "mix.prim.permute"(%3181) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3183 = "mix.prim.unsqueeze"(%3180) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3184 = "mix.prim.permute"(%3183) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3185 = "mix.prim.permute"(%3182) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3186 = "mix.prim.reshape"(%3185) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3187 = "mix.prim.permute"(%3184) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3188 = "mix.prim.reshape"(%3187) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3189 = "mix.prim.batch_matmul"(%3186, %3188) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3190 = "mix.prim.reshape"(%3189) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3191 = "mix.prim.permute"(%3190) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3192 = "mix.prim.reshape"(%3191) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3193 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3194 = "mix.prim.mul"(%3192, %3193) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3195 = "mix.prim.reshape"(%3194) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3196 = "mix.comp.masked_fill"(%3195, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %3197 = "mix.comp.softmax"(%3196) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3198 = "mix.prim.reshape"(%3197) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3199 = "mix.prim.reshape"(%3127) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3200 = "mix.prim.transpose"(%3199) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3201 = "mix.prim.batch_matmul"(%3198, %3200) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3202 = "mix.prim.reshape"(%3201) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3203 = "mix.prim.permute"(%3202) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3204 = "mix.prim.reshape"(%3203) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3205 = "mix.prim.reshape"(%3204) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3206 = "mix.prim.transpose"(%3113) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3207 = "mix.prim.matmul"(%3205, %3206) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3208 = "mix.prim.add"(%3207, %3114) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3209 = "mix.prim.reshape"(%3208) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3210 = "mix.prim.mul"(%3101, %3209) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3211 = "mix.comp.weight"() <{param_loc = "transformer.h.25.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3212 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3213 = "mix.prim.pow"(%3210, %3212) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3214 = "mix.comp.mean"(%3213) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3215 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3216 = "mix.prim.add"(%3214, %3215) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3217 = "mix.prim.rsqrt"(%3216) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3218 = "mix.prim.mul"(%3210, %3217) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3219 = "mix.prim.mul"(%3211, %3218) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3220 = "mix.module.linear"(%3219) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.25.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3221 = "mix.comp.silu"(%3220) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3222 = "mix.module.linear"(%3219) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.25.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3223 = "mix.prim.mul"(%3221, %3222) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3224 = "mix.module.linear"(%3223) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.25.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3225 = "mix.prim.add"(%3224, %3210) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3226 = "mix.comp.weight"() <{param_loc = "transformer.h.26.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3227 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3228 = "mix.prim.pow"(%3225, %3227) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3229 = "mix.comp.mean"(%3228) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3230 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3231 = "mix.prim.add"(%3229, %3230) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3232 = "mix.prim.rsqrt"(%3231) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3233 = "mix.prim.mul"(%3225, %3232) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3234 = "mix.prim.mul"(%3226, %3233) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3235 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3236 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3237 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3238 = "mix.comp.weight"() <{param_loc = "transformer.h.26.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3239 = "mix.prim.transpose"(%3234) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3240 = "mix.prim.transpose"(%3235) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3241 = "mix.prim.reshape"(%3239) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3242 = "mix.prim.matmul"(%3241, %3240) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3243 = "mix.prim.reshape"(%3242) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3244 = "mix.prim.reshape"(%3243) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3245 = "mix.prim.transpose"(%3236) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3246 = "mix.prim.reshape"(%3239) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3247 = "mix.prim.matmul"(%3246, %3245) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3248 = "mix.prim.reshape"(%3247) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3249 = "mix.prim.reshape"(%3248) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3250 = "mix.prim.slice"(%3249) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3251 = "mix.prim.slice"(%3249) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3252 = "mix.prim.reshape"(%3244) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3253 = "mix.prim.reshape"(%3250) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3254 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3255 = "mix.prim.convert"(%3254) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3256 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3257 = "mix.prim.div"(%3255, %3256) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3258 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3259 = "mix.prim.pow"(%3258, %3257) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3260 = "mix.prim.reciprocal"(%3259) : (tensor<80xf16>) -> tensor<80xf16>
    %3261 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3262 = "mix.prim.mul"(%3261, %3260) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3263 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3264 = "mix.prim.unsqueeze"(%3263) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3265 = "mix.prim.permute"(%3264) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3266 = "mix.prim.unsqueeze"(%3262) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3267 = "mix.prim.permute"(%3266) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3268 = "mix.prim.mul"(%3265, %3267) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3269 = "mix.prim.concat"(%3268, %3268) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3270 = "mix.prim.cos"(%3269) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3271 = "mix.prim.slice"(%3270) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3272 = "mix.prim.unsqueeze"(%3271) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3273 = "mix.prim.slice"(%3272) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3274 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3275 = "mix.prim.mul"(%3273, %3274) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3276 = "mix.prim.sin"(%3269) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3277 = "mix.prim.slice"(%3276) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3278 = "mix.prim.unsqueeze"(%3277) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3279 = "mix.prim.slice"(%3278) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3280 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3281 = "mix.prim.mul"(%3279, %3280) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3282 = "mix.prim.slice"(%3275) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3283 = "mix.prim.slice"(%3281) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3284 = "mix.prim.mul"(%3252, %3282) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3285 = "mix.prim.slice"(%3252) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3286 = "mix.prim.slice"(%3252) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3287 = "mix.prim.neg"(%3286) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3288 = "mix.prim.concat"(%3287, %3285) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3289 = "mix.prim.mul"(%3288, %3283) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3290 = "mix.prim.add"(%3284, %3289) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3291 = "mix.prim.mul"(%3253, %3282) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3292 = "mix.prim.slice"(%3253) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3293 = "mix.prim.slice"(%3253) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3294 = "mix.prim.neg"(%3293) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3295 = "mix.prim.concat"(%3294, %3292) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3296 = "mix.prim.mul"(%3295, %3283) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3297 = "mix.prim.add"(%3291, %3296) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3298 = "mix.prim.reshape"(%3290) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3299 = "mix.prim.reshape"(%3297) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3300 = "mix.prim.reshape"(%3298) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3301 = "mix.prim.reshape"(%3299) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3302 = "mix.prim.transpose"(%3300) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3303 = "mix.prim.transpose"(%3301) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3304 = "mix.prim.transpose"(%3303) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3305 = "mix.prim.unsqueeze"(%3302) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3306 = "mix.prim.permute"(%3305) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3307 = "mix.prim.unsqueeze"(%3304) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3308 = "mix.prim.permute"(%3307) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3309 = "mix.prim.permute"(%3306) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3310 = "mix.prim.reshape"(%3309) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3311 = "mix.prim.permute"(%3308) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3312 = "mix.prim.reshape"(%3311) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3313 = "mix.prim.batch_matmul"(%3310, %3312) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3314 = "mix.prim.reshape"(%3313) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3315 = "mix.prim.permute"(%3314) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3316 = "mix.prim.reshape"(%3315) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3317 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3318 = "mix.prim.mul"(%3316, %3317) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3319 = "mix.prim.reshape"(%3318) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3320 = "mix.comp.masked_fill"(%3319, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %3321 = "mix.comp.softmax"(%3320) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3322 = "mix.prim.reshape"(%3321) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3323 = "mix.prim.reshape"(%3251) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3324 = "mix.prim.transpose"(%3323) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3325 = "mix.prim.batch_matmul"(%3322, %3324) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3326 = "mix.prim.reshape"(%3325) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3327 = "mix.prim.permute"(%3326) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3328 = "mix.prim.reshape"(%3327) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3329 = "mix.prim.reshape"(%3328) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3330 = "mix.prim.transpose"(%3237) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3331 = "mix.prim.matmul"(%3329, %3330) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3332 = "mix.prim.add"(%3331, %3238) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3333 = "mix.prim.reshape"(%3332) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3334 = "mix.prim.mul"(%3225, %3333) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3335 = "mix.comp.weight"() <{param_loc = "transformer.h.26.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3336 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3337 = "mix.prim.pow"(%3334, %3336) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3338 = "mix.comp.mean"(%3337) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3339 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3340 = "mix.prim.add"(%3338, %3339) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3341 = "mix.prim.rsqrt"(%3340) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3342 = "mix.prim.mul"(%3334, %3341) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3343 = "mix.prim.mul"(%3335, %3342) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3344 = "mix.module.linear"(%3343) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.26.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3345 = "mix.comp.silu"(%3344) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3346 = "mix.module.linear"(%3343) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.26.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3347 = "mix.prim.mul"(%3345, %3346) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3348 = "mix.module.linear"(%3347) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.26.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3349 = "mix.prim.add"(%3348, %3334) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3350 = "mix.comp.weight"() <{param_loc = "transformer.h.27.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3351 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3352 = "mix.prim.pow"(%3349, %3351) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3353 = "mix.comp.mean"(%3352) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3354 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3355 = "mix.prim.add"(%3353, %3354) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3356 = "mix.prim.rsqrt"(%3355) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3357 = "mix.prim.mul"(%3349, %3356) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3358 = "mix.prim.mul"(%3350, %3357) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3359 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3360 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3361 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3362 = "mix.comp.weight"() <{param_loc = "transformer.h.27.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3363 = "mix.prim.transpose"(%3358) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3364 = "mix.prim.transpose"(%3359) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3365 = "mix.prim.reshape"(%3363) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3366 = "mix.prim.matmul"(%3365, %3364) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3367 = "mix.prim.reshape"(%3366) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3368 = "mix.prim.reshape"(%3367) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3369 = "mix.prim.transpose"(%3360) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3370 = "mix.prim.reshape"(%3363) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3371 = "mix.prim.matmul"(%3370, %3369) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3372 = "mix.prim.reshape"(%3371) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3373 = "mix.prim.reshape"(%3372) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3374 = "mix.prim.slice"(%3373) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3375 = "mix.prim.slice"(%3373) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3376 = "mix.prim.reshape"(%3368) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3377 = "mix.prim.reshape"(%3374) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3378 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3379 = "mix.prim.convert"(%3378) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3380 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3381 = "mix.prim.div"(%3379, %3380) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3382 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3383 = "mix.prim.pow"(%3382, %3381) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3384 = "mix.prim.reciprocal"(%3383) : (tensor<80xf16>) -> tensor<80xf16>
    %3385 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3386 = "mix.prim.mul"(%3385, %3384) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3387 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3388 = "mix.prim.unsqueeze"(%3387) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3389 = "mix.prim.permute"(%3388) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3390 = "mix.prim.unsqueeze"(%3386) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3391 = "mix.prim.permute"(%3390) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3392 = "mix.prim.mul"(%3389, %3391) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3393 = "mix.prim.concat"(%3392, %3392) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3394 = "mix.prim.cos"(%3393) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3395 = "mix.prim.slice"(%3394) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3396 = "mix.prim.unsqueeze"(%3395) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3397 = "mix.prim.slice"(%3396) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3398 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3399 = "mix.prim.mul"(%3397, %3398) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3400 = "mix.prim.sin"(%3393) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3401 = "mix.prim.slice"(%3400) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3402 = "mix.prim.unsqueeze"(%3401) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3403 = "mix.prim.slice"(%3402) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3404 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3405 = "mix.prim.mul"(%3403, %3404) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3406 = "mix.prim.slice"(%3399) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3407 = "mix.prim.slice"(%3405) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3408 = "mix.prim.mul"(%3376, %3406) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3409 = "mix.prim.slice"(%3376) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3410 = "mix.prim.slice"(%3376) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3411 = "mix.prim.neg"(%3410) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3412 = "mix.prim.concat"(%3411, %3409) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3413 = "mix.prim.mul"(%3412, %3407) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3414 = "mix.prim.add"(%3408, %3413) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3415 = "mix.prim.mul"(%3377, %3406) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3416 = "mix.prim.slice"(%3377) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3417 = "mix.prim.slice"(%3377) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3418 = "mix.prim.neg"(%3417) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3419 = "mix.prim.concat"(%3418, %3416) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3420 = "mix.prim.mul"(%3419, %3407) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3421 = "mix.prim.add"(%3415, %3420) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3422 = "mix.prim.reshape"(%3414) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3423 = "mix.prim.reshape"(%3421) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3424 = "mix.prim.reshape"(%3422) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3425 = "mix.prim.reshape"(%3423) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3426 = "mix.prim.transpose"(%3424) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3427 = "mix.prim.transpose"(%3425) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3428 = "mix.prim.transpose"(%3427) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3429 = "mix.prim.unsqueeze"(%3426) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3430 = "mix.prim.permute"(%3429) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3431 = "mix.prim.unsqueeze"(%3428) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3432 = "mix.prim.permute"(%3431) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3433 = "mix.prim.permute"(%3430) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3434 = "mix.prim.reshape"(%3433) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3435 = "mix.prim.permute"(%3432) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3436 = "mix.prim.reshape"(%3435) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3437 = "mix.prim.batch_matmul"(%3434, %3436) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3438 = "mix.prim.reshape"(%3437) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3439 = "mix.prim.permute"(%3438) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3440 = "mix.prim.reshape"(%3439) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3441 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3442 = "mix.prim.mul"(%3440, %3441) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3443 = "mix.prim.reshape"(%3442) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3444 = "mix.comp.masked_fill"(%3443, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %3445 = "mix.comp.softmax"(%3444) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3446 = "mix.prim.reshape"(%3445) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3447 = "mix.prim.reshape"(%3375) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3448 = "mix.prim.transpose"(%3447) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3449 = "mix.prim.batch_matmul"(%3446, %3448) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3450 = "mix.prim.reshape"(%3449) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3451 = "mix.prim.permute"(%3450) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3452 = "mix.prim.reshape"(%3451) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3453 = "mix.prim.reshape"(%3452) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3454 = "mix.prim.transpose"(%3361) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3455 = "mix.prim.matmul"(%3453, %3454) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3456 = "mix.prim.add"(%3455, %3362) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3457 = "mix.prim.reshape"(%3456) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3458 = "mix.prim.mul"(%3349, %3457) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3459 = "mix.comp.weight"() <{param_loc = "transformer.h.27.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3460 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3461 = "mix.prim.pow"(%3458, %3460) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3462 = "mix.comp.mean"(%3461) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3463 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3464 = "mix.prim.add"(%3462, %3463) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3465 = "mix.prim.rsqrt"(%3464) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3466 = "mix.prim.mul"(%3458, %3465) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3467 = "mix.prim.mul"(%3459, %3466) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3468 = "mix.module.linear"(%3467) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.27.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3469 = "mix.comp.silu"(%3468) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3470 = "mix.module.linear"(%3467) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.27.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3471 = "mix.prim.mul"(%3469, %3470) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3472 = "mix.module.linear"(%3471) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.27.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3473 = "mix.prim.add"(%3472, %3458) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3474 = "mix.comp.weight"() <{param_loc = "transformer.h.28.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3475 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3476 = "mix.prim.pow"(%3473, %3475) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3477 = "mix.comp.mean"(%3476) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3478 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3479 = "mix.prim.add"(%3477, %3478) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3480 = "mix.prim.rsqrt"(%3479) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3481 = "mix.prim.mul"(%3473, %3480) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3482 = "mix.prim.mul"(%3474, %3481) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3483 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3484 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3485 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3486 = "mix.comp.weight"() <{param_loc = "transformer.h.28.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3487 = "mix.prim.transpose"(%3482) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3488 = "mix.prim.transpose"(%3483) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3489 = "mix.prim.reshape"(%3487) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3490 = "mix.prim.matmul"(%3489, %3488) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3491 = "mix.prim.reshape"(%3490) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3492 = "mix.prim.reshape"(%3491) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3493 = "mix.prim.transpose"(%3484) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3494 = "mix.prim.reshape"(%3487) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3495 = "mix.prim.matmul"(%3494, %3493) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3496 = "mix.prim.reshape"(%3495) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3497 = "mix.prim.reshape"(%3496) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3498 = "mix.prim.slice"(%3497) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3499 = "mix.prim.slice"(%3497) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3500 = "mix.prim.reshape"(%3492) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3501 = "mix.prim.reshape"(%3498) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3502 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3503 = "mix.prim.convert"(%3502) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3504 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3505 = "mix.prim.div"(%3503, %3504) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3506 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3507 = "mix.prim.pow"(%3506, %3505) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3508 = "mix.prim.reciprocal"(%3507) : (tensor<80xf16>) -> tensor<80xf16>
    %3509 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3510 = "mix.prim.mul"(%3509, %3508) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3511 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3512 = "mix.prim.unsqueeze"(%3511) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3513 = "mix.prim.permute"(%3512) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3514 = "mix.prim.unsqueeze"(%3510) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3515 = "mix.prim.permute"(%3514) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3516 = "mix.prim.mul"(%3513, %3515) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3517 = "mix.prim.concat"(%3516, %3516) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3518 = "mix.prim.cos"(%3517) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3519 = "mix.prim.slice"(%3518) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3520 = "mix.prim.unsqueeze"(%3519) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3521 = "mix.prim.slice"(%3520) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3522 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3523 = "mix.prim.mul"(%3521, %3522) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3524 = "mix.prim.sin"(%3517) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3525 = "mix.prim.slice"(%3524) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3526 = "mix.prim.unsqueeze"(%3525) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3527 = "mix.prim.slice"(%3526) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3528 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3529 = "mix.prim.mul"(%3527, %3528) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3530 = "mix.prim.slice"(%3523) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3531 = "mix.prim.slice"(%3529) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3532 = "mix.prim.mul"(%3500, %3530) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3533 = "mix.prim.slice"(%3500) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3534 = "mix.prim.slice"(%3500) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3535 = "mix.prim.neg"(%3534) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3536 = "mix.prim.concat"(%3535, %3533) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3537 = "mix.prim.mul"(%3536, %3531) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3538 = "mix.prim.add"(%3532, %3537) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3539 = "mix.prim.mul"(%3501, %3530) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3540 = "mix.prim.slice"(%3501) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3541 = "mix.prim.slice"(%3501) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3542 = "mix.prim.neg"(%3541) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3543 = "mix.prim.concat"(%3542, %3540) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3544 = "mix.prim.mul"(%3543, %3531) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3545 = "mix.prim.add"(%3539, %3544) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3546 = "mix.prim.reshape"(%3538) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3547 = "mix.prim.reshape"(%3545) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3548 = "mix.prim.reshape"(%3546) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3549 = "mix.prim.reshape"(%3547) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3550 = "mix.prim.transpose"(%3548) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3551 = "mix.prim.transpose"(%3549) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3552 = "mix.prim.transpose"(%3551) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3553 = "mix.prim.unsqueeze"(%3550) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3554 = "mix.prim.permute"(%3553) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3555 = "mix.prim.unsqueeze"(%3552) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3556 = "mix.prim.permute"(%3555) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3557 = "mix.prim.permute"(%3554) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3558 = "mix.prim.reshape"(%3557) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3559 = "mix.prim.permute"(%3556) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3560 = "mix.prim.reshape"(%3559) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3561 = "mix.prim.batch_matmul"(%3558, %3560) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3562 = "mix.prim.reshape"(%3561) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3563 = "mix.prim.permute"(%3562) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3564 = "mix.prim.reshape"(%3563) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3565 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3566 = "mix.prim.mul"(%3564, %3565) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3567 = "mix.prim.reshape"(%3566) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3568 = "mix.comp.masked_fill"(%3567, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %3569 = "mix.comp.softmax"(%3568) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3570 = "mix.prim.reshape"(%3569) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3571 = "mix.prim.reshape"(%3499) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3572 = "mix.prim.transpose"(%3571) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3573 = "mix.prim.batch_matmul"(%3570, %3572) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3574 = "mix.prim.reshape"(%3573) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3575 = "mix.prim.permute"(%3574) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3576 = "mix.prim.reshape"(%3575) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3577 = "mix.prim.reshape"(%3576) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3578 = "mix.prim.transpose"(%3485) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3579 = "mix.prim.matmul"(%3577, %3578) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3580 = "mix.prim.add"(%3579, %3486) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3581 = "mix.prim.reshape"(%3580) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3582 = "mix.prim.mul"(%3473, %3581) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3583 = "mix.comp.weight"() <{param_loc = "transformer.h.28.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3584 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3585 = "mix.prim.pow"(%3582, %3584) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3586 = "mix.comp.mean"(%3585) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3587 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3588 = "mix.prim.add"(%3586, %3587) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3589 = "mix.prim.rsqrt"(%3588) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3590 = "mix.prim.mul"(%3582, %3589) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3591 = "mix.prim.mul"(%3583, %3590) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3592 = "mix.module.linear"(%3591) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.28.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3593 = "mix.comp.silu"(%3592) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3594 = "mix.module.linear"(%3591) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.28.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3595 = "mix.prim.mul"(%3593, %3594) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3596 = "mix.module.linear"(%3595) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.28.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3597 = "mix.prim.add"(%3596, %3582) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3598 = "mix.comp.weight"() <{param_loc = "transformer.h.29.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3599 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3600 = "mix.prim.pow"(%3597, %3599) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3601 = "mix.comp.mean"(%3600) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3602 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3603 = "mix.prim.add"(%3601, %3602) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3604 = "mix.prim.rsqrt"(%3603) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3605 = "mix.prim.mul"(%3597, %3604) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3606 = "mix.prim.mul"(%3598, %3605) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3607 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3608 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3609 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3610 = "mix.comp.weight"() <{param_loc = "transformer.h.29.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3611 = "mix.prim.transpose"(%3606) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3612 = "mix.prim.transpose"(%3607) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3613 = "mix.prim.reshape"(%3611) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3614 = "mix.prim.matmul"(%3613, %3612) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3615 = "mix.prim.reshape"(%3614) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3616 = "mix.prim.reshape"(%3615) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3617 = "mix.prim.transpose"(%3608) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3618 = "mix.prim.reshape"(%3611) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3619 = "mix.prim.matmul"(%3618, %3617) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3620 = "mix.prim.reshape"(%3619) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3621 = "mix.prim.reshape"(%3620) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3622 = "mix.prim.slice"(%3621) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3623 = "mix.prim.slice"(%3621) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3624 = "mix.prim.reshape"(%3616) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3625 = "mix.prim.reshape"(%3622) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3626 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3627 = "mix.prim.convert"(%3626) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3628 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3629 = "mix.prim.div"(%3627, %3628) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3630 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3631 = "mix.prim.pow"(%3630, %3629) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3632 = "mix.prim.reciprocal"(%3631) : (tensor<80xf16>) -> tensor<80xf16>
    %3633 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3634 = "mix.prim.mul"(%3633, %3632) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3635 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3636 = "mix.prim.unsqueeze"(%3635) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3637 = "mix.prim.permute"(%3636) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3638 = "mix.prim.unsqueeze"(%3634) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3639 = "mix.prim.permute"(%3638) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3640 = "mix.prim.mul"(%3637, %3639) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3641 = "mix.prim.concat"(%3640, %3640) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3642 = "mix.prim.cos"(%3641) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3643 = "mix.prim.slice"(%3642) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3644 = "mix.prim.unsqueeze"(%3643) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3645 = "mix.prim.slice"(%3644) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3646 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3647 = "mix.prim.mul"(%3645, %3646) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3648 = "mix.prim.sin"(%3641) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3649 = "mix.prim.slice"(%3648) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3650 = "mix.prim.unsqueeze"(%3649) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3651 = "mix.prim.slice"(%3650) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3652 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3653 = "mix.prim.mul"(%3651, %3652) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3654 = "mix.prim.slice"(%3647) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3655 = "mix.prim.slice"(%3653) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3656 = "mix.prim.mul"(%3624, %3654) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3657 = "mix.prim.slice"(%3624) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3658 = "mix.prim.slice"(%3624) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3659 = "mix.prim.neg"(%3658) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3660 = "mix.prim.concat"(%3659, %3657) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3661 = "mix.prim.mul"(%3660, %3655) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3662 = "mix.prim.add"(%3656, %3661) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3663 = "mix.prim.mul"(%3625, %3654) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3664 = "mix.prim.slice"(%3625) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3665 = "mix.prim.slice"(%3625) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3666 = "mix.prim.neg"(%3665) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3667 = "mix.prim.concat"(%3666, %3664) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3668 = "mix.prim.mul"(%3667, %3655) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3669 = "mix.prim.add"(%3663, %3668) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3670 = "mix.prim.reshape"(%3662) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3671 = "mix.prim.reshape"(%3669) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3672 = "mix.prim.reshape"(%3670) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3673 = "mix.prim.reshape"(%3671) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3674 = "mix.prim.transpose"(%3672) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3675 = "mix.prim.transpose"(%3673) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3676 = "mix.prim.transpose"(%3675) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3677 = "mix.prim.unsqueeze"(%3674) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3678 = "mix.prim.permute"(%3677) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3679 = "mix.prim.unsqueeze"(%3676) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3680 = "mix.prim.permute"(%3679) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3681 = "mix.prim.permute"(%3678) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3682 = "mix.prim.reshape"(%3681) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3683 = "mix.prim.permute"(%3680) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3684 = "mix.prim.reshape"(%3683) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3685 = "mix.prim.batch_matmul"(%3682, %3684) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3686 = "mix.prim.reshape"(%3685) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3687 = "mix.prim.permute"(%3686) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3688 = "mix.prim.reshape"(%3687) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3689 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3690 = "mix.prim.mul"(%3688, %3689) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3691 = "mix.prim.reshape"(%3690) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3692 = "mix.comp.masked_fill"(%3691, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %3693 = "mix.comp.softmax"(%3692) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3694 = "mix.prim.reshape"(%3693) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3695 = "mix.prim.reshape"(%3623) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3696 = "mix.prim.transpose"(%3695) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3697 = "mix.prim.batch_matmul"(%3694, %3696) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3698 = "mix.prim.reshape"(%3697) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3699 = "mix.prim.permute"(%3698) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3700 = "mix.prim.reshape"(%3699) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3701 = "mix.prim.reshape"(%3700) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3702 = "mix.prim.transpose"(%3609) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3703 = "mix.prim.matmul"(%3701, %3702) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3704 = "mix.prim.add"(%3703, %3610) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3705 = "mix.prim.reshape"(%3704) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3706 = "mix.prim.mul"(%3597, %3705) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3707 = "mix.comp.weight"() <{param_loc = "transformer.h.29.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3708 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3709 = "mix.prim.pow"(%3706, %3708) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3710 = "mix.comp.mean"(%3709) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3711 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3712 = "mix.prim.add"(%3710, %3711) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3713 = "mix.prim.rsqrt"(%3712) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3714 = "mix.prim.mul"(%3706, %3713) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3715 = "mix.prim.mul"(%3707, %3714) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3716 = "mix.module.linear"(%3715) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.29.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3717 = "mix.comp.silu"(%3716) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3718 = "mix.module.linear"(%3715) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.29.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3719 = "mix.prim.mul"(%3717, %3718) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3720 = "mix.module.linear"(%3719) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.29.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3721 = "mix.prim.add"(%3720, %3706) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3722 = "mix.comp.weight"() <{param_loc = "transformer.h.30.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3723 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3724 = "mix.prim.pow"(%3721, %3723) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3725 = "mix.comp.mean"(%3724) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3726 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3727 = "mix.prim.add"(%3725, %3726) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3728 = "mix.prim.rsqrt"(%3727) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3729 = "mix.prim.mul"(%3721, %3728) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3730 = "mix.prim.mul"(%3722, %3729) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3731 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3732 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3733 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3734 = "mix.comp.weight"() <{param_loc = "transformer.h.30.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3735 = "mix.prim.transpose"(%3730) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3736 = "mix.prim.transpose"(%3731) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3737 = "mix.prim.reshape"(%3735) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3738 = "mix.prim.matmul"(%3737, %3736) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3739 = "mix.prim.reshape"(%3738) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3740 = "mix.prim.reshape"(%3739) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3741 = "mix.prim.transpose"(%3732) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3742 = "mix.prim.reshape"(%3735) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3743 = "mix.prim.matmul"(%3742, %3741) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3744 = "mix.prim.reshape"(%3743) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3745 = "mix.prim.reshape"(%3744) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3746 = "mix.prim.slice"(%3745) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3747 = "mix.prim.slice"(%3745) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3748 = "mix.prim.reshape"(%3740) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3749 = "mix.prim.reshape"(%3746) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3750 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3751 = "mix.prim.convert"(%3750) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3752 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3753 = "mix.prim.div"(%3751, %3752) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3754 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3755 = "mix.prim.pow"(%3754, %3753) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3756 = "mix.prim.reciprocal"(%3755) : (tensor<80xf16>) -> tensor<80xf16>
    %3757 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3758 = "mix.prim.mul"(%3757, %3756) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3759 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3760 = "mix.prim.unsqueeze"(%3759) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3761 = "mix.prim.permute"(%3760) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3762 = "mix.prim.unsqueeze"(%3758) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3763 = "mix.prim.permute"(%3762) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3764 = "mix.prim.mul"(%3761, %3763) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3765 = "mix.prim.concat"(%3764, %3764) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3766 = "mix.prim.cos"(%3765) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3767 = "mix.prim.slice"(%3766) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3768 = "mix.prim.unsqueeze"(%3767) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3769 = "mix.prim.slice"(%3768) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3770 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3771 = "mix.prim.mul"(%3769, %3770) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3772 = "mix.prim.sin"(%3765) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3773 = "mix.prim.slice"(%3772) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3774 = "mix.prim.unsqueeze"(%3773) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3775 = "mix.prim.slice"(%3774) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3776 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3777 = "mix.prim.mul"(%3775, %3776) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3778 = "mix.prim.slice"(%3771) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3779 = "mix.prim.slice"(%3777) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3780 = "mix.prim.mul"(%3748, %3778) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3781 = "mix.prim.slice"(%3748) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3782 = "mix.prim.slice"(%3748) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3783 = "mix.prim.neg"(%3782) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3784 = "mix.prim.concat"(%3783, %3781) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3785 = "mix.prim.mul"(%3784, %3779) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3786 = "mix.prim.add"(%3780, %3785) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3787 = "mix.prim.mul"(%3749, %3778) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3788 = "mix.prim.slice"(%3749) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3789 = "mix.prim.slice"(%3749) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3790 = "mix.prim.neg"(%3789) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3791 = "mix.prim.concat"(%3790, %3788) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3792 = "mix.prim.mul"(%3791, %3779) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3793 = "mix.prim.add"(%3787, %3792) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3794 = "mix.prim.reshape"(%3786) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3795 = "mix.prim.reshape"(%3793) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3796 = "mix.prim.reshape"(%3794) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3797 = "mix.prim.reshape"(%3795) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3798 = "mix.prim.transpose"(%3796) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3799 = "mix.prim.transpose"(%3797) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3800 = "mix.prim.transpose"(%3799) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3801 = "mix.prim.unsqueeze"(%3798) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3802 = "mix.prim.permute"(%3801) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3803 = "mix.prim.unsqueeze"(%3800) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3804 = "mix.prim.permute"(%3803) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3805 = "mix.prim.permute"(%3802) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3806 = "mix.prim.reshape"(%3805) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3807 = "mix.prim.permute"(%3804) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3808 = "mix.prim.reshape"(%3807) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3809 = "mix.prim.batch_matmul"(%3806, %3808) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3810 = "mix.prim.reshape"(%3809) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3811 = "mix.prim.permute"(%3810) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3812 = "mix.prim.reshape"(%3811) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3813 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3814 = "mix.prim.mul"(%3812, %3813) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3815 = "mix.prim.reshape"(%3814) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3816 = "mix.comp.masked_fill"(%3815, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %3817 = "mix.comp.softmax"(%3816) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3818 = "mix.prim.reshape"(%3817) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3819 = "mix.prim.reshape"(%3747) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3820 = "mix.prim.transpose"(%3819) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3821 = "mix.prim.batch_matmul"(%3818, %3820) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3822 = "mix.prim.reshape"(%3821) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3823 = "mix.prim.permute"(%3822) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3824 = "mix.prim.reshape"(%3823) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3825 = "mix.prim.reshape"(%3824) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3826 = "mix.prim.transpose"(%3733) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3827 = "mix.prim.matmul"(%3825, %3826) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3828 = "mix.prim.add"(%3827, %3734) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3829 = "mix.prim.reshape"(%3828) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3830 = "mix.prim.mul"(%3721, %3829) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3831 = "mix.comp.weight"() <{param_loc = "transformer.h.30.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3832 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3833 = "mix.prim.pow"(%3830, %3832) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3834 = "mix.comp.mean"(%3833) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3835 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3836 = "mix.prim.add"(%3834, %3835) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3837 = "mix.prim.rsqrt"(%3836) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3838 = "mix.prim.mul"(%3830, %3837) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3839 = "mix.prim.mul"(%3831, %3838) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3840 = "mix.module.linear"(%3839) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.30.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3841 = "mix.comp.silu"(%3840) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3842 = "mix.module.linear"(%3839) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.30.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3843 = "mix.prim.mul"(%3841, %3842) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3844 = "mix.module.linear"(%3843) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.30.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3845 = "mix.prim.add"(%3844, %3830) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3846 = "mix.comp.weight"() <{param_loc = "transformer.h.31.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3847 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3848 = "mix.prim.pow"(%3845, %3847) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3849 = "mix.comp.mean"(%3848) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3850 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3851 = "mix.prim.add"(%3849, %3850) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3852 = "mix.prim.rsqrt"(%3851) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3853 = "mix.prim.mul"(%3845, %3852) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3854 = "mix.prim.mul"(%3846, %3853) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3855 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3856 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3857 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3858 = "mix.comp.weight"() <{param_loc = "transformer.h.31.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3859 = "mix.prim.transpose"(%3854) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3860 = "mix.prim.transpose"(%3855) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3861 = "mix.prim.reshape"(%3859) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3862 = "mix.prim.matmul"(%3861, %3860) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3863 = "mix.prim.reshape"(%3862) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3864 = "mix.prim.reshape"(%3863) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3865 = "mix.prim.transpose"(%3856) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3866 = "mix.prim.reshape"(%3859) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3867 = "mix.prim.matmul"(%3866, %3865) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3868 = "mix.prim.reshape"(%3867) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3869 = "mix.prim.reshape"(%3868) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3870 = "mix.prim.slice"(%3869) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3871 = "mix.prim.slice"(%3869) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3872 = "mix.prim.reshape"(%3864) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3873 = "mix.prim.reshape"(%3870) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3874 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3875 = "mix.prim.convert"(%3874) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %3876 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %3877 = "mix.prim.div"(%3875, %3876) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %3878 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %3879 = "mix.prim.pow"(%3878, %3877) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3880 = "mix.prim.reciprocal"(%3879) : (tensor<80xf16>) -> tensor<80xf16>
    %3881 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3882 = "mix.prim.mul"(%3881, %3880) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %3883 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %3884 = "mix.prim.unsqueeze"(%3883) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %3885 = "mix.prim.permute"(%3884) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %3886 = "mix.prim.unsqueeze"(%3882) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %3887 = "mix.prim.permute"(%3886) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %3888 = "mix.prim.mul"(%3885, %3887) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %3889 = "mix.prim.concat"(%3888, %3888) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %3890 = "mix.prim.cos"(%3889) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3891 = "mix.prim.slice"(%3890) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3892 = "mix.prim.unsqueeze"(%3891) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3893 = "mix.prim.slice"(%3892) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3894 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3895 = "mix.prim.mul"(%3893, %3894) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3896 = "mix.prim.sin"(%3889) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3897 = "mix.prim.slice"(%3896) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %3898 = "mix.prim.unsqueeze"(%3897) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %3899 = "mix.prim.slice"(%3898) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3900 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %3901 = "mix.prim.mul"(%3899, %3900) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %3902 = "mix.prim.slice"(%3895) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3903 = "mix.prim.slice"(%3901) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %3904 = "mix.prim.mul"(%3872, %3902) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3905 = "mix.prim.slice"(%3872) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3906 = "mix.prim.slice"(%3872) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3907 = "mix.prim.neg"(%3906) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3908 = "mix.prim.concat"(%3907, %3905) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3909 = "mix.prim.mul"(%3908, %3903) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3910 = "mix.prim.add"(%3904, %3909) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3911 = "mix.prim.mul"(%3873, %3902) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3912 = "mix.prim.slice"(%3873) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3913 = "mix.prim.slice"(%3873) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %3914 = "mix.prim.neg"(%3913) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %3915 = "mix.prim.concat"(%3914, %3912) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %3916 = "mix.prim.mul"(%3915, %3903) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %3917 = "mix.prim.add"(%3911, %3916) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %3918 = "mix.prim.reshape"(%3910) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3919 = "mix.prim.reshape"(%3917) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %3920 = "mix.prim.reshape"(%3918) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3921 = "mix.prim.reshape"(%3919) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3922 = "mix.prim.transpose"(%3920) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3923 = "mix.prim.transpose"(%3921) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3924 = "mix.prim.transpose"(%3923) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %3925 = "mix.prim.unsqueeze"(%3922) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %3926 = "mix.prim.permute"(%3925) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %3927 = "mix.prim.unsqueeze"(%3924) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %3928 = "mix.prim.permute"(%3927) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %3929 = "mix.prim.permute"(%3926) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %3930 = "mix.prim.reshape"(%3929) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %3931 = "mix.prim.permute"(%3928) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %3932 = "mix.prim.reshape"(%3931) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %3933 = "mix.prim.batch_matmul"(%3930, %3932) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %3934 = "mix.prim.reshape"(%3933) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %3935 = "mix.prim.permute"(%3934) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %3936 = "mix.prim.reshape"(%3935) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %3937 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %3938 = "mix.prim.mul"(%3936, %3937) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %3939 = "mix.prim.reshape"(%3938) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3940 = "mix.comp.masked_fill"(%3939, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %3941 = "mix.comp.softmax"(%3940) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %3942 = "mix.prim.reshape"(%3941) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %3943 = "mix.prim.reshape"(%3871) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3944 = "mix.prim.transpose"(%3943) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %3945 = "mix.prim.batch_matmul"(%3942, %3944) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %3946 = "mix.prim.reshape"(%3945) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %3947 = "mix.prim.permute"(%3946) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %3948 = "mix.prim.reshape"(%3947) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %3949 = "mix.prim.reshape"(%3948) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %3950 = "mix.prim.transpose"(%3857) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3951 = "mix.prim.matmul"(%3949, %3950) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3952 = "mix.prim.add"(%3951, %3858) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %3953 = "mix.prim.reshape"(%3952) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %3954 = "mix.prim.mul"(%3845, %3953) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3955 = "mix.comp.weight"() <{param_loc = "transformer.h.31.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %3956 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3957 = "mix.prim.pow"(%3954, %3956) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3958 = "mix.comp.mean"(%3957) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3959 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3960 = "mix.prim.add"(%3958, %3959) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3961 = "mix.prim.rsqrt"(%3960) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3962 = "mix.prim.mul"(%3954, %3961) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3963 = "mix.prim.mul"(%3955, %3962) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3964 = "mix.module.linear"(%3963) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.31.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3965 = "mix.comp.silu"(%3964) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3966 = "mix.module.linear"(%3963) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.31.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %3967 = "mix.prim.mul"(%3965, %3966) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %3968 = "mix.module.linear"(%3967) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.31.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %3969 = "mix.prim.add"(%3968, %3954) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3970 = "mix.comp.weight"() <{param_loc = "transformer.h.32.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %3971 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %3972 = "mix.prim.pow"(%3969, %3971) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %3973 = "mix.comp.mean"(%3972) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %3974 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %3975 = "mix.prim.add"(%3973, %3974) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %3976 = "mix.prim.rsqrt"(%3975) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %3977 = "mix.prim.mul"(%3969, %3976) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %3978 = "mix.prim.mul"(%3970, %3977) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %3979 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %3980 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %3981 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %3982 = "mix.comp.weight"() <{param_loc = "transformer.h.32.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %3983 = "mix.prim.transpose"(%3978) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %3984 = "mix.prim.transpose"(%3979) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %3985 = "mix.prim.reshape"(%3983) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3986 = "mix.prim.matmul"(%3985, %3984) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %3987 = "mix.prim.reshape"(%3986) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %3988 = "mix.prim.reshape"(%3987) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %3989 = "mix.prim.transpose"(%3980) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %3990 = "mix.prim.reshape"(%3983) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %3991 = "mix.prim.matmul"(%3990, %3989) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %3992 = "mix.prim.reshape"(%3991) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %3993 = "mix.prim.reshape"(%3992) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %3994 = "mix.prim.slice"(%3993) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3995 = "mix.prim.slice"(%3993) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %3996 = "mix.prim.reshape"(%3988) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3997 = "mix.prim.reshape"(%3994) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %3998 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %3999 = "mix.prim.convert"(%3998) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4000 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4001 = "mix.prim.div"(%3999, %4000) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4002 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4003 = "mix.prim.pow"(%4002, %4001) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4004 = "mix.prim.reciprocal"(%4003) : (tensor<80xf16>) -> tensor<80xf16>
    %4005 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4006 = "mix.prim.mul"(%4005, %4004) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4007 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4008 = "mix.prim.unsqueeze"(%4007) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4009 = "mix.prim.permute"(%4008) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4010 = "mix.prim.unsqueeze"(%4006) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4011 = "mix.prim.permute"(%4010) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4012 = "mix.prim.mul"(%4009, %4011) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4013 = "mix.prim.concat"(%4012, %4012) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4014 = "mix.prim.cos"(%4013) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4015 = "mix.prim.slice"(%4014) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4016 = "mix.prim.unsqueeze"(%4015) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4017 = "mix.prim.slice"(%4016) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4018 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4019 = "mix.prim.mul"(%4017, %4018) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4020 = "mix.prim.sin"(%4013) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4021 = "mix.prim.slice"(%4020) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4022 = "mix.prim.unsqueeze"(%4021) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4023 = "mix.prim.slice"(%4022) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4024 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4025 = "mix.prim.mul"(%4023, %4024) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4026 = "mix.prim.slice"(%4019) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4027 = "mix.prim.slice"(%4025) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4028 = "mix.prim.mul"(%3996, %4026) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4029 = "mix.prim.slice"(%3996) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4030 = "mix.prim.slice"(%3996) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4031 = "mix.prim.neg"(%4030) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4032 = "mix.prim.concat"(%4031, %4029) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4033 = "mix.prim.mul"(%4032, %4027) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4034 = "mix.prim.add"(%4028, %4033) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4035 = "mix.prim.mul"(%3997, %4026) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4036 = "mix.prim.slice"(%3997) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4037 = "mix.prim.slice"(%3997) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4038 = "mix.prim.neg"(%4037) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4039 = "mix.prim.concat"(%4038, %4036) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4040 = "mix.prim.mul"(%4039, %4027) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4041 = "mix.prim.add"(%4035, %4040) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4042 = "mix.prim.reshape"(%4034) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4043 = "mix.prim.reshape"(%4041) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4044 = "mix.prim.reshape"(%4042) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4045 = "mix.prim.reshape"(%4043) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4046 = "mix.prim.transpose"(%4044) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4047 = "mix.prim.transpose"(%4045) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4048 = "mix.prim.transpose"(%4047) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4049 = "mix.prim.unsqueeze"(%4046) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4050 = "mix.prim.permute"(%4049) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4051 = "mix.prim.unsqueeze"(%4048) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4052 = "mix.prim.permute"(%4051) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4053 = "mix.prim.permute"(%4050) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4054 = "mix.prim.reshape"(%4053) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4055 = "mix.prim.permute"(%4052) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4056 = "mix.prim.reshape"(%4055) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4057 = "mix.prim.batch_matmul"(%4054, %4056) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4058 = "mix.prim.reshape"(%4057) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4059 = "mix.prim.permute"(%4058) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4060 = "mix.prim.reshape"(%4059) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4061 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4062 = "mix.prim.mul"(%4060, %4061) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4063 = "mix.prim.reshape"(%4062) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4064 = "mix.comp.masked_fill"(%4063, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %4065 = "mix.comp.softmax"(%4064) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4066 = "mix.prim.reshape"(%4065) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4067 = "mix.prim.reshape"(%3995) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4068 = "mix.prim.transpose"(%4067) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4069 = "mix.prim.batch_matmul"(%4066, %4068) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4070 = "mix.prim.reshape"(%4069) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4071 = "mix.prim.permute"(%4070) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4072 = "mix.prim.reshape"(%4071) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4073 = "mix.prim.reshape"(%4072) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4074 = "mix.prim.transpose"(%3981) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4075 = "mix.prim.matmul"(%4073, %4074) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4076 = "mix.prim.add"(%4075, %3982) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4077 = "mix.prim.reshape"(%4076) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4078 = "mix.prim.mul"(%3969, %4077) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4079 = "mix.comp.weight"() <{param_loc = "transformer.h.32.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4080 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4081 = "mix.prim.pow"(%4078, %4080) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4082 = "mix.comp.mean"(%4081) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4083 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4084 = "mix.prim.add"(%4082, %4083) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4085 = "mix.prim.rsqrt"(%4084) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4086 = "mix.prim.mul"(%4078, %4085) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4087 = "mix.prim.mul"(%4079, %4086) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4088 = "mix.module.linear"(%4087) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.32.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4089 = "mix.comp.silu"(%4088) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4090 = "mix.module.linear"(%4087) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.32.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4091 = "mix.prim.mul"(%4089, %4090) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4092 = "mix.module.linear"(%4091) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.32.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4093 = "mix.prim.add"(%4092, %4078) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4094 = "mix.comp.weight"() <{param_loc = "transformer.h.33.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4095 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4096 = "mix.prim.pow"(%4093, %4095) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4097 = "mix.comp.mean"(%4096) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4098 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4099 = "mix.prim.add"(%4097, %4098) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4100 = "mix.prim.rsqrt"(%4099) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4101 = "mix.prim.mul"(%4093, %4100) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4102 = "mix.prim.mul"(%4094, %4101) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4103 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4104 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4105 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4106 = "mix.comp.weight"() <{param_loc = "transformer.h.33.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4107 = "mix.prim.transpose"(%4102) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4108 = "mix.prim.transpose"(%4103) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4109 = "mix.prim.reshape"(%4107) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4110 = "mix.prim.matmul"(%4109, %4108) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4111 = "mix.prim.reshape"(%4110) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4112 = "mix.prim.reshape"(%4111) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4113 = "mix.prim.transpose"(%4104) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4114 = "mix.prim.reshape"(%4107) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4115 = "mix.prim.matmul"(%4114, %4113) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4116 = "mix.prim.reshape"(%4115) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4117 = "mix.prim.reshape"(%4116) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4118 = "mix.prim.slice"(%4117) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4119 = "mix.prim.slice"(%4117) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4120 = "mix.prim.reshape"(%4112) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4121 = "mix.prim.reshape"(%4118) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4122 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4123 = "mix.prim.convert"(%4122) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4124 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4125 = "mix.prim.div"(%4123, %4124) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4126 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4127 = "mix.prim.pow"(%4126, %4125) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4128 = "mix.prim.reciprocal"(%4127) : (tensor<80xf16>) -> tensor<80xf16>
    %4129 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4130 = "mix.prim.mul"(%4129, %4128) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4131 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4132 = "mix.prim.unsqueeze"(%4131) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4133 = "mix.prim.permute"(%4132) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4134 = "mix.prim.unsqueeze"(%4130) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4135 = "mix.prim.permute"(%4134) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4136 = "mix.prim.mul"(%4133, %4135) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4137 = "mix.prim.concat"(%4136, %4136) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4138 = "mix.prim.cos"(%4137) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4139 = "mix.prim.slice"(%4138) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4140 = "mix.prim.unsqueeze"(%4139) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4141 = "mix.prim.slice"(%4140) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4142 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4143 = "mix.prim.mul"(%4141, %4142) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4144 = "mix.prim.sin"(%4137) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4145 = "mix.prim.slice"(%4144) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4146 = "mix.prim.unsqueeze"(%4145) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4147 = "mix.prim.slice"(%4146) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4148 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4149 = "mix.prim.mul"(%4147, %4148) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4150 = "mix.prim.slice"(%4143) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4151 = "mix.prim.slice"(%4149) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4152 = "mix.prim.mul"(%4120, %4150) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4153 = "mix.prim.slice"(%4120) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4154 = "mix.prim.slice"(%4120) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4155 = "mix.prim.neg"(%4154) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4156 = "mix.prim.concat"(%4155, %4153) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4157 = "mix.prim.mul"(%4156, %4151) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4158 = "mix.prim.add"(%4152, %4157) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4159 = "mix.prim.mul"(%4121, %4150) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4160 = "mix.prim.slice"(%4121) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4161 = "mix.prim.slice"(%4121) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4162 = "mix.prim.neg"(%4161) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4163 = "mix.prim.concat"(%4162, %4160) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4164 = "mix.prim.mul"(%4163, %4151) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4165 = "mix.prim.add"(%4159, %4164) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4166 = "mix.prim.reshape"(%4158) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4167 = "mix.prim.reshape"(%4165) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4168 = "mix.prim.reshape"(%4166) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4169 = "mix.prim.reshape"(%4167) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4170 = "mix.prim.transpose"(%4168) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4171 = "mix.prim.transpose"(%4169) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4172 = "mix.prim.transpose"(%4171) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4173 = "mix.prim.unsqueeze"(%4170) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4174 = "mix.prim.permute"(%4173) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4175 = "mix.prim.unsqueeze"(%4172) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4176 = "mix.prim.permute"(%4175) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4177 = "mix.prim.permute"(%4174) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4178 = "mix.prim.reshape"(%4177) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4179 = "mix.prim.permute"(%4176) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4180 = "mix.prim.reshape"(%4179) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4181 = "mix.prim.batch_matmul"(%4178, %4180) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4182 = "mix.prim.reshape"(%4181) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4183 = "mix.prim.permute"(%4182) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4184 = "mix.prim.reshape"(%4183) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4185 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4186 = "mix.prim.mul"(%4184, %4185) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4187 = "mix.prim.reshape"(%4186) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4188 = "mix.comp.masked_fill"(%4187, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %4189 = "mix.comp.softmax"(%4188) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4190 = "mix.prim.reshape"(%4189) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4191 = "mix.prim.reshape"(%4119) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4192 = "mix.prim.transpose"(%4191) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4193 = "mix.prim.batch_matmul"(%4190, %4192) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4194 = "mix.prim.reshape"(%4193) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4195 = "mix.prim.permute"(%4194) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4196 = "mix.prim.reshape"(%4195) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4197 = "mix.prim.reshape"(%4196) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4198 = "mix.prim.transpose"(%4105) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4199 = "mix.prim.matmul"(%4197, %4198) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4200 = "mix.prim.add"(%4199, %4106) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4201 = "mix.prim.reshape"(%4200) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4202 = "mix.prim.mul"(%4093, %4201) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4203 = "mix.comp.weight"() <{param_loc = "transformer.h.33.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4204 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4205 = "mix.prim.pow"(%4202, %4204) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4206 = "mix.comp.mean"(%4205) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4207 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4208 = "mix.prim.add"(%4206, %4207) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4209 = "mix.prim.rsqrt"(%4208) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4210 = "mix.prim.mul"(%4202, %4209) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4211 = "mix.prim.mul"(%4203, %4210) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4212 = "mix.module.linear"(%4211) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.33.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4213 = "mix.comp.silu"(%4212) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4214 = "mix.module.linear"(%4211) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.33.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4215 = "mix.prim.mul"(%4213, %4214) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4216 = "mix.module.linear"(%4215) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.33.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4217 = "mix.prim.add"(%4216, %4202) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4218 = "mix.comp.weight"() <{param_loc = "transformer.h.34.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4219 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4220 = "mix.prim.pow"(%4217, %4219) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4221 = "mix.comp.mean"(%4220) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4222 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4223 = "mix.prim.add"(%4221, %4222) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4224 = "mix.prim.rsqrt"(%4223) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4225 = "mix.prim.mul"(%4217, %4224) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4226 = "mix.prim.mul"(%4218, %4225) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4227 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4228 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4229 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4230 = "mix.comp.weight"() <{param_loc = "transformer.h.34.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4231 = "mix.prim.transpose"(%4226) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4232 = "mix.prim.transpose"(%4227) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4233 = "mix.prim.reshape"(%4231) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4234 = "mix.prim.matmul"(%4233, %4232) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4235 = "mix.prim.reshape"(%4234) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4236 = "mix.prim.reshape"(%4235) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4237 = "mix.prim.transpose"(%4228) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4238 = "mix.prim.reshape"(%4231) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4239 = "mix.prim.matmul"(%4238, %4237) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4240 = "mix.prim.reshape"(%4239) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4241 = "mix.prim.reshape"(%4240) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4242 = "mix.prim.slice"(%4241) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4243 = "mix.prim.slice"(%4241) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4244 = "mix.prim.reshape"(%4236) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4245 = "mix.prim.reshape"(%4242) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4246 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4247 = "mix.prim.convert"(%4246) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4248 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4249 = "mix.prim.div"(%4247, %4248) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4250 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4251 = "mix.prim.pow"(%4250, %4249) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4252 = "mix.prim.reciprocal"(%4251) : (tensor<80xf16>) -> tensor<80xf16>
    %4253 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4254 = "mix.prim.mul"(%4253, %4252) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4255 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4256 = "mix.prim.unsqueeze"(%4255) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4257 = "mix.prim.permute"(%4256) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4258 = "mix.prim.unsqueeze"(%4254) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4259 = "mix.prim.permute"(%4258) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4260 = "mix.prim.mul"(%4257, %4259) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4261 = "mix.prim.concat"(%4260, %4260) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4262 = "mix.prim.cos"(%4261) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4263 = "mix.prim.slice"(%4262) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4264 = "mix.prim.unsqueeze"(%4263) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4265 = "mix.prim.slice"(%4264) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4266 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4267 = "mix.prim.mul"(%4265, %4266) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4268 = "mix.prim.sin"(%4261) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4269 = "mix.prim.slice"(%4268) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4270 = "mix.prim.unsqueeze"(%4269) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4271 = "mix.prim.slice"(%4270) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4272 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4273 = "mix.prim.mul"(%4271, %4272) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4274 = "mix.prim.slice"(%4267) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4275 = "mix.prim.slice"(%4273) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4276 = "mix.prim.mul"(%4244, %4274) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4277 = "mix.prim.slice"(%4244) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4278 = "mix.prim.slice"(%4244) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4279 = "mix.prim.neg"(%4278) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4280 = "mix.prim.concat"(%4279, %4277) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4281 = "mix.prim.mul"(%4280, %4275) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4282 = "mix.prim.add"(%4276, %4281) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4283 = "mix.prim.mul"(%4245, %4274) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4284 = "mix.prim.slice"(%4245) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4285 = "mix.prim.slice"(%4245) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4286 = "mix.prim.neg"(%4285) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4287 = "mix.prim.concat"(%4286, %4284) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4288 = "mix.prim.mul"(%4287, %4275) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4289 = "mix.prim.add"(%4283, %4288) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4290 = "mix.prim.reshape"(%4282) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4291 = "mix.prim.reshape"(%4289) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4292 = "mix.prim.reshape"(%4290) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4293 = "mix.prim.reshape"(%4291) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4294 = "mix.prim.transpose"(%4292) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4295 = "mix.prim.transpose"(%4293) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4296 = "mix.prim.transpose"(%4295) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4297 = "mix.prim.unsqueeze"(%4294) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4298 = "mix.prim.permute"(%4297) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4299 = "mix.prim.unsqueeze"(%4296) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4300 = "mix.prim.permute"(%4299) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4301 = "mix.prim.permute"(%4298) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4302 = "mix.prim.reshape"(%4301) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4303 = "mix.prim.permute"(%4300) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4304 = "mix.prim.reshape"(%4303) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4305 = "mix.prim.batch_matmul"(%4302, %4304) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4306 = "mix.prim.reshape"(%4305) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4307 = "mix.prim.permute"(%4306) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4308 = "mix.prim.reshape"(%4307) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4309 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4310 = "mix.prim.mul"(%4308, %4309) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4311 = "mix.prim.reshape"(%4310) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4312 = "mix.comp.masked_fill"(%4311, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %4313 = "mix.comp.softmax"(%4312) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4314 = "mix.prim.reshape"(%4313) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4315 = "mix.prim.reshape"(%4243) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4316 = "mix.prim.transpose"(%4315) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4317 = "mix.prim.batch_matmul"(%4314, %4316) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4318 = "mix.prim.reshape"(%4317) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4319 = "mix.prim.permute"(%4318) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4320 = "mix.prim.reshape"(%4319) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4321 = "mix.prim.reshape"(%4320) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4322 = "mix.prim.transpose"(%4229) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4323 = "mix.prim.matmul"(%4321, %4322) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4324 = "mix.prim.add"(%4323, %4230) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4325 = "mix.prim.reshape"(%4324) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4326 = "mix.prim.mul"(%4217, %4325) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4327 = "mix.comp.weight"() <{param_loc = "transformer.h.34.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4328 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4329 = "mix.prim.pow"(%4326, %4328) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4330 = "mix.comp.mean"(%4329) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4331 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4332 = "mix.prim.add"(%4330, %4331) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4333 = "mix.prim.rsqrt"(%4332) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4334 = "mix.prim.mul"(%4326, %4333) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4335 = "mix.prim.mul"(%4327, %4334) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4336 = "mix.module.linear"(%4335) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.34.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4337 = "mix.comp.silu"(%4336) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4338 = "mix.module.linear"(%4335) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.34.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4339 = "mix.prim.mul"(%4337, %4338) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4340 = "mix.module.linear"(%4339) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.34.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4341 = "mix.prim.add"(%4340, %4326) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4342 = "mix.comp.weight"() <{param_loc = "transformer.h.35.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4343 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4344 = "mix.prim.pow"(%4341, %4343) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4345 = "mix.comp.mean"(%4344) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4346 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4347 = "mix.prim.add"(%4345, %4346) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4348 = "mix.prim.rsqrt"(%4347) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4349 = "mix.prim.mul"(%4341, %4348) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4350 = "mix.prim.mul"(%4342, %4349) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4351 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4352 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4353 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4354 = "mix.comp.weight"() <{param_loc = "transformer.h.35.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4355 = "mix.prim.transpose"(%4350) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4356 = "mix.prim.transpose"(%4351) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4357 = "mix.prim.reshape"(%4355) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4358 = "mix.prim.matmul"(%4357, %4356) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4359 = "mix.prim.reshape"(%4358) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4360 = "mix.prim.reshape"(%4359) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4361 = "mix.prim.transpose"(%4352) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4362 = "mix.prim.reshape"(%4355) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4363 = "mix.prim.matmul"(%4362, %4361) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4364 = "mix.prim.reshape"(%4363) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4365 = "mix.prim.reshape"(%4364) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4366 = "mix.prim.slice"(%4365) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4367 = "mix.prim.slice"(%4365) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4368 = "mix.prim.reshape"(%4360) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4369 = "mix.prim.reshape"(%4366) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4370 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4371 = "mix.prim.convert"(%4370) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4372 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4373 = "mix.prim.div"(%4371, %4372) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4374 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4375 = "mix.prim.pow"(%4374, %4373) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4376 = "mix.prim.reciprocal"(%4375) : (tensor<80xf16>) -> tensor<80xf16>
    %4377 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4378 = "mix.prim.mul"(%4377, %4376) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4379 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4380 = "mix.prim.unsqueeze"(%4379) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4381 = "mix.prim.permute"(%4380) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4382 = "mix.prim.unsqueeze"(%4378) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4383 = "mix.prim.permute"(%4382) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4384 = "mix.prim.mul"(%4381, %4383) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4385 = "mix.prim.concat"(%4384, %4384) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4386 = "mix.prim.cos"(%4385) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4387 = "mix.prim.slice"(%4386) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4388 = "mix.prim.unsqueeze"(%4387) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4389 = "mix.prim.slice"(%4388) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4390 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4391 = "mix.prim.mul"(%4389, %4390) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4392 = "mix.prim.sin"(%4385) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4393 = "mix.prim.slice"(%4392) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4394 = "mix.prim.unsqueeze"(%4393) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4395 = "mix.prim.slice"(%4394) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4396 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4397 = "mix.prim.mul"(%4395, %4396) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4398 = "mix.prim.slice"(%4391) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4399 = "mix.prim.slice"(%4397) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4400 = "mix.prim.mul"(%4368, %4398) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4401 = "mix.prim.slice"(%4368) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4402 = "mix.prim.slice"(%4368) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4403 = "mix.prim.neg"(%4402) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4404 = "mix.prim.concat"(%4403, %4401) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4405 = "mix.prim.mul"(%4404, %4399) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4406 = "mix.prim.add"(%4400, %4405) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4407 = "mix.prim.mul"(%4369, %4398) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4408 = "mix.prim.slice"(%4369) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4409 = "mix.prim.slice"(%4369) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4410 = "mix.prim.neg"(%4409) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4411 = "mix.prim.concat"(%4410, %4408) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4412 = "mix.prim.mul"(%4411, %4399) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4413 = "mix.prim.add"(%4407, %4412) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4414 = "mix.prim.reshape"(%4406) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4415 = "mix.prim.reshape"(%4413) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4416 = "mix.prim.reshape"(%4414) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4417 = "mix.prim.reshape"(%4415) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4418 = "mix.prim.transpose"(%4416) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4419 = "mix.prim.transpose"(%4417) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4420 = "mix.prim.transpose"(%4419) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4421 = "mix.prim.unsqueeze"(%4418) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4422 = "mix.prim.permute"(%4421) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4423 = "mix.prim.unsqueeze"(%4420) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4424 = "mix.prim.permute"(%4423) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4425 = "mix.prim.permute"(%4422) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4426 = "mix.prim.reshape"(%4425) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4427 = "mix.prim.permute"(%4424) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4428 = "mix.prim.reshape"(%4427) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4429 = "mix.prim.batch_matmul"(%4426, %4428) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4430 = "mix.prim.reshape"(%4429) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4431 = "mix.prim.permute"(%4430) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4432 = "mix.prim.reshape"(%4431) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4433 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4434 = "mix.prim.mul"(%4432, %4433) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4435 = "mix.prim.reshape"(%4434) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4436 = "mix.comp.masked_fill"(%4435, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %4437 = "mix.comp.softmax"(%4436) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4438 = "mix.prim.reshape"(%4437) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4439 = "mix.prim.reshape"(%4367) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4440 = "mix.prim.transpose"(%4439) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4441 = "mix.prim.batch_matmul"(%4438, %4440) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4442 = "mix.prim.reshape"(%4441) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4443 = "mix.prim.permute"(%4442) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4444 = "mix.prim.reshape"(%4443) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4445 = "mix.prim.reshape"(%4444) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4446 = "mix.prim.transpose"(%4353) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4447 = "mix.prim.matmul"(%4445, %4446) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4448 = "mix.prim.add"(%4447, %4354) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4449 = "mix.prim.reshape"(%4448) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4450 = "mix.prim.mul"(%4341, %4449) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4451 = "mix.comp.weight"() <{param_loc = "transformer.h.35.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4452 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4453 = "mix.prim.pow"(%4450, %4452) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4454 = "mix.comp.mean"(%4453) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4455 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4456 = "mix.prim.add"(%4454, %4455) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4457 = "mix.prim.rsqrt"(%4456) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4458 = "mix.prim.mul"(%4450, %4457) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4459 = "mix.prim.mul"(%4451, %4458) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4460 = "mix.module.linear"(%4459) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.35.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4461 = "mix.comp.silu"(%4460) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4462 = "mix.module.linear"(%4459) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.35.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4463 = "mix.prim.mul"(%4461, %4462) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4464 = "mix.module.linear"(%4463) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.35.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4465 = "mix.prim.add"(%4464, %4450) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4466 = "mix.comp.weight"() <{param_loc = "transformer.h.36.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4467 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4468 = "mix.prim.pow"(%4465, %4467) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4469 = "mix.comp.mean"(%4468) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4470 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4471 = "mix.prim.add"(%4469, %4470) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4472 = "mix.prim.rsqrt"(%4471) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4473 = "mix.prim.mul"(%4465, %4472) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4474 = "mix.prim.mul"(%4466, %4473) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4475 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4476 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4477 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4478 = "mix.comp.weight"() <{param_loc = "transformer.h.36.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4479 = "mix.prim.transpose"(%4474) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4480 = "mix.prim.transpose"(%4475) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4481 = "mix.prim.reshape"(%4479) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4482 = "mix.prim.matmul"(%4481, %4480) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4483 = "mix.prim.reshape"(%4482) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4484 = "mix.prim.reshape"(%4483) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4485 = "mix.prim.transpose"(%4476) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4486 = "mix.prim.reshape"(%4479) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4487 = "mix.prim.matmul"(%4486, %4485) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4488 = "mix.prim.reshape"(%4487) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4489 = "mix.prim.reshape"(%4488) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4490 = "mix.prim.slice"(%4489) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4491 = "mix.prim.slice"(%4489) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4492 = "mix.prim.reshape"(%4484) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4493 = "mix.prim.reshape"(%4490) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4494 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4495 = "mix.prim.convert"(%4494) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4496 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4497 = "mix.prim.div"(%4495, %4496) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4498 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4499 = "mix.prim.pow"(%4498, %4497) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4500 = "mix.prim.reciprocal"(%4499) : (tensor<80xf16>) -> tensor<80xf16>
    %4501 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4502 = "mix.prim.mul"(%4501, %4500) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4503 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4504 = "mix.prim.unsqueeze"(%4503) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4505 = "mix.prim.permute"(%4504) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4506 = "mix.prim.unsqueeze"(%4502) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4507 = "mix.prim.permute"(%4506) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4508 = "mix.prim.mul"(%4505, %4507) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4509 = "mix.prim.concat"(%4508, %4508) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4510 = "mix.prim.cos"(%4509) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4511 = "mix.prim.slice"(%4510) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4512 = "mix.prim.unsqueeze"(%4511) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4513 = "mix.prim.slice"(%4512) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4514 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4515 = "mix.prim.mul"(%4513, %4514) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4516 = "mix.prim.sin"(%4509) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4517 = "mix.prim.slice"(%4516) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4518 = "mix.prim.unsqueeze"(%4517) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4519 = "mix.prim.slice"(%4518) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4520 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4521 = "mix.prim.mul"(%4519, %4520) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4522 = "mix.prim.slice"(%4515) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4523 = "mix.prim.slice"(%4521) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4524 = "mix.prim.mul"(%4492, %4522) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4525 = "mix.prim.slice"(%4492) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4526 = "mix.prim.slice"(%4492) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4527 = "mix.prim.neg"(%4526) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4528 = "mix.prim.concat"(%4527, %4525) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4529 = "mix.prim.mul"(%4528, %4523) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4530 = "mix.prim.add"(%4524, %4529) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4531 = "mix.prim.mul"(%4493, %4522) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4532 = "mix.prim.slice"(%4493) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4533 = "mix.prim.slice"(%4493) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4534 = "mix.prim.neg"(%4533) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4535 = "mix.prim.concat"(%4534, %4532) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4536 = "mix.prim.mul"(%4535, %4523) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4537 = "mix.prim.add"(%4531, %4536) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4538 = "mix.prim.reshape"(%4530) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4539 = "mix.prim.reshape"(%4537) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4540 = "mix.prim.reshape"(%4538) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4541 = "mix.prim.reshape"(%4539) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4542 = "mix.prim.transpose"(%4540) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4543 = "mix.prim.transpose"(%4541) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4544 = "mix.prim.transpose"(%4543) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4545 = "mix.prim.unsqueeze"(%4542) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4546 = "mix.prim.permute"(%4545) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4547 = "mix.prim.unsqueeze"(%4544) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4548 = "mix.prim.permute"(%4547) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4549 = "mix.prim.permute"(%4546) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4550 = "mix.prim.reshape"(%4549) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4551 = "mix.prim.permute"(%4548) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4552 = "mix.prim.reshape"(%4551) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4553 = "mix.prim.batch_matmul"(%4550, %4552) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4554 = "mix.prim.reshape"(%4553) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4555 = "mix.prim.permute"(%4554) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4556 = "mix.prim.reshape"(%4555) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4557 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4558 = "mix.prim.mul"(%4556, %4557) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4559 = "mix.prim.reshape"(%4558) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4560 = "mix.comp.masked_fill"(%4559, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %4561 = "mix.comp.softmax"(%4560) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4562 = "mix.prim.reshape"(%4561) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4563 = "mix.prim.reshape"(%4491) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4564 = "mix.prim.transpose"(%4563) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4565 = "mix.prim.batch_matmul"(%4562, %4564) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4566 = "mix.prim.reshape"(%4565) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4567 = "mix.prim.permute"(%4566) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4568 = "mix.prim.reshape"(%4567) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4569 = "mix.prim.reshape"(%4568) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4570 = "mix.prim.transpose"(%4477) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4571 = "mix.prim.matmul"(%4569, %4570) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4572 = "mix.prim.add"(%4571, %4478) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4573 = "mix.prim.reshape"(%4572) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4574 = "mix.prim.mul"(%4465, %4573) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4575 = "mix.comp.weight"() <{param_loc = "transformer.h.36.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4576 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4577 = "mix.prim.pow"(%4574, %4576) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4578 = "mix.comp.mean"(%4577) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4579 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4580 = "mix.prim.add"(%4578, %4579) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4581 = "mix.prim.rsqrt"(%4580) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4582 = "mix.prim.mul"(%4574, %4581) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4583 = "mix.prim.mul"(%4575, %4582) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4584 = "mix.module.linear"(%4583) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.36.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4585 = "mix.comp.silu"(%4584) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4586 = "mix.module.linear"(%4583) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.36.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4587 = "mix.prim.mul"(%4585, %4586) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4588 = "mix.module.linear"(%4587) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.36.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4589 = "mix.prim.add"(%4588, %4574) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4590 = "mix.comp.weight"() <{param_loc = "transformer.h.37.input_layernorm.weight"}> : () -> tensor<5120xf16>
    %4591 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4592 = "mix.prim.pow"(%4589, %4591) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4593 = "mix.comp.mean"(%4592) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4594 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4595 = "mix.prim.add"(%4593, %4594) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4596 = "mix.prim.rsqrt"(%4595) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4597 = "mix.prim.mul"(%4589, %4596) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4598 = "mix.prim.mul"(%4590, %4597) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4599 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.query.weight"}> : () -> tensor<5120x5120xf16>
    %4600 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.key_value.weight"}> : () -> tensor<10240x5120xf16>
    %4601 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.dense.weight"}> : () -> tensor<5120x5120xf16>
    %4602 = "mix.comp.weight"() <{param_loc = "transformer.h.37.self_attention.dense.bias"}> : () -> tensor<5120xf16>
    %4603 = "mix.prim.transpose"(%4598) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<1x40x5120xf16>) -> tensor<40x1x5120xf16>
    %4604 = "mix.prim.transpose"(%4599) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4605 = "mix.prim.reshape"(%4603) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4606 = "mix.prim.matmul"(%4605, %4604) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4607 = "mix.prim.reshape"(%4606) <{shape = [40, 1, 5120]}> : (tensor<40x5120xf16>) -> tensor<40x1x5120xf16>
    %4608 = "mix.prim.reshape"(%4607) <{shape = [40, 1, 32, 160]}> : (tensor<40x1x5120xf16>) -> tensor<40x1x32x160xf16>
    %4609 = "mix.prim.transpose"(%4600) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<10240x5120xf16>) -> tensor<5120x10240xf16>
    %4610 = "mix.prim.reshape"(%4603) <{shape = [40, 5120]}> : (tensor<40x1x5120xf16>) -> tensor<40x5120xf16>
    %4611 = "mix.prim.matmul"(%4610, %4609) : (tensor<40x5120xf16>, tensor<5120x10240xf16>) -> tensor<40x10240xf16>
    %4612 = "mix.prim.reshape"(%4611) <{shape = [40, 1, 10240]}> : (tensor<40x10240xf16>) -> tensor<40x1x10240xf16>
    %4613 = "mix.prim.reshape"(%4612) <{shape = [40, 1, 32, 320]}> : (tensor<40x1x10240xf16>) -> tensor<40x1x32x320xf16>
    %4614 = "mix.prim.slice"(%4613) <{dim = 3 : i64, end = 160 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4615 = "mix.prim.slice"(%4613) <{dim = 3 : i64, end = 320 : i64, start = 160 : i64, step = 1 : i64}> : (tensor<40x1x32x320xf16>) -> tensor<40x1x32x160xf16>
    %4616 = "mix.prim.reshape"(%4608) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4617 = "mix.prim.reshape"(%4614) <{shape = [40, 32, -1]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4618 = "mix.prim.constant"() <{value = dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79]> : tensor<80xi32>}> : () -> tensor<80xi32>
    %4619 = "mix.prim.convert"(%4618) <{element_ty = f16}> : (tensor<80xi32>) -> tensor<80xf16>
    %4620 = "mix.prim.constant"() <{value = 160 : i32}> : () -> i32
    %4621 = "mix.prim.div"(%4619, %4620) : (tensor<80xf16>, i32) -> tensor<80xf16>
    %4622 = "mix.prim.constant"() <{value = 3.041600e+04 : f16}> : () -> f16
    %4623 = "mix.prim.pow"(%4622, %4621) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4624 = "mix.prim.reciprocal"(%4623) : (tensor<80xf16>) -> tensor<80xf16>
    %4625 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4626 = "mix.prim.mul"(%4625, %4624) : (f16, tensor<80xf16>) -> tensor<80xf16>
    %4627 = "mix.prim.constant"() <{value = dense<1.000000e+00> : tensor<40xf16>}> : () -> tensor<40xf16>
    %4628 = "mix.prim.unsqueeze"(%4627) <{axis = 1 : i32}> : (tensor<40xf16>) -> tensor<40x1xf16>
    %4629 = "mix.prim.permute"(%4628) <{dims = [0, 1]}> : (tensor<40x1xf16>) -> tensor<40x1xf16>
    %4630 = "mix.prim.unsqueeze"(%4626) <{axis = 1 : i32}> : (tensor<80xf16>) -> tensor<80x1xf16>
    %4631 = "mix.prim.permute"(%4630) <{dims = [1, 0]}> : (tensor<80x1xf16>) -> tensor<1x80xf16>
    %4632 = "mix.prim.mul"(%4629, %4631) : (tensor<40x1xf16>, tensor<1x80xf16>) -> tensor<40x80xf16>
    %4633 = "mix.prim.concat"(%4632, %4632) <{axis = 1 : i64}> : (tensor<40x80xf16>, tensor<40x80xf16>) -> tensor<40x160xf16>
    %4634 = "mix.prim.cos"(%4633) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4635 = "mix.prim.slice"(%4634) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4636 = "mix.prim.unsqueeze"(%4635) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4637 = "mix.prim.slice"(%4636) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4638 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4639 = "mix.prim.mul"(%4637, %4638) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4640 = "mix.prim.sin"(%4633) : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4641 = "mix.prim.slice"(%4640) <{dim = 0 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x160xf16>) -> tensor<40x160xf16>
    %4642 = "mix.prim.unsqueeze"(%4641) <{axis = 1 : i32}> : (tensor<40x160xf16>) -> tensor<40x1x160xf16>
    %4643 = "mix.prim.slice"(%4642) <{dim = 2 : i64, end = 2147483647 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4644 = "mix.prim.constant"() <{value = 1.000000e+00 : f16}> : () -> f16
    %4645 = "mix.prim.mul"(%4643, %4644) : (tensor<40x1x160xf16>, f16) -> tensor<40x1x160xf16>
    %4646 = "mix.prim.slice"(%4639) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4647 = "mix.prim.slice"(%4645) <{dim = 0 : i64, end = 40 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x1x160xf16>) -> tensor<40x1x160xf16>
    %4648 = "mix.prim.mul"(%4616, %4646) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4649 = "mix.prim.slice"(%4616) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4650 = "mix.prim.slice"(%4616) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4651 = "mix.prim.neg"(%4650) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4652 = "mix.prim.concat"(%4651, %4649) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4653 = "mix.prim.mul"(%4652, %4647) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4654 = "mix.prim.add"(%4648, %4653) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4655 = "mix.prim.mul"(%4617, %4646) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4656 = "mix.prim.slice"(%4617) <{dim = 2 : i64, end = 80 : i64, start = 0 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4657 = "mix.prim.slice"(%4617) <{dim = 2 : i64, end = 2147483647 : i64, start = 80 : i64, step = 1 : i64}> : (tensor<40x32x160xf16>) -> tensor<40x32x80xf16>
    %4658 = "mix.prim.neg"(%4657) : (tensor<40x32x80xf16>) -> tensor<40x32x80xf16>
    %4659 = "mix.prim.concat"(%4658, %4656) <{axis = 2 : i64}> : (tensor<40x32x80xf16>, tensor<40x32x80xf16>) -> tensor<40x32x160xf16>
    %4660 = "mix.prim.mul"(%4659, %4647) : (tensor<40x32x160xf16>, tensor<40x1x160xf16>) -> tensor<40x32x160xf16>
    %4661 = "mix.prim.add"(%4655, %4660) : (tensor<40x32x160xf16>, tensor<40x32x160xf16>) -> tensor<40x32x160xf16>
    %4662 = "mix.prim.reshape"(%4654) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4663 = "mix.prim.reshape"(%4661) <{shape = [40, 1, 32, 160]}> : (tensor<40x32x160xf16>) -> tensor<40x1x32x160xf16>
    %4664 = "mix.prim.reshape"(%4662) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4665 = "mix.prim.reshape"(%4663) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4666 = "mix.prim.transpose"(%4664) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4667 = "mix.prim.transpose"(%4665) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4668 = "mix.prim.transpose"(%4667) <{dim1 = 1 : i32, dim2 = 2 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x160x40xf16>
    %4669 = "mix.prim.unsqueeze"(%4666) <{axis = 3 : i32}> : (tensor<32x40x160xf16>) -> tensor<32x40x160x1xf16>
    %4670 = "mix.prim.permute"(%4669) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x1x160xf16>
    %4671 = "mix.prim.unsqueeze"(%4668) <{axis = 3 : i32}> : (tensor<32x160x40xf16>) -> tensor<32x160x40x1xf16>
    %4672 = "mix.prim.permute"(%4671) <{dims = [0, 3, 2, 1]}> : (tensor<32x160x40x1xf16>) -> tensor<32x1x40x160xf16>
    %4673 = "mix.prim.permute"(%4670) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x160xf16>) -> tensor<32x40x160x1xf16>
    %4674 = "mix.prim.reshape"(%4673) <{shape = [32, 40, 160]}> : (tensor<32x40x160x1xf16>) -> tensor<32x40x160xf16>
    %4675 = "mix.prim.permute"(%4672) <{dims = [0, 3, 2, 1]}> : (tensor<32x1x40x160xf16>) -> tensor<32x160x40x1xf16>
    %4676 = "mix.prim.reshape"(%4675) <{shape = [32, 160, 40]}> : (tensor<32x160x40x1xf16>) -> tensor<32x160x40xf16>
    %4677 = "mix.prim.batch_matmul"(%4674, %4676) : (tensor<32x40x160xf16>, tensor<32x160x40xf16>) -> tensor<32x40x40xf16>
    %4678 = "mix.prim.reshape"(%4677) <{shape = [32, 40, 1, 40]}> : (tensor<32x40x40xf16>) -> tensor<32x40x1x40xf16>
    %4679 = "mix.prim.permute"(%4678) <{dims = [0, 1, 3, 2]}> : (tensor<32x40x1x40xf16>) -> tensor<32x40x40x1xf16>
    %4680 = "mix.prim.reshape"(%4679) <{shape = [32, 40, 40]}> : (tensor<32x40x40x1xf16>) -> tensor<32x40x40xf16>
    %4681 = "mix.prim.constant"() <{value = 7.904050e-02 : f16}> : () -> f16
    %4682 = "mix.prim.mul"(%4680, %4681) : (tensor<32x40x40xf16>, f16) -> tensor<32x40x40xf16>
    %4683 = "mix.prim.reshape"(%4682) <{shape = [1, 32, 40, 40]}> : (tensor<32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4684 = "mix.comp.masked_fill"(%4683, %1) <{value = 0xFC00 : f16}> : (tensor<1x32x40x40xf16>, tensor<1x1x5x5xi1>) -> tensor<1x32x40x40xf16>
    %4685 = "mix.comp.softmax"(%4684) <{axis = -1 : si32}> : (tensor<1x32x40x40xf16>) -> tensor<1x32x40x40xf16>
    %4686 = "mix.prim.reshape"(%4685) <{shape = [32, 40, 40]}> : (tensor<1x32x40x40xf16>) -> tensor<32x40x40xf16>
    %4687 = "mix.prim.reshape"(%4615) <{shape = [40, 32, 160]}> : (tensor<40x1x32x160xf16>) -> tensor<40x32x160xf16>
    %4688 = "mix.prim.transpose"(%4687) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<40x32x160xf16>) -> tensor<32x40x160xf16>
    %4689 = "mix.prim.batch_matmul"(%4686, %4688) : (tensor<32x40x40xf16>, tensor<32x40x160xf16>) -> tensor<32x40x160xf16>
    %4690 = "mix.prim.reshape"(%4689) <{shape = [1, 32, 40, 160]}> : (tensor<32x40x160xf16>) -> tensor<1x32x40x160xf16>
    %4691 = "mix.prim.permute"(%4690) <{dims = [0, 2, 1, 3]}> : (tensor<1x32x40x160xf16>) -> tensor<1x40x32x160xf16>
    %4692 = "mix.prim.reshape"(%4691) <{shape = [1, 40, 5120]}> : (tensor<1x40x32x160xf16>) -> tensor<1x40x5120xf16>
    %4693 = "mix.prim.reshape"(%4692) <{shape = [40, 5120]}> : (tensor<1x40x5120xf16>) -> tensor<40x5120xf16>
    %4694 = "mix.prim.transpose"(%4601) <{dim1 = 0 : i32, dim2 = 1 : i32}> : (tensor<5120x5120xf16>) -> tensor<5120x5120xf16>
    %4695 = "mix.prim.matmul"(%4693, %4694) : (tensor<40x5120xf16>, tensor<5120x5120xf16>) -> tensor<40x5120xf16>
    %4696 = "mix.prim.add"(%4695, %4602) : (tensor<40x5120xf16>, tensor<5120xf16>) -> tensor<40x5120xf16>
    %4697 = "mix.prim.reshape"(%4696) <{shape = [1, 40, 5120]}> : (tensor<40x5120xf16>) -> tensor<1x40x5120xf16>
    %4698 = "mix.prim.mul"(%4589, %4697) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4699 = "mix.comp.weight"() <{param_loc = "transformer.h.37.post_attention_layernorm.weight"}> : () -> tensor<5120xf16>
    %4700 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4701 = "mix.prim.pow"(%4698, %4700) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4702 = "mix.comp.mean"(%4701) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4703 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4704 = "mix.prim.add"(%4702, %4703) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4705 = "mix.prim.rsqrt"(%4704) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4706 = "mix.prim.mul"(%4698, %4705) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4707 = "mix.prim.mul"(%4699, %4706) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4708 = "mix.module.linear"(%4707) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.37.mlp.gate_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4709 = "mix.comp.silu"(%4708) : (tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4710 = "mix.module.linear"(%4707) <{dtype = f16, in_feature = 5120 : i32, out_feature = 12288 : i32, params_loc = "transformer.h.37.mlp.up_proj"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x12288xf16>
    %4711 = "mix.prim.mul"(%4709, %4710) : (tensor<1x40x12288xf16>, tensor<1x40x12288xf16>) -> tensor<1x40x12288xf16>
    %4712 = "mix.module.linear"(%4711) <{dtype = f16, has_bias, in_feature = 12288 : i32, out_feature = 5120 : i32, params_loc = "transformer.h.37.mlp.down_proj"}> : (tensor<1x40x12288xf16>) -> tensor<1x40x5120xf16>
    %4713 = "mix.prim.add"(%4712, %4698) : (tensor<1x40x5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4714 = "mix.comp.weight"() <{param_loc = "transformer.ln_f.weight"}> : () -> tensor<5120xf16>
    %4715 = "arith.constant"() <{value = dense<1.000000e+00> : tensor<1xf16>}> : () -> tensor<1xf16>
    %4716 = "mix.prim.pow"(%4713, %4715) : (tensor<1x40x5120xf16>, tensor<1xf16>) -> tensor<1x40x5120xf16>
    %4717 = "mix.comp.mean"(%4716) <{dims = [2 : i32], keepDim = true}> : (tensor<1x40x5120xf16>) -> tensor<1x40x1xf16>
    %4718 = "arith.constant"() <{value = 1.013280e-06 : f16}> : () -> f16
    %4719 = "mix.prim.add"(%4717, %4718) : (tensor<1x40x1xf16>, f16) -> tensor<1x40x1xf16>
    %4720 = "mix.prim.rsqrt"(%4719) : (tensor<1x40x1xf16>) -> tensor<1x40x1xf16>
    %4721 = "mix.prim.mul"(%4713, %4720) : (tensor<1x40x5120xf16>, tensor<1x40x1xf16>) -> tensor<1x40x5120xf16>
    %4722 = "mix.prim.mul"(%4714, %4721) : (tensor<5120xf16>, tensor<1x40x5120xf16>) -> tensor<1x40x5120xf16>
    %4723 = "mix.module.linear"(%4722) <{dtype = f16, in_feature = 5120 : i32, out_feature = 120000 : i32, params_loc = "lm_head.weight"}> : (tensor<1x40x5120xf16>) -> tensor<1x40x120000xf16>
    "func.return"(%4723) : (tensor<1x40x120000xf16>) -> ()
  }) : () -> ()
}) : () -> ()
