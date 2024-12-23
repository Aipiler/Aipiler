"builtin.module"() ({
  "func.func"() <{function_type = (tensor<*xf32>) -> (), sym_name = "printMemrefF32", sym_visibility = "private"}> ({
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "self_attention", sym_visibility = "private"}> ({
    %0 = "mix.prim.weight"() <{param_loc = "hidden_states"}> : () -> tensor<1x5x5120xf32>
    %1 = "mix.prim.weight"() <{param_loc = "residual"}> : () -> tensor<1x5x5120xf32>
    %2 = "mix.prim.weight"() <{param_loc = "attention_mask"}> : () -> tensor<1x1x5x5xf32>
    %3 = "mix.prim.permute"(%0) <{dims = [1 : i32, 0 : i32]}> : (tensor<1x5x5120xf32>) -> tensor<1x5x5120xf32>
    %4 = "mix.prim.weight"() <{param_loc = "Self_attn.query.weight"}> : () -> tensor<5120x5120xf32>
    %5 = "mix.prim.permute"(%4) <{dims = [1 : i32, 0 : i32]}> : (tensor<5120x5120xf32>) -> tensor<5120x5120xf32>
    %6 = "mix.prim.reshape"(%3) <{shape = [5 : i32, 5120 : i32]}> : (tensor<1x5x5120xf32>) -> tensor<5x5120xf32>
    %7 = "mix.prim.matmul"(%6, %5) : (tensor<5x5120xf32>, tensor<5120x5120xf32>) -> tensor<5x5120xf32>
  }) : () -> ()
}) : () -> ()