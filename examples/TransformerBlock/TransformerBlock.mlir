module {
  ml_program.subgraph private @TransformerBlock(%arg0: tensor<1x5x5120xf32>, %arg1: tensor<1x5x5120xf32>) -> tensor<1x5x5120xf32> {
    %0 = "mix.module.rmsnorm"(%arg0) <{eps = 1.000000e-05 : f64, hidden_size = 5120 : i64}> : (tensor<1x5x5120xf32>) -> tensor<1x5x5120xf32>
    %1 = "mix.module.self_attn"(%0, %arg0, %arg1) : (tensor<1x5x5120xf32>, tensor<1x5x5120xf32>, tensor<1x5x5120xf32>) -> tensor<1x5x5120xf32>
    %2 = "mix.module.rmsnorm"(%1) <{eps = 1.000000e-05 : f64, hidden_size = 5120 : i64}> : (tensor<1x5x5120xf32>) -> tensor<1x5x5120xf32>
    %3 = "mix.module.mlp"(%2, %1) : (tensor<1x5x5120xf32>, tensor<1x5x5120xf32>) -> tensor<1x5x5120xf32>
    ml_program.output %3 : tensor<1x5x5120xf32>
  }
}