module {
  ml_program.subgraph private @graph0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "mix.module.gelu"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
    ml_program.output %0 : tensor<2x2xf32>
  }
}