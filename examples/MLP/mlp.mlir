module {
  ml_program.subgraph private @graph0(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = "mix.MLP"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ml_program.output %0 : tensor<32x32xf32>
  }
}