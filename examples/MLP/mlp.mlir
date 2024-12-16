module {
  ml_program.subgraph private @graph0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %0 = "mix.module.mlp"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %c1_i32 = arith.constant 1.0: f32
    %1 = "mix.prim.add"(%c1_i32, %0) : (f32, tensor<?x?xf32>) -> tensor<?x?xf32>
    ml_program.output %1 : tensor<?x?xf32>
  }
}