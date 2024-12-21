module {
  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
  func.func private @graph0(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>){
    %0 = "mix.module.mlp"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    %c1_i32 = arith.constant 1.0: f32
    %1 = "mix.prim.add"(%c1_i32, %0) : (f32, tensor<2x2xf32>) -> tensor<2x2xf32>
    %tensor_unranked = tensor.cast %1 : tensor<2x2xf32> to tensor<*xf32>
    call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
    return
  }
}

