module {
  func.func private @printMemrefF32(%ptr : tensor<*xf32>)
  func.func private @graph0(%arg0: tensor<2x2xf32>) {
    %0 = "mix.module.rmsnorm"(%arg0) <{eps = 9.9999999999999995E-7 : f64, hidden_size = 2 : i64}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
    %tensor_unranked = tensor.cast %0 : tensor<2x2xf32> to tensor<*xf32>
    call @printMemrefF32(%tensor_unranked) : (tensor<*xf32>) -> ()
    return
  }
}