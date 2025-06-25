#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @module {
  func.func @main(%arg0: tensor<16x1024xf32>, %arg1: tensor<1024x4096xf32>, %arg2: tensor<4096x1024xf32>) -> tensor<16x1024xf32> {
    %0 = tensor.empty() : tensor<16x1024x4096xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x1024x4096xf32>) -> tensor<16x1024x4096xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<16x1024xf32>, tensor<1024x4096xf32>) outs(%1 : tensor<16x1024x4096xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %12 = arith.mulf %in, %in_3 : f32
      linalg.yield %12 : f32
    } -> tensor<16x1024x4096xf32>
    %3 = tensor.empty() : tensor<16x4096xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<16x4096xf32>) -> tensor<16x4096xf32>
    %5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<16x1024x4096xf32>) outs(%4 : tensor<16x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.addf %out, %in : f32
      linalg.yield %12 : f32
    } -> tensor<16x4096xf32>
    %6 = tensor.empty() : tensor<16x4096x1024xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %7 = linalg.fill ins(%cst_1 : f32) outs(%6 : tensor<16x4096x1024xf32>) -> tensor<16x4096x1024xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel"]} ins(%5, %arg2 : tensor<16x4096xf32>, tensor<4096x1024xf32>) outs(%7 : tensor<16x4096x1024xf32>) {
    ^bb0(%in: f32, %in_3: f32, %out: f32):
      %12 = arith.mulf %in, %in_3 : f32
      linalg.yield %12 : f32
    } -> tensor<16x4096x1024xf32>
    %9 = tensor.empty() : tensor<16x1024xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %10 = linalg.fill ins(%cst_2 : f32) outs(%9 : tensor<16x1024xf32>) -> tensor<16x1024xf32>
    %11 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%8 : tensor<16x4096x1024xf32>) outs(%10 : tensor<16x1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %12 = arith.addf %out, %in : f32
      linalg.yield %12 : f32
    } -> tensor<16x1024xf32>
    return %11 : tensor<16x1024xf32>
  }
}