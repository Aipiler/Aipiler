#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0) -> (d0)>
#map4 = affine_map<(d0) -> ()>
#map5 = affine_map<(d0, d1) -> (d1, d0)>
#map6 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map7 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map9 = affine_map<(d0, d1, d2) -> (d0, d2)>
module @module {
  func.func @main(%arg0: tensor<1024x1024xf32>, %arg1: tensor<1024xf32>, %arg2: tensor<1024x4096xf32>) -> tensor<1024x4096xf32> {
    %cst = arith.constant 9.99999997E-7 : f32
    %cst_0 = arith.constant 1.024000e+03 : f32
    %cst_1 = arith.constant 2.000000e+00 : f32
    %0 = tensor.empty() : tensor<1024x1024xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %cst_1 : tensor<1024x1024xf32>, f32) outs(%1 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %27 = math.powf %in, %in_11 : f32
      linalg.yield %27 : f32
    } -> tensor<1024x1024xf32>
    %3 = tensor.empty() : tensor<1024xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %4 = linalg.fill ins(%cst_3 : f32) outs(%3 : tensor<1024xf32>) -> tensor<1024xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map2], iterator_types = ["parallel", "reduction"]} ins(%2 : tensor<1024x1024xf32>) outs(%4 : tensor<1024xf32>) {
    ^bb0(%in: f32, %out: f32):
      %27 = arith.addf %out, %in : f32
      linalg.yield %27 : f32
    } -> tensor<1024xf32>
    %6 = tensor.empty() : tensor<1024xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %7 = linalg.fill ins(%cst_4 : f32) outs(%6 : tensor<1024xf32>) -> tensor<1024xf32>
    %8 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel"]} ins(%5, %cst_0 : tensor<1024xf32>, f32) outs(%7 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %27 = arith.divf %in, %in_11 : f32
      linalg.yield %27 : f32
    } -> tensor<1024xf32>
    %9 = tensor.empty() : tensor<1024xf32>
    %cst_5 = arith.constant 0.000000e+00 : f32
    %10 = linalg.fill ins(%cst_5 : f32) outs(%9 : tensor<1024xf32>) -> tensor<1024xf32>
    %11 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel"]} ins(%8, %cst : tensor<1024xf32>, f32) outs(%10 : tensor<1024xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %27 = arith.addf %in, %in_11 : f32
      linalg.yield %27 : f32
    } -> tensor<1024xf32>
    %12 = tensor.empty() : tensor<1024xf32>
    %cst_6 = arith.constant 0.000000e+00 : f32
    %13 = linalg.fill ins(%cst_6 : f32) outs(%12 : tensor<1024xf32>) -> tensor<1024xf32>
    %14 = linalg.elemwise_unary {cast = #linalg.type_fn<cast_signed>, fun = #linalg.unary_fn<rsqrt>} ins(%11 : tensor<1024xf32>) outs(%13 : tensor<1024xf32>) -> tensor<1024xf32>
    %15 = tensor.empty() : tensor<1024x1024xf32>
    %cst_7 = arith.constant 0.000000e+00 : f32
    %16 = linalg.fill ins(%cst_7 : f32) outs(%15 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %17 = linalg.generic {indexing_maps = [#map, #map2, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %14 : tensor<1024x1024xf32>, tensor<1024xf32>) outs(%16 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %27 = arith.mulf %in, %in_11 : f32
      linalg.yield %27 : f32
    } -> tensor<1024x1024xf32>
    %18 = tensor.empty() : tensor<1024x1024xf32>
    %cst_8 = arith.constant 0.000000e+00 : f32
    %19 = linalg.fill ins(%cst_8 : f32) outs(%18 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %20 = linalg.generic {indexing_maps = [#map2, #map5, #map5], iterator_types = ["parallel", "parallel"]} ins(%arg1, %17 : tensor<1024xf32>, tensor<1024x1024xf32>) outs(%19 : tensor<1024x1024xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %27 = arith.mulf %in, %in_11 : f32
      linalg.yield %27 : f32
    } -> tensor<1024x1024xf32>
    %21 = tensor.empty() : tensor<1024x1024x4096xf32>
    %cst_9 = arith.constant 0.000000e+00 : f32
    %22 = linalg.fill ins(%cst_9 : f32) outs(%21 : tensor<1024x1024x4096xf32>) -> tensor<1024x1024x4096xf32>
    %23 = linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20, %arg2 : tensor<1024x1024xf32>, tensor<1024x4096xf32>) outs(%22 : tensor<1024x1024x4096xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %27 = arith.mulf %in, %in_11 : f32
      linalg.yield %27 : f32
    } -> tensor<1024x1024x4096xf32>
    %24 = tensor.empty() : tensor<1024x4096xf32>
    %cst_10 = arith.constant 0.000000e+00 : f32
    %25 = linalg.fill ins(%cst_10 : f32) outs(%24 : tensor<1024x4096xf32>) -> tensor<1024x4096xf32>
    %26 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "reduction", "parallel"]} ins(%23 : tensor<1024x1024x4096xf32>) outs(%25 : tensor<1024x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %27 = arith.addf %out, %in : f32
      linalg.yield %27 : f32
    } -> tensor<1024x4096xf32>
    return %26 : tensor<1024x4096xf32>
  }
}