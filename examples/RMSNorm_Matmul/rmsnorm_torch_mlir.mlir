module @module {
  func.func @main(%arg0: !torch.vtensor<[1024,1024],f32>, %arg1: !torch.vtensor<[1024],f32>, %arg2: !torch.vtensor<[1024,4096],f32>) -> !torch.vtensor<[1024,4096],f32> attributes {torch.assume_strict_symbolic_shapes} {
    %int6 = torch.constant.int 6
    %0 = torch.prims.convert_element_type %arg0, %int6 : !torch.vtensor<[1024,1024],f32>, !torch.int -> !torch.vtensor<[1024,1024],f32>
    %int2 = torch.constant.int 2
    %1 = torch.aten.pow.Tensor_Scalar %0, %int2 : !torch.vtensor<[1024,1024],f32>, !torch.int -> !torch.vtensor<[1024,1024],f32>
    %int-1 = torch.constant.int -1
    %2 = torch.prim.ListConstruct %int-1 : (!torch.int) -> !torch.list<int>
    %true = torch.constant.bool true
    %none = torch.constant.none
    %3 = torch.aten.mean.dim %1, %2, %true, %none : !torch.vtensor<[1024,1024],f32>, !torch.list<int>, !torch.bool, !torch.none -> !torch.vtensor<[1024,1],f32>
    %float9.999990e-07 = torch.constant.float 9.9999999999999995E-7
    %int1 = torch.constant.int 1
    %4 = torch.aten.add.Scalar %3, %float9.999990e-07, %int1 : !torch.vtensor<[1024,1],f32>, !torch.float, !torch.int -> !torch.vtensor<[1024,1],f32>
    %5 = torch.aten.rsqrt %4 : !torch.vtensor<[1024,1],f32> -> !torch.vtensor<[1024,1],f32>
    %6 = torch.aten.mul.Tensor %0, %5 : !torch.vtensor<[1024,1024],f32>, !torch.vtensor<[1024,1],f32> -> !torch.vtensor<[1024,1024],f32>
    %int6_0 = torch.constant.int 6
    %7 = torch.prims.convert_element_type %6, %int6_0 : !torch.vtensor<[1024,1024],f32>, !torch.int -> !torch.vtensor<[1024,1024],f32>
    %8 = torch.aten.mul.Tensor %arg1, %7 : !torch.vtensor<[1024],f32>, !torch.vtensor<[1024,1024],f32> -> !torch.vtensor<[1024,1024],f32>
    %9 = torch.aten.matmul %8, %arg2 : !torch.vtensor<[1024,1024],f32>, !torch.vtensor<[1024,4096],f32> -> !torch.vtensor<[1024,4096],f32>
    return %9 : !torch.vtensor<[1024,4096],f32>
  }
}