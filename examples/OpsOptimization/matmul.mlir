func.func private @printMemrefF32(memref<*xf32>)
func.func private @rtclock() -> f64

func.func @batch_matmul(%arg0: memref<?x?x?xf32>, %arg1: memref<?x?x?xf32>, %arg2: memref<?x?x?xf32>) {
  %t_start = call @rtclock() : () -> f64
  linalg.batch_matmul 
    ins(%arg0, %arg1 : memref<?x?x?xf32>, memref<?x?x?xf32>) 
    outs(%arg2 : memref<?x?x?xf32>)
  %t_end = call @rtclock() : () -> f64
  %time = arith.subf %t_end, %t_start : f64
  %printed_output = memref.cast %arg2 : memref<?x?x?xf32> to memref<*xf32>
  call @printMemrefF32(%printed_output) : (memref<*xf32>) -> ()
  vector.print %time : f64
  return
}

func.func @alloc_f32(%arg0: index, %arg1: index, %arg2: index, %arg4: f32) -> memref<?x?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.alloc(%arg0, %arg1, %arg2) : memref<?x?x?xf32>
  scf.for %idx0 = %c0 to %arg0 step %c1 {
    scf.for %idx1 = %c0 to %arg1 step %c1 {
      scf.for %idx2 = %c0 to %arg2 step %c1 {
        memref.store %arg4, %0[%idx0, %idx1, %idx2] : memref<?x?x?xf32>
      }
    }
  }
  return %0 : memref<?x?x?xf32>
}

func.func @main(){
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c576 = arith.constant 576 : index
  %c1024 = arith.constant 1024 : index
  %c1000 = arith.constant 1000 : index
  %f0 = arith.constant 0.0 : f32
  %f2 = arith.constant 2.0 : f32
  %f3 = arith.constant 3.0 : f32

  %m0 = call @alloc_f32(%c1, %c1, %c576, %f2) : (index, index, index, f32) -> memref<?x?x?xf32>
  %m1 = call @alloc_f32(%c1, %c576, %c1024, %f3) : (index, index, index, f32) -> memref<?x?x?xf32>
  %m2 = call @alloc_f32(%c1, %c1, %c1024, %f0) : (index, index, index, f32) -> memref<?x?x?xf32>

  call @batch_matmul(%m0, %m1, %m2) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()

  %m3 = call @alloc_f32(%c1, %c1, %c1024, %f2) : (index, index, index, f32) -> memref<?x?x?xf32>
  %m4 = call @alloc_f32(%c1, %c1024, %c1000, %f3) : (index, index, index, f32) -> memref<?x?x?xf32>
  %m5 = call @alloc_f32(%c1, %c1, %c1000, %f0) : (index, index, index, f32) -> memref<?x?x?xf32>

  call @batch_matmul(%m3, %m4, %m5) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()

  return
}