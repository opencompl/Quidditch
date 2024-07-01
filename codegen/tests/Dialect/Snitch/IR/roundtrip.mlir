// RUN: quidditch-opt %s --verify-roundtrip

func.func @test(%arg0 : memref<f64>) {
  quidditch_snitch.memref.microkernel(%arg0) : memref<f64> {
  ^bb0(%arg1 : memref<f64>):

  }
  return
}

func.func @test3(%arg0 : tensor<?x4xf64>) -> tensor<?x4xf64> {
  %0 = quidditch_snitch.copy_tensor %arg0 to L1 : tensor<?x4xf64>
  return %0 : tensor<?x4xf64>
}
