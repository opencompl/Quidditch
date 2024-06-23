// RUN: quidditch-opt %s --verify-roundtrip

func.func @test(%arg0 : memref<f64>) {
  quidditch_snitch.memref.microkernel(%arg0) : (memref<f64>) -> () {
  ^bb0(%arg1 : memref<f64>):
    quidditch_snitch.microkernel_yield
  }
  return
}

func.func @test2(%arg0 : tensor<f64>) {
  %0 = quidditch_snitch.tensor.microkernel -> tensor<f64> {
    quidditch_snitch.microkernel_yield %arg0 : tensor<f64>
  }
  return
}
