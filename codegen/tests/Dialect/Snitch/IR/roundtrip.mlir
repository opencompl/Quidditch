// RUN: quidditch-opt %s --verify-roundtrip

func.func @test(%arg0 : memref<f64>) {
  quidditch_snitch.memref.microkernel(%arg0) : memref<f64> {
  ^bb0(%arg1 : memref<f64>):

  }
  quidditch_snitch.wait_for_dma_transfers
  return
}

func.func @test3(%arg0 : tensor<?x4xf64>) -> (tensor<?x4xf64>, !quidditch_snitch.dma_token) {
  %0:2 = quidditch_snitch.start_tensor_copy %arg0 to L1 : tensor<?x4xf64> -> tensor<?x4xf64>
  return %0#0, %0#1 : tensor<?x4xf64>, !quidditch_snitch.dma_token
}
