// RUN: quidditch-opt %s --quidditch-convert-snitch-to-llvm | FileCheck %s

// CHECK-LABEL: @test
func.func @test(%arg0 : !quidditch_snitch.dma_token) {
  // TODO: This should be a call to snrt_dma_wait but is currently bugged.
  // CHECK: call @snrt_dma_wait_all()
  quidditch_snitch.wait_for_dma_transfers %arg0 : !quidditch_snitch.dma_token
  return
}
