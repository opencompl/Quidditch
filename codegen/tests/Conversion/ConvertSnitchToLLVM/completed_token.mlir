// RUN: quidditch-opt %s --quidditch-convert-to-llvm | FileCheck %s

// CHECK-LABEL: @test
func.func private @test() -> !quidditch_snitch.dma_token {
  // CHECK: %[[T:.*]] = llvm.mlir.constant(0 : {{.*}})
  // CHECK: return %[[T]]
  %0 = quidditch_snitch.completed_token
  return %0 : !quidditch_snitch.dma_token
}
