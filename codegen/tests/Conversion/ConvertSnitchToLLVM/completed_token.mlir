// RUN: quidditch-opt %s --quidditch-convert-to-llvm | FileCheck %s

// CHECK-LABEL: @test
func.func private @test() -> !dma.token {
  // CHECK: %[[T:.*]] = llvm.mlir.constant(0 : {{.*}})
  // CHECK: return %[[T]]
  %0 = dma.completed_token
  return %0 : !dma.token
}
