// RUN: quidditch-opt %s --quidditch-convert-snitch-to-llvm | FileCheck %s

// CHECK-LABEL: @test
func.func @test() -> !quidditch_snitch.dma_token {
  // CHECK: %[[T:.*]] = llvm.mlir.constant(0 : {{.*}})
  // CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[T]]
  // CHECK: return %[[C]]
  %0 = quidditch_snitch.completed_token
  return %0 : !quidditch_snitch.dma_token
}
