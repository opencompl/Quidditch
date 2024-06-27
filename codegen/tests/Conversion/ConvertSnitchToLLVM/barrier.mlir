// RUN: quidditch-opt %s --quidditch-convert-snitch-to-llvm | FileCheck %s

// CHECK-LABEL: @test
func.func @test() {
  // CHECK: call @snrt_cluster_hw_barrier()
  quidditch_snitch.barrier
  return
}
