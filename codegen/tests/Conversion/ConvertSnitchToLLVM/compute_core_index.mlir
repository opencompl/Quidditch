// RUN: quidditch-opt %s --quidditch-convert-to-llvm | FileCheck %s

// CHECK-LABEL: @test
func.func private @test() -> index {
  // CHECK: call @snrt_cluster_core_idx()
  %0 = quidditch_snitch.compute_core_index
  return %0 : index
}
