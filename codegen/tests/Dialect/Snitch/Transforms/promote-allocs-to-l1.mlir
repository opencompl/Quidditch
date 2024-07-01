// RUN: quidditch-opt %s -p "builtin.module(func.func(quidditch-promote-allocs-to-l1))" | FileCheck %s

// CHECK-LABEL: @test(
// CHECK-SAME: %[[A:[[:alnum:]]+]]: tensor<32x32xf32>
// CHECK-SAME: %[[B:[[:alnum:]]+]]: tensor<32x32xf32>
func.func @test(%a : tensor<32x32xf32>, %b : tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[E:.*]] = bufferization.alloc_tensor() {memory_space = #quidditch_snitch.l1_encoding}
  %e = bufferization.alloc_tensor() : tensor<32x32xf32>
  // CHECK: linalg.matmul ins(%[[A]], %[[B]] : {{.*}}) outs(%[[E]] : {{.*}})
  %r = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%e : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %r : tensor<32x32xf32>
}
