// RUN: quidditch-opt %s -p "builtin.module(func.func(quidditch-promote-to-l1))" | FileCheck %s

// CHECK-LABEL: @test(
// CHECK-SAME: %[[A:[[:alnum:]]+]]: tensor<32x32xf32>
// CHECK-SAME: %[[B:[[:alnum:]]+]]: tensor<32x32xf32>
func.func @test(%a : tensor<32x32xf32>, %b : tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[E:.*]] = bufferization.alloc_tensor() {memory_space = #quidditch_snitch.l1_encoding}
  %e = bufferization.alloc_tensor() : tensor<32x32xf32>
  // CHECK: %[[A2:.*]] = quidditch_snitch.copy_l1_tensor %[[A]] to L1
  // CHECK: %[[B2:.*]] = quidditch_snitch.copy_l1_tensor %[[B]] to L1
  // CHECK: %[[E2:.*]] = quidditch_snitch.copy_l1_tensor %[[E]] to L1
  // CHECK: %[[R:.*]] = quidditch_snitch.tensor.microkernel
  %r = quidditch_snitch.tensor.microkernel -> tensor<32x32xf32> {
    // CHECK: linalg.matmul ins(%[[A2]], %[[B2]] : {{.*}}) outs(%[[E2]] : {{.*}})
    %r2 = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%e : tensor<32x32xf32>) -> tensor<32x32xf32>
    quidditch_snitch.microkernel_yield %r2 : tensor<32x32xf32>
  }
  // CHECK: %[[R2:.*]] = quidditch_snitch.copy_l1_tensor %[[R]] from L1
  // CHECK: return %[[R2]]
  return %r : tensor<32x32xf32>
}
