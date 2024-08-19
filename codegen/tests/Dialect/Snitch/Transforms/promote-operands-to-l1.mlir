// RUN: quidditch-opt %s -p "builtin.module(func.func(quidditch-promote-operands-to-l1))" --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @test(
// CHECK-SAME: %[[A:[[:alnum:]]+]]: tensor<32x32xf32>
// CHECK-SAME: %[[B:[[:alnum:]]+]]: tensor<32x32xf32>
func.func @test(%a : tensor<32x32xf32>, %b : tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[E:.*]] = bufferization.alloc_tensor
  %e = bufferization.alloc_tensor() : tensor<32x32xf32>
  // CHECK: %[[A1:.*]], %[[TOKEN:.*]] = quidditch_snitch.start_tensor_copy %[[A]] to L1
  // CHECK: %[[A2:.*]] = quidditch_snitch.wait_for_tensor_copy of %[[A]] to %[[A1]] using %[[TOKEN]]
  // CHECK: %[[B1:.*]], %[[TOKEN:.*]] = quidditch_snitch.start_tensor_copy %[[B]] to L1
  // CHECK: %[[B2:.*]] = quidditch_snitch.wait_for_tensor_copy of %[[B]] to %[[B1]] using %[[TOKEN]]
  // CHECK: %[[E1:.*]], %[[TOKEN:.*]] = quidditch_snitch.start_tensor_copy %[[E]] to L1
  // CHECK: %[[E2:.*]] = quidditch_snitch.wait_for_tensor_copy of %[[E]] to %[[E1]] using %[[TOKEN]]
  // CHECK: linalg.matmul ins(%[[A2]], %[[B2]] : {{.*}}) outs(%[[E2]] : {{.*}})
  %r = linalg.matmul ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>) outs(%e : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %r : tensor<32x32xf32>
}

// CHECK-LABEL: @test_dominance(
// CHECK-SAME: %[[A:[[:alnum:]]+]]
func.func @test_dominance(%a : tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: "test.use"(%[[A]])
  "test.use"(%a) : (tensor<32x32xf32>) -> ()
  %e = bufferization.alloc_tensor() : tensor<32x32xf32>
  %r = linalg.abs ins(%a : tensor<32x32xf32>) outs(%e : tensor<32x32xf32>) -> tensor<32x32xf32>
  return %r : tensor<32x32xf32>
}
