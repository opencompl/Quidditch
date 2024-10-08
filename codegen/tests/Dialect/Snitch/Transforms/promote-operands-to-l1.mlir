// RUN: quidditch-opt %s -p "builtin.module(func.func(quidditch-promote-operands-to-l1))" --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @test(
// CHECK-SAME: %[[A:[[:alnum:]]+]]: tensor<32x32xf32>
// CHECK-SAME: %[[B:[[:alnum:]]+]]: tensor<32x32xf32>
func.func @test(%a : tensor<32x32xf32>, %b : tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[E:.*]] = bufferization.alloc_tensor
  %e = bufferization.alloc_tensor() : tensor<32x32xf32>
  // CHECK: %[[A1:.*]], %[[TOKEN:.*]] = dma.start_tensor_copy of %[[A]] to #quidditch_snitch.l1_encoding
  // CHECK: %[[A2:.*]] = dma.wait_for_tensor_copy of %[[A]]
  // CHECK-SAME: to %[[A1]]
  // CEHCK-SAME: using %[[TOKEN]]
  // CHECK: %[[B1:.*]], %[[TOKEN:.*]] = dma.start_tensor_copy of %[[B]] to #quidditch_snitch.l1_encoding
  // CHECK: %[[B2:.*]] = dma.wait_for_tensor_copy of %[[B]]
  // CHECK-SAME: to %[[B1]]
  // CHECK-SAME: using %[[TOKEN]]
  // CHECK: %[[E1:.*]], %[[TOKEN:.*]] = dma.start_tensor_copy of %[[E]] to #quidditch_snitch.l1_encoding
  // CHECK: %[[E2:.*]] = dma.wait_for_tensor_copy of %[[E]]
  // CHECK-SAME: to %[[E1]]
  // CHECK-SAME: using %[[TOKEN]]
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
