// RUN: quidditch-opt %s -p "builtin.module(func.func(quidditch-promote-pads-to-l1))" --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @test_zero_f32(
// CHECK-SAME: %[[A:[[:alnum:]]+]]: tensor<32x32xf32>
func.func @test_zero_f32(%a : tensor<32x32xf32>) -> tensor<33x33xf32> {
  %c = arith.constant 0.0 : f32
  // CHECK: %[[R:.*]], %[[T:.*]] = dma.start_tensor_copy of %[[A]]
  // CHECK-SAME: pad with zero by [1, 1]
  // CHECK: %[[R2:.*]] = dma.wait_for_tensor_copy of %[[A]]
  // CHECK-SAME: to %[[R]]
  // CHECK-SAME: using %[[T]]
  %0 = tensor.pad %a low[0, 0] high[1, 1] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %c : f32
  } : tensor<32x32xf32> to tensor<33x33xf32>
  // CHECK: return %[[R2]]
  return %0 : tensor<33x33xf32>
}

// CHECK-LABEL: @test_poison(
// CHECK-SAME: %[[A:[[:alnum:]]+]]: tensor<32x32xf32>
func.func @test_poison(%a : tensor<32x32xf32>) -> tensor<33x33xf32> {
  %c = ub.poison : f32
  // CHECK: %[[R:.*]], %[[T:.*]] = dma.start_tensor_copy of %[[A]]
  // CHECK-SAME: pad with undef by [1, 1]
  // CHECK: %[[R2:.*]] = dma.wait_for_tensor_copy of %[[A]]
  // CHECK-SAME: to %[[R]]
  // CHECK-SAME: using %[[T]]
  %0 = tensor.pad %a low[0, 0] high[1, 1] {
  ^bb0(%arg0: index, %arg1: index):
    tensor.yield %c : f32
  } : tensor<32x32xf32> to tensor<33x33xf32>
  // CHECK: return %[[R2]]
  return %0 : tensor<33x33xf32>
}
