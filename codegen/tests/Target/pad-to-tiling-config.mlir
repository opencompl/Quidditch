// RUN: quidditch-opt %s -p "builtin.module(func.func(quidditch-pad-to-tiling-config))" | FileCheck %s

// CHECK-LABEL: @test(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @test(%arg0 : tensor<33xf32>) -> tensor<33xf32> attributes {
  hal.executable.target = #hal.executable.target<"", "", {compute_cores = 8 : i32}>
} {
  // CHECK: %[[POISON:.*]] = ub.poison
  // CHECK: %[[PAD:.*]] = tensor.pad %[[ARG0]]
  // CHECK-SAME: low[0]
  // CHECK-SAME: high[2]
  // CHECK: tensor.yield %[[POISON]]
  // CHECK: %[[EMPTY:.*]] = tensor.empty
  %1 = tensor.empty() : tensor<33xf32>
  // CHECK: %[[ABS:.*]] = linalg.abs ins(%[[PAD]] :
  // CHECK-SAME: outs(%[[EMPTY]] :
  %0 = linalg.abs ins(%arg0 : tensor<33xf32>) outs(%1 : tensor<33xf32>) -> tensor<33xf32>
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ABS]][0] [33]
  // CHECK: return %[[SLICE]]
  return %0 : tensor<33xf32>
}

// CHECK-LABEL: @test_contraction_activation(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
func.func @test_contraction_activation(%arg0 : tensor<1x33xf32>, %arg1 : tensor<11x33xf32>) -> tensor<1x11xf32> attributes {
  hal.executable.target = #hal.executable.target<"", "", {compute_cores = 8 : i32}>
} {
  // CHECK: %[[POISON:.*]] = ub.poison
  // CHECK: %[[EMPTY:.*]] = tensor.empty
  // CHECK: %[[PAD:.*]] = tensor.pad %[[ARG1]]
  // CHECK-SAME: low[0, 0]
  // CHECK-SAME: high[1, 0]
  // CHECK: tensor.yield %[[POISON]]
  %1 = tensor.empty() : tensor<1x11xf32>
  // CHECK: %[[MATMUL:.*]] = linalg.matmul_transpose_b ins(%[[ARG0]], %[[PAD]] :
  // CHECK-SAME: outs(%[[EMPTY]] :
  %0 = linalg.matmul_transpose_b ins(%arg0, %arg1 : tensor<1x33xf32>, tensor<11x33xf32>) outs(%1 : tensor<1x11xf32>) -> tensor<1x11xf32>
  // CHECK: %[[ABS:.*]] = linalg.abs ins(%[[MATMUL]] :
  // CHECK-SAME: outs(%[[MATMUL]] :
  %2 = linalg.abs ins(%0 : tensor<1x11xf32>) outs(%0 : tensor<1x11xf32>) -> tensor<1x11xf32>
  // CHECK: %[[SLICE:.*]] = tensor.extract_slice %[[ABS]][0, 0] [1, 11]
  // CHECK: return %[[SLICE]]
  return %2 : tensor<1x11xf32>
}
