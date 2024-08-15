// RUN: quidditch-opt %s -p "builtin.module(func.func(quidditch-form-microkernels))" --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @linalgs_in_scf
func.func @linalgs_in_scf(%cond : i1) -> tensor<32xf32> {
  %cst0 = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32xf32>
  // CHECK: scf.if
  %result2 = scf.if %cond -> tensor<32xf32> {
    // CHECK: %[[RAW:.*]] = quidditch_snitch.tensor.microkernel
    // CHECK-SAME: {
    // CHECK: %[[RES:.*]] = linalg.fill
    // CHECK-NEXT: microkernel_yield %[[RES]]
    // CHECK-NEXT: }
    // CHECK: %[[SYNC:.*]] = quidditch_snitch.sync_tensor %[[RAW]]
    %result = linalg.fill ins(%cst0 : f32) outs(%empty : tensor<32xf32>) -> tensor<32xf32>
    // CHECK: yield %[[SYNC]]
    scf.yield %result : tensor<32xf32>
  } else {
    scf.yield %empty : tensor<32xf32>
  }
  return %result2 : tensor<32xf32>
}
