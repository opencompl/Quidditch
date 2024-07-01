// RUN: quidditch-opt %s --canonicalize --split-input-file --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @sink_constants
func.func @sink_constants() {
  %c = arith.constant 1 : i32
  // CHECK: quidditch_snitch.memref.microkernel()
  quidditch_snitch.memref.microkernel(%c) : i32 {
  ^bb0(%arg0 : i32):
    // CHECK-NEXT: %[[C:.*]] = arith.constant
    // CHECK-NEXT: "test.transform"(%[[C]])
    "test.transform"(%arg0) : (i32) -> ()
  }
  return
}

// CHECK-LABEL: @dead_argument
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @dead_argument(%arg0 : i32) {
  // CHECK: quidditch_snitch.memref.microkernel()
  quidditch_snitch.memref.microkernel(%arg0) : i32 {
  ^bb0(%arg1 : i32):

  }
  return
}

// CHECK-LABEL: @identical_argument
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @identical_argument(%arg0 : i32) {
  // CHECK: quidditch_snitch.memref.microkernel(%[[ARG0]])
  quidditch_snitch.memref.microkernel(%arg0, %arg0) : i32, i32 {
  ^bb0(%arg1 : i32, %arg2 : i32):
    // CHECK: "test.transform"(%[[ARG1:.*]], %[[ARG1]])
    "test.transform"(%arg1, %arg2) : (i32, i32) -> ()
  }
  return
}

// CHECK-LABEL: @double_copy
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @double_copy(%arg0 : tensor<32xf64>) -> tensor<32xf64> {
  // CHECK-NEXT: %[[R:.*]] = quidditch_snitch.copy_tensor %[[ARG0]] to L1
  %0 = quidditch_snitch.copy_tensor %arg0 to L3 : tensor<32xf64>
  %1 = quidditch_snitch.copy_tensor %0 to L1 : tensor<32xf64>
  // CHECK-NEXT: return %[[R]]
  return %1 : tensor<32xf64>
}
