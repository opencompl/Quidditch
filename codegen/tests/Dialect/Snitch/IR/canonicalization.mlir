// RUN: quidditch-opt %s --canonicalize --split-input-file --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @dead_result
func.func @dead_result() {
  // CHECK: quidditch_snitch.memref.microkernel() : () -> ()
  %0 = quidditch_snitch.memref.microkernel() : () -> i32 {
    %c = arith.constant 1 : i32
    quidditch_snitch.microkernel_yield %c : i32
  }
  return
}

// CHECK-LABEL: @sink_constants
func.func @sink_constants() -> i32 {
  %c = arith.constant 1 : i32
  // CHECK: quidditch_snitch.memref.microkernel() : () -> i32
  %0 = quidditch_snitch.memref.microkernel(%c) : (i32) -> i32 {
  ^bb0(%arg0 : i32):
    // CHECK-NEXT: %[[C:.*]] = arith.constant
    // CHECK-NEXT: "test.transform"(%[[C]])
    %1 = "test.transform"(%arg0) : (i32) -> i32
    quidditch_snitch.microkernel_yield %1 : i32
  }
  return %0 : i32
}

// CHECK-LABEL: @invariant_result
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @invariant_result(%arg0 : i32) -> i32 {
  %0 = quidditch_snitch.memref.microkernel(%arg0) : (i32) -> i32 {
  ^bb0(%arg1 : i32):
    quidditch_snitch.microkernel_yield %arg1 : i32
  }
  // CHECK: return %[[ARG0]]
  return %0 : i32
}

// CHECK-LABEL: @dead_argument
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @dead_argument(%arg0 : i32) {
  // CHECK: quidditch_snitch.memref.microkernel()
  quidditch_snitch.memref.microkernel(%arg0) : (i32) -> () {
  ^bb0(%arg1 : i32):
    quidditch_snitch.microkernel_yield
  }
  return
}

// CHECK-LABEL: @identical_argument
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @identical_argument(%arg0 : i32) -> i32 {
  // CHECK: quidditch_snitch.memref.microkernel(%[[ARG0]])
  %0 = quidditch_snitch.memref.microkernel(%arg0, %arg0) : (i32, i32) -> i32 {
  ^bb0(%arg1 : i32, %arg2 : i32):
    // CHECK: "test.transform"(%[[ARG1:.*]], %[[ARG1]])
    %1 = "test.transform"(%arg1, %arg2) : (i32, i32) -> i32
    quidditch_snitch.microkernel_yield %1 : i32
  }
  return %0 : i32
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
