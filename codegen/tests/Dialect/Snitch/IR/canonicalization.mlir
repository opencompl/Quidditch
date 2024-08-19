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

// CHECK-LABEL: @wait_gets_removed
func.func @wait_gets_removed() {
  // CHECK-NEXT: return
  %0 = quidditch_snitch.completed_token
  quidditch_snitch.wait_for_dma_transfers %0 : !quidditch_snitch.dma_token
  return
}

// CHECK-LABEL: @noop_transfer
func.func @noop_transfer(%arg0 : memref<?xf32>) -> !quidditch_snitch.dma_token {
  // CHECK-NEXT: %[[R:.*]] = quidditch_snitch.completed_token
  // CHECK-NEXT: return %[[R]]
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<?xf32> to %arg0 : memref<?xf32>
  return %0 : !quidditch_snitch.dma_token
}

// CHECK-LABEL: @pipeline_dead_block_arg(
func.func @pipeline_dead_block_arg(%tensor : tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // CHECK: pipeline
  // CHECK-SAME: {
  quidditch_snitch.pipeline %c0 to %c10 step %c1 {
  // CHECK: ^{{.*}}(%{{.*}}: index):
  ^bb0(%iv: index):
    "test.test"() : () -> ()
    quidditch_snitch.pipeline_yield %tensor : tensor<?xf32>
  }, {
  // CHECK: ^{{.*}}(%{{.*}}: index):
  ^bb0(%iv: index, %arg0: tensor<?xf32>):
    quidditch_snitch.pipeline_yield
  }
  return
}

// CHECK-LABEL: @pipeline_invariant(
func.func @pipeline_invariant(%tensor : tensor<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  // CHECK: pipeline
  // CHECK-SAME: {
  quidditch_snitch.pipeline %c0 to %c10 step %c1 {
  // CHECK: ^{{.*}}(%{{.*}}: index):
  ^bb0(%iv: index):
    quidditch_snitch.pipeline_yield %tensor : tensor<?xf32>
  }, {
  // CHECK: ^{{.*}}(%{{.*}}: index):
  ^bb0(%iv: index, %arg0: tensor<?xf32>):
    "test.test"(%arg0) : (tensor<?xf32>) -> ()
    quidditch_snitch.pipeline_yield
  }
  return
}

// CHECK-LABEL: @tensor_wait_gets_removed
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
func.func @tensor_wait_gets_removed(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: return %[[ARG1]]
  %t = quidditch_snitch.completed_token
  %0 = quidditch_snitch.wait_for_tensor_copy of %arg0 : tensor<?xf32> to %arg1 using %t -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @tensor_noop_transfer
func.func @tensor_noop_transfer(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, !quidditch_snitch.dma_token) {
  // CHECK: %[[T:.*]] = quidditch_snitch.completed_token
  // CHECK: %[[R:.*]] = quidditch_snitch.wait_for_tensor_copy
  // CHECK-NEXT: return %[[R]], %[[T]]
  %r, %t = quidditch_snitch.start_tensor_copy %arg0 to L1 : tensor<?xf32> -> tensor<?xf32>
  %0 = quidditch_snitch.wait_for_tensor_copy of %arg0 : tensor<?xf32> to %r using %t -> tensor<?xf32>
  %r2, %t2 = quidditch_snitch.start_tensor_copy %0 to L1 : tensor<?xf32> -> tensor<?xf32>
  return %r2, %t2 : tensor<?xf32>, !quidditch_snitch.dma_token
}

// CHECK-LABEL: @tensor_noop_pad
func.func @tensor_noop_pad(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, !quidditch_snitch.dma_token) {
  // CHECK: %[[R:.*]], %[[T:.*]] = quidditch_snitch.start_tensor_copy
  // CHECK-NOT: pad with
  %r, %t = quidditch_snitch.start_tensor_copy %arg0 to L1 pad with zero to [0] : tensor<?xf32> -> tensor<?xf32>
  // CHECK-NEXT: return %[[R]], %[[T]]
  return %r, %t : tensor<?xf32>, !quidditch_snitch.dma_token
}

// CHECK-LABEL: @tensor_pad_constant
func.func @tensor_pad_constant(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, !quidditch_snitch.dma_token) {
  %zero = arith.constant 0 : index
  // CHECK: %[[R:.*]], %[[T:.*]] = quidditch_snitch.start_tensor_copy
  // CHECK-NOT: pad with
  %r, %t = quidditch_snitch.start_tensor_copy %arg0 to L1 pad with zero to [%zero] : tensor<?xf32> -> tensor<?xf32>
  // CHECK-NEXT: return %[[R]], %[[T]]
  return %r, %t : tensor<?xf32>, !quidditch_snitch.dma_token
}
