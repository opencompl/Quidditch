// RUN: quidditch-opt %s --one-shot-bufferize | FileCheck %s

// CHECK: func @copy_l1_buffer(
func.func @copy_l1_buffer(%arg0 : tensor<32xf32>) -> (tensor<32xf32>, !quidditch_snitch.dma_token) {
  // CHECK: %[[ARG0:.*]] = bufferization.to_memref

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK-SAME: : memref<32xf32, #quidditch_snitch.l1_encoding>
  // CHECK: %[[TOKEN:.*]] = quidditch_snitch.start_dma_transfer from %[[ARG0]]
  // CHECK-SAME: to %[[ALLOC]]
  // CHECK: %[[R:.*]] = bufferization.to_tensor %[[ALLOC]]
  %r, %token = quidditch_snitch.start_tensor_copy %arg0 to L1 : tensor<32xf32>
  // CHECK: return %[[R]], %[[TOKEN]]
  return %r, %token : tensor<32xf32>, !quidditch_snitch.dma_token
}

// CHECK: func @copy_l1_buffer_elided(
func.func @copy_l1_buffer_elided(%arg0 : tensor<32xf32>) -> tensor<32xf32> {
  // CHECK: memref.alloc()
  // CHECK-NOT: memref.alloc()
  %r:2 = quidditch_snitch.start_tensor_copy %arg0 to L1 : tensor<32xf32>
  %r2 = quidditch_snitch.wait_for_tensor_copy of %arg0 to %r#0 using %r#1 : tensor<32xf32>
  %r3:2 = quidditch_snitch.start_tensor_copy %r2 to L1 : tensor<32xf32>
  %r4 = quidditch_snitch.wait_for_tensor_copy of %r2 to %r3#0 using %r3#1 : tensor<32xf32>
  // CHECK: return
  return %r4 : tensor<32xf32>
}

// CHECK: func @copy_l1_buffer_alloca_elided(
func.func @copy_l1_buffer_alloca_elided() -> tensor<32xf32> {
  // CHECK: memref.alloc()
  // CHECK-NOT: memref.alloc()
  %r = bufferization.alloc_tensor() {memory_space = #quidditch_snitch.l1_encoding} : tensor<32xf32>
  %r2:2 = quidditch_snitch.start_tensor_copy %r to L1 : tensor<32xf32>
  // CHECK: return
  return %r2#0 : tensor<32xf32>
}

// CHECK: func @scf_for_copy_l1_buffer(
func.func @scf_for_copy_l1_buffer() -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[MEMREF:.*]] = memref.alloc
  %r = bufferization.alloc_tensor() {memory_space = #quidditch_snitch.l1_encoding} : tensor<32xf32>
  %r2:2 = quidditch_snitch.start_tensor_copy %r to L1 : tensor<32xf32>
  // CHECK-NEXT: quidditch_snitch.completed_token
  // CHECK-NEXT: %[[R:.*]] = scf.for
  // CHECK-SAME: iter_args(%[[ITER:.*]] = %[[MEMREF]])
  // CHECK-NEXT: quidditch_snitch.completed_token
  // CHECK-NEXT: scf.yield %[[ITER]]
  // CHECK: bufferization.to_tensor %[[R]]
  %r3 = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %r2#0) -> (tensor<32xf32>) {
    %r4:2 = quidditch_snitch.start_tensor_copy %iter to L1 : tensor<32xf32>
    scf.yield %r4#0 : tensor<32xf32>
  }
  return %r3 : tensor<32xf32>
}

// CHECK: func @copy_l1_buffer_dynamic_dims(
func.func @copy_l1_buffer_dynamic_dims(%arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[ARG0:.*]] = bufferization.to_memref
  // CHECK: %[[ZERO:.*]] = arith.constant 0
  // CHECK: %[[DIM:.*]] = memref.dim %[[ARG0]], %[[ZERO]]
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[DIM]])
  // CHECK-SAME: : memref<?xf32, #quidditch_snitch.l1_encoding>
  // CHECK: quidditch_snitch.start_dma_transfer from %[[ARG0]]
  // CHECK-SAME: to %[[ALLOC]]
  // CHECK: %[[R:.*]] = bufferization.to_tensor %[[ALLOC]]
  %r:2 = quidditch_snitch.start_tensor_copy %arg0 to L1 : tensor<?xf32>
  // CHECK: return %[[R]]
  return %r#0 : tensor<?xf32>
}

// CHECK-LABEL: @pipeline_op(
func.func @pipeline_op(%arg0_dim : index) -> tensor<?xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[C10:.*]] = arith.constant 10
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %arg0 = tensor.empty(%arg0_dim) : tensor<?xf32>

  // CHECK: pipeline %[[C0]] to %[[C10]] step %[[C1]] {
  %t = quidditch_snitch.pipeline %c0 to %c10 step %c1 inits(%arg0) -> tensor<?xf32> {
  // CHECK: ^{{.*}}(%[[IV:.*]]: index):
  ^bb0(%iv: index, %tensor: tensor<?xf32>):
    // CHECK: quidditch_snitch.pipeline_yield
    quidditch_snitch.pipeline_yield %tensor : tensor<?xf32>
  }, {
  // CHECK: ^{{.*}}(%[[IV:.*]]: index, %{{.*}}: memref<?xf32{{.*}}>):
  ^bb0(%iv: index, %tensor: tensor<?xf32>):
    quidditch_snitch.pipeline_yield %tensor : tensor<?xf32>
  // CHECK-NEXT: }
  }
  return %t : tensor<?xf32>
}
