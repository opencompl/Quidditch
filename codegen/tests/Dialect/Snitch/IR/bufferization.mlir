// RUN: quidditch-opt %s --one-shot-bufferize | FileCheck %s

// CHECK: func @copy_l1_buffer(
func.func @copy_l1_buffer(%arg0 : tensor<32xf32>) -> tensor<32xf32> {
  // CHECK: %[[ARG0:.*]] = bufferization.to_memref

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK-SAME: : memref<32xf32, #quidditch_snitch.l1_encoding>
  // CHECK: memref.copy %[[ARG0]], %[[ALLOC]]
  // CHECK: %[[R:.*]] = bufferization.to_tensor %[[ALLOC]]
  %r = quidditch_snitch.copy_tensor %arg0 to L1 : tensor<32xf32>
  // CHECK: return %[[R]]
  return %r : tensor<32xf32>
}

// CHECK: func @copy_l1_buffer_elided(
func.func @copy_l1_buffer_elided(%arg0 : tensor<32xf32>) -> tensor<32xf32> {
  // CHECK: memref.alloc()
  // CHECK-NOT: memref.alloc()
  %r = quidditch_snitch.copy_tensor %arg0 to L1 : tensor<32xf32>
  %r2 = quidditch_snitch.copy_tensor %r to L1 : tensor<32xf32>
  // CHECK: return
  return %r2 : tensor<32xf32>
}

// CHECK: func @copy_l1_buffer_alloca_elided(
func.func @copy_l1_buffer_alloca_elided() -> tensor<32xf32> {
  // CHECK: memref.alloc()
  // CHECK-NOT: memref.alloc()
  %r = bufferization.alloc_tensor() {memory_space = #quidditch_snitch.l1_encoding} : tensor<32xf32>
  %r2 = quidditch_snitch.copy_tensor %r to L1 : tensor<32xf32>
  // CHECK: return
  return %r2 : tensor<32xf32>
}

// CHECK: func @scf_for_copy_l1_buffer(
func.func @scf_for_copy_l1_buffer() -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[MEMREF:.*]] = memref.alloc
  %r = bufferization.alloc_tensor() {memory_space = #quidditch_snitch.l1_encoding} : tensor<32xf32>
  %r2 = quidditch_snitch.copy_tensor %r to L1 : tensor<32xf32>
  // CHECK-NEXT: %[[R:.*]] = scf.for
  // CHECK-SAME: iter_args(%[[ITER:.*]] = %[[MEMREF]])
  // CHECK-NEXT: scf.yield %[[ITER]]
  // CHECK: bufferization.to_tensor %[[R]]
  %r3 = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %r2) -> (tensor<32xf32>) {
    %r4 = quidditch_snitch.copy_tensor %iter to L1 : tensor<32xf32>
    scf.yield %r4 : tensor<32xf32>
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
  // CHECK: memref.copy %[[ARG0]], %[[ALLOC]]
  // CHECK: %[[R:.*]] = bufferization.to_tensor %[[ALLOC]]
  %r = quidditch_snitch.copy_tensor %arg0 to L1 : tensor<?xf32>
  // CHECK: return %[[R]]
  return %r : tensor<?xf32>
}
