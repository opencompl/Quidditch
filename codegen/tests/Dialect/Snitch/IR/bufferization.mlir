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
