// RUN: quidditch-opt %s --one-shot-bufferize | FileCheck %s

// CHECK: #[[$MAP2:.*]] = affine_map<()[s0, s1] -> (s0 + s1)>

// CHECK: func @copy_l1_buffer(
func.func @copy_l1_buffer(%arg0 : tensor<32xf32>) -> (tensor<32xf32>, !dma.token) {
  // CHECK: %[[ARG0:.*]] = bufferization.to_memref

  // CHECK: %[[ALLOC:.*]] = memref.alloc()
  // CHECK-SAME: : memref<32xf32, #quidditch_snitch.l1_encoding>
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]]
  // CHECK-SAME: to memref<32xf32, strided<[1]>, #quidditch_snitch.l1_encoding>
  // CHECK: %[[TOKEN:.*]] = dma.start_transfer from %[[ARG0]]
  // CHECK-SAME: to %[[SUBVIEW]]
  // CHECK: %[[R:.*]] = bufferization.to_tensor %[[ALLOC]]
  %r, %token = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding -> tensor<32xf32>
  // CHECK: return %[[R]], %[[TOKEN]]
  return %r, %token : tensor<32xf32>, !dma.token
}

// CHECK: func @copy_l1_buffer_elided(
func.func @copy_l1_buffer_elided(%arg0 : tensor<32xf32>) -> tensor<32xf32> {
  // CHECK: memref.alloc()
  // CHECK-NOT: memref.alloc()
  %r:2 = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding -> tensor<32xf32>
  %r2 = dma.wait_for_tensor_copy of %arg0 : tensor<32xf32> to %r#0 using %r#1 -> tensor<32xf32>
  %r3:2 = dma.start_tensor_copy of %r2 to #quidditch_snitch.l1_encoding -> tensor<32xf32>
  %r4 = dma.wait_for_tensor_copy of %r2 : tensor<32xf32> to %r3#0 using %r3#1 -> tensor<32xf32>
  // CHECK: return
  return %r4 : tensor<32xf32>
}

// CHECK: func @copy_l1_buffer_alloca_elided(
func.func @copy_l1_buffer_alloca_elided() -> tensor<32xf32> {
  // CHECK: memref.alloc()
  // CHECK-NOT: memref.alloc()
  %r = bufferization.alloc_tensor() {memory_space = #quidditch_snitch.l1_encoding} : tensor<32xf32>
  %r2:2 = dma.start_tensor_copy of %r to #quidditch_snitch.l1_encoding : tensor<32xf32> -> tensor<32xf32>
  // CHECK: return
  return %r2#0 : tensor<32xf32>
}

// CHECK: func @scf_for_copy_l1_buffer(
func.func @scf_for_copy_l1_buffer() -> tensor<32xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[MEMREF:.*]] = memref.alloc
  %r = bufferization.alloc_tensor() {memory_space = #quidditch_snitch.l1_encoding} : tensor<32xf32>
  %r2:2 = dma.start_tensor_copy of %r to #quidditch_snitch.l1_encoding : tensor<32xf32> -> tensor<32xf32>
  // CHECK-NEXT: dma.completed_token
  // CHECK-NEXT: %[[R:.*]] = scf.for
  // CHECK-SAME: iter_args(%[[ITER:.*]] = %[[MEMREF]])
  // CHECK-NEXT: dma.completed_token
  // CHECK-NEXT: scf.yield %[[ITER]]
  // CHECK: bufferization.to_tensor %[[R]]
  %r3 = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %r2#0) -> (tensor<32xf32>) {
    %r4:2 = dma.start_tensor_copy of %iter to #quidditch_snitch.l1_encoding -> tensor<32xf32>
    scf.yield %r4#0 : tensor<32xf32>
  }
  return %r3 : tensor<32xf32>
}

// CHECK: func @copy_l1_buffer_dynamic_dims(
func.func @copy_l1_buffer_dynamic_dims(%arg0 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK: %[[ARG0:.*]] = bufferization.to_memref
  // CHECK: %[[ZERO:.*]] = arith.constant 0
  // CHECK: %[[DIM_IN:.*]] = memref.dim %[[ARG0]], %[[ZERO]]
  // CHECK: %[[DIM:.*]] = affine.apply #{{.*}}()[%[[DIM_IN]]]
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[DIM]])
  // CHECK-SAME: : memref<?xf32, #quidditch_snitch.l1_encoding>
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOC]]
  // CHECK-SAME: to memref<?xf32, strided<[1]>, #quidditch_snitch.l1_encoding>
  // CHECK: dma.start_transfer from %[[ARG0]]
  // CHECK-SAME: to %[[SUBVIEW]]
  // CHECK: %[[R:.*]] = bufferization.to_tensor %[[ALLOC]]
  %r:2 = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding -> tensor<?xf32>
  // CHECK: return %[[R]]
  return %r#0 : tensor<?xf32>
}

// CHECK-LABEL: @tensor_copy_pad
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[PAD0:[[:alnum:]]+]]
// CHECK-SAME: %[[PAD1:[[:alnum:]]+]]
func.func @tensor_copy_pad(%arg0 : tensor<?x?xf32>, %pad0 : index, %pad1 : index) -> (tensor<?x?xf32>, !dma.token) {
  // CHECK: %[[COPY:.*]] = bufferization.to_memref %[[ARG0]]
  // CHECK: %[[ZERO:.*]] = arith.constant 0
  // CHECK: %[[DIM0:.*]] = memref.dim %[[COPY]], %[[ZERO]]
  // CHECK: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[DIM1:.*]] = memref.dim %[[COPY]], %[[ONE]]
  // CHECK: %[[NEW_DIM0:.*]] = affine.apply #[[$MAP2]]()[%[[DIM0]], %[[PAD0]]]
  // CHECK: %[[NEW_DIM1:.*]] = affine.apply #[[$MAP2]]()[%[[DIM1]], %[[PAD1]]]
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[NEW_DIM0]], %[[NEW_DIM1]])
  // CHECK: %[[ZERO_TOKEN:.*]] = dma.start_zero_mem_transfer %[[ALLOC]]
  // CHECK: dma.wait_for_transfer %[[ZERO_TOKEN]]
  // CHECK: %[[UNPADDED:.*]] = memref.subview %[[ALLOC]][0, 0] [%[[DIM0]], %[[DIM1]]] [1, 1]
  // CHECK: %[[TOKEN:.*]] = dma.start_transfer from %[[COPY]]
  // CHECK-SAME: to %[[UNPADDED]]
  %r, %t = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding pad with zero by [%pad0, %pad1] : tensor<?x?xf32> -> tensor<?x?xf32>
  // CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]]
  // CHECK: return %[[TENSOR]], %[[TOKEN]]
  return %r, %t : tensor<?x?xf32>, !dma.token
}

// CHECK-LABEL: @tensor_copy_pad_undef
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[PAD0:[[:alnum:]]+]]
// CHECK-SAME: %[[PAD1:[[:alnum:]]+]]
func.func @tensor_copy_pad_undef(%arg0 : tensor<?x?xf32>, %pad0 : index, %pad1 : index) -> (tensor<?x?xf32>, !dma.token) {
  // CHECK: %[[COPY:.*]] = bufferization.to_memref %[[ARG0]]
  // CHECK: %[[ZERO:.*]] = arith.constant 0
  // CHECK: %[[DIM0:.*]] = memref.dim %[[COPY]], %[[ZERO]]
  // CHECK: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[DIM1:.*]] = memref.dim %[[COPY]], %[[ONE]]
  // CHECK: %[[NEW_DIM0:.*]] = affine.apply #[[$MAP2]]()[%[[DIM0]], %[[PAD0]]]
  // CHECK: %[[NEW_DIM1:.*]] = affine.apply #[[$MAP2]]()[%[[DIM1]], %[[PAD1]]]
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[NEW_DIM0]], %[[NEW_DIM1]])
  // CHECK-NOT: start_zero_mem_transfer
  // CHECK: %[[UNPADDED:.*]] = memref.subview %[[ALLOC]][0, 0] [%[[DIM0]], %[[DIM1]]] [1, 1]
  // CHECK-NEXT: %[[TOKEN:.*]] = dma.start_transfer from %[[COPY]]
  // CHECK-SAME: to %[[UNPADDED]]
  %r, %t = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding pad with undef by [%pad0, %pad1] : tensor<?x?xf32> -> tensor<?x?xf32>
  // CHECK: %[[TENSOR:.*]] = bufferization.to_tensor %[[ALLOC]]
  // CHECK: return %[[TENSOR]], %[[TOKEN]]
  return %r, %t : tensor<?x?xf32>, !dma.token
}
