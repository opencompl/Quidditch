// RUN: quidditch-opt %s --canonicalize --split-input-file --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @wait_gets_removed
func.func @wait_gets_removed() {
  // CHECK-NEXT: return
  %0 = dma.completed_token
  dma.wait_for_transfer %0
  return
}

// CHECK-LABEL: @noop_transfer
func.func @noop_transfer(%arg0 : memref<?xf32>) -> !dma.token {
  // CHECK-NEXT: %[[R:.*]] = dma.completed_token
  // CHECK-NEXT: return %[[R]]
  %0 = dma.start_transfer from %arg0 : memref<?xf32> to %arg0 : memref<?xf32>
  return %0 : !dma.token
}

// CHECK-LABEL: @tensor_wait_gets_removed
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
func.func @tensor_wait_gets_removed(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NEXT: return %[[ARG1]]
  %t = dma.completed_token
  %0 = dma.wait_for_tensor_copy of %arg0 : tensor<?xf32> to %arg1 using %t -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: @tensor_noop_transfer
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @tensor_noop_transfer(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, !dma.token) {
  // CHECK: %[[T2:.*]] = dma.completed_token
  // CHECK: %[[R:.*]], %[[T:.*]] = dma.start_tensor_copy of %[[ARG0]]
  %r, %t = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding -> tensor<?xf32>
  // CHECK: %[[R2:.*]] = dma.wait_for_tensor_copy of %[[ARG0]]
  // CHECK-SAME: to %[[R]] using %[[T]]
  %0 = dma.wait_for_tensor_copy of %arg0 : tensor<?xf32> to %r using %t -> tensor<?xf32>

  // CHECK-NOT: wait_for_tensor_copy
  %r2, %t2 = dma.start_tensor_copy of %0 to #quidditch_snitch.l1_encoding -> tensor<?xf32>

  // CHECK: return %[[R2]], %[[T2]]
  return %r2, %t2 : tensor<?xf32>, !dma.token
}

// CHECK-LABEL: @tensor_noop_pad
func.func @tensor_noop_pad(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, !dma.token) {
  // CHECK: %[[R:.*]], %[[T:.*]] = dma.start_tensor_copy
  // CHECK-NOT: pad with
  %r, %t = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding pad with zero by [0] : tensor<?xf32> -> tensor<?xf32>
  // CHECK-NEXT: return %[[R]], %[[T]]
  return %r, %t : tensor<?xf32>, !dma.token
}

// CHECK-LABEL: @tensor_pad_constant
func.func @tensor_pad_constant(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, !dma.token) {
  %zero = arith.constant 0 : index
  // CHECK: %[[R:.*]], %[[T:.*]] = dma.start_tensor_copy
  // CHECK-NOT: pad with
  %r, %t = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding pad with zero by [%zero] : tensor<?xf32> -> tensor<?xf32>
  // CHECK-NEXT: return %[[R]], %[[T]]
  return %r, %t : tensor<?xf32>, !dma.token
}

// CHECK-LABEL: @tensor_noop_transfer_pad
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @tensor_noop_transfer_pad(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, !dma.token) {
  // CHECK: %[[T2:.*]] = dma.completed_token
  // CHECK: %[[R:.*]], %[[T:.*]] = dma.start_tensor_copy of %[[ARG0]]
  %r, %t = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding pad with zero by [1] : tensor<?xf32> -> tensor<?xf32>
  // CHECK: %[[R2:.*]] = dma.wait_for_tensor_copy of %[[ARG0]]
  // CHECK-SAME: to %[[R]] using %[[T]]
  %0 = dma.wait_for_tensor_copy of %arg0 : tensor<?xf32> to %r using %t -> tensor<?xf32>

  // CHECK-NOT: wait_for_tensor_copy
  %r2, %t2 = dma.start_tensor_copy of %0 to #quidditch_snitch.l1_encoding -> tensor<?xf32>

  // CHECK: return %[[R2]], %[[T2]]
  return %r2, %t2 : tensor<?xf32>, !dma.token
}

// CHECK-LABEL: @tensor_noop_transfer_pad_neg
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @tensor_noop_transfer_pad_neg(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, !dma.token) {
  // CHECK: start_tensor_copy
  // CHECK: wait_for_tensor_copy
  // CHECK: %[[R:.*]], %[[T:.*]] = dma.start_tensor_copy
  // CHECK: return %[[R]], %[[T]]

  %r, %t = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding -> tensor<?xf32>
  %0 = dma.wait_for_tensor_copy of %arg0 : tensor<?xf32> to %r using %t -> tensor<?xf32>
  %r2, %t2 = dma.start_tensor_copy of %0 to #quidditch_snitch.l1_encoding pad with zero by [1] : tensor<?xf32> -> tensor<?xf32>
  return %r2, %t2 : tensor<?xf32>, !dma.token
}

// CHECK-LABEL: @tensor_noop_transfer_same_padding
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func @tensor_noop_transfer_same_padding(%arg0 : tensor<?xf32>) -> (tensor<?xf32>, !dma.token) {
  // CHECK: %[[T2:.*]] = dma.completed_token
  // CHECK: %[[R:.*]], %[[T:.*]] = dma.start_tensor_copy of %[[ARG0]]
  %r, %t = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding pad with zero by [1] : tensor<?xf32> -> tensor<?xf32>
  // CHECK: %[[R2:.*]] = dma.wait_for_tensor_copy of %[[ARG0]]
  // CHECK-SAME: to %[[R]] using %[[T]]
  %0 = dma.wait_for_tensor_copy of %arg0 : tensor<?xf32> to %r using %t -> tensor<?xf32>

  // CHECK-NOT: wait_for_tensor_copy
  %r2, %t2 = dma.start_tensor_copy of %0 to #quidditch_snitch.l1_encoding pad with zero by [1] : tensor<?xf32> -> tensor<?xf32>

  // CHECK: return %[[R2]], %[[T2]]
  return %r2, %t2 : tensor<?xf32>, !dma.token
}
