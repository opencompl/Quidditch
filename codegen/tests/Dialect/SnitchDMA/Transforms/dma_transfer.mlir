// RUN: quidditch-opt %s --quidditch-snitch-legalize-dma-operations | FileCheck %s

// CHECK-LABEL: @collapse(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
func.func private @collapse(%arg0 : memref<?x4xf32>, %arg1 : memref<?x4xf32, strided<[4, 1], offset: ?>>) -> !dma.token {
  // CHECK: %[[COLLAPSE_ARG0:.*]] = memref.collapse_shape %[[ARG0]]
  // CHECK-SAME{LITERAL}: [[0, 1]]
  // CHECK: %[[COLLAPSE_ARG1:.*]] = memref.collapse_shape %[[ARG1]]
  // CHECK-SAME{LITERAL}: [[0, 1]]
  // CHECK: %[[TOKEN:.*]] = dma.start_transfer
  // CHECK-SAME: from %[[COLLAPSE_ARG0]]
  // CHECK-SAME: to %[[COLLAPSE_ARG1]]
  %0 = dma.start_transfer from %arg0 : memref<?x4xf32> to %arg1 : memref<?x4xf32, strided<[4, 1], offset: ?>>
  // CHECK: return %[[TOKEN]]
  return %0 : !dma.token
}

// CHECK-LABEL: @rank_reduction(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
func.func private @rank_reduction(%subview_3 : memref<1x?xf64, strided<[161, 1], offset: ?>>, %subview_5 : memref<1x?xf64, strided<[81, 1]>>) {
  // CHECK: %[[ONE:.*]] = arith.constant 1
  // CHECK: %[[DIM0:.*]] = memref.dim %[[ARG0]], %[[ONE]]
  // CHECK: %[[ARG0_VIEW:.*]] = memref.subview %[[ARG0]][0, 0] [1, %[[DIM0]]] [1, 1]
  // CHECK: %[[ARG1_VIEW:.*]] = memref.subview %[[ARG1]][0, 0] [1, %[[DIM0]]] [1, 1]
  // CHECK: dma.start_transfer
  // CHECK-SAME: from %[[ARG0_VIEW]]
  // CHECK-SAME: to %[[ARG1_VIEW]]
  %12 = dma.start_transfer from %subview_3 : memref<1x?xf64, strided<[161, 1], offset: ?>> to %subview_5 : memref<1x?xf64, strided<[81, 1]>>
  return
}

// CHECK-LABEL: @legal_2d(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
func.func private @legal_2d(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32, strided<[8, 1]>>) -> !dma.token {
  // CHECK: dma.start_transfer
  // CHECK-SAME: from %[[ARG0]]
  // CHECK-SAME: to %[[ARG1]]
  %0 = dma.start_transfer from %arg0 : memref<2x4xf32> to %arg1 : memref<2x4xf32, strided<[8, 1]>>
  return %0 : !dma.token
}

// CHECK-LABEL: @transfer_3d(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
func.func private @transfer_3d(%arg0 : memref<3x2x4xf32>, %arg1 : memref<3x2x4xf32, strided<[16, 8, 1], offset: 2>>) -> !dma.token {
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0
  // CHECK-DAG: %[[ONE:.*]] = arith.constant 1
  // CHECK-DAG: %[[THREE:.*]] = arith.constant 3
  // CHECK: %[[TOKEN:.*]] = dma.completed_token
  // CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[ZERO]] to %[[THREE]] step %[[ONE]] iter_args(%[[ITER:.*]] = %[[TOKEN]])
  // CHECK:   %[[SOURCE_VIEW:.*]] = memref.subview %[[ARG0]][%[[IV]], 0, 0] [1, 2, 4] [1, 1, 1]
  // CHECK:   %[[DEST_VIEW:.*]] = memref.subview %[[ARG1]][%[[IV]], 0, 0] [1, 2, 4] [1, 1, 1]
  // CHECK:   %[[TOKEN2:.*]] = dma.start_transfer
  // CHECK-SAME: from %[[SOURCE_VIEW]]
  // CHECK-SAME: to %[[DEST_VIEW]]
  // CHECK:   %[[COMBINED:.*]] = dma.combine_tokens %[[TOKEN2]], %[[ITER]]
  // CHECK:   scf.yield %[[COMBINED]]
  %0 = dma.start_transfer from %arg0 : memref<3x2x4xf32> to %arg1 : memref<3x2x4xf32, strided<[16, 8, 1], offset: 2>>
  // CHECK: return %[[FOR]]
  return %0 : !dma.token
}

// CHECK-LABEL: @illegal_1d(
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
func.func private @illegal_1d(%arg0 : memref<?xf32>, %arg1 : memref<?xf32, strided<[?]>>) -> !dma.token {
  // CHECK: %[[ZERO:.*]] = arith.constant 0
  // CHECK: %[[DIM0:.*]] = memref.dim %[[ARG0]], %[[ZERO]]
  // CHECK: %[[EXPAND_ARG0:.*]] = memref.expand_shape %[[ARG0]]
  // CHECK-SAME{LITERAL}: [[0, 1]]
  // CHECK-SAME: output_shape [%[[DIM0]], 1]
  // CHECK: %[[ZERO:.*]] = arith.constant 0
  // CHECK: %[[DIM0:.*]] = memref.dim %[[ARG1]], %[[ZERO]]
  // CHECK: %[[EXPAND_ARG1:.*]] = memref.expand_shape %[[ARG1]]
  // CHECK-SAME{LITERAL}: [[0, 1]]
  // CHECK-SAME: output_shape [%[[DIM0]], 1]
  // CHECK: %[[TOKEN:.*]] = dma.start_transfer
  // CHECK-SAME: from %[[EXPAND_ARG0]]
  // CHECK-SAME: to %[[EXPAND_ARG1]]
  %0 = dma.start_transfer from %arg0 : memref<?xf32> to %arg1 : memref<?xf32, strided<[?]>>
  return %0 : !dma.token
}
