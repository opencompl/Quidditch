// RUN: quidditch-opt %s --quidditch-convert-to-llvm | FileCheck %s

// CHECK-LABEL: @test1d(
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_SIZE:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1_PTR:[[:alnum:]]+]]
func.func private @test1d(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>) -> !dma.token {
  // CHECK: %[[FOUR:.*]] = llvm.mlir.constant(4 :
  // CHECK: %[[SIZE:.*]] = llvm.mul %[[ARG0_SIZE]], %[[FOUR]]
  // CHECK: %[[R:.*]] = llvm.call @snrt_dma_start_1d(%[[ARG1_PTR]], %[[ARG0_PTR]], %[[SIZE]])
  %0 = dma.start_transfer from %arg0 : memref<?xf32> to %arg1 : memref<?xf32>
  // CHECK: return %[[R]]
  return %0 : !dma.token
}

// CHECK-LABEL: @test2d(
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_DIM0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG0_DIM1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG0_STRIDE0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG0_STRIDE1:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1_STRIDE0:[[:alnum:]]+]]
func.func private @test2d(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32, strided<[8, 1], offset: 0>>) -> !dma.token {
  // CHECK-DAG: %[[ELEMENT_WIDTH:.*]] = llvm.mlir.constant(4 :
  // CHECK: %[[INNER_SIZE:.*]] = llvm.mul %[[ARG0_DIM1]], %[[ELEMENT_WIDTH]]
  // CHECK: %[[ARG0_STRIDE:.*]] = llvm.mul %[[ARG0_STRIDE0]], %[[ELEMENT_WIDTH]]
  // CHECK: %[[ARG1_STRIDE:.*]] = llvm.mul %[[ARG1_STRIDE0]], %[[ELEMENT_WIDTH]]
  // CHECK: llvm.call @snrt_dma_start_2d(%[[ARG1_PTR]], %[[ARG0_PTR]], %[[INNER_SIZE]], %[[ARG1_STRIDE]], %[[ARG0_STRIDE]], %[[ARG0_DIM0]])
  %0 = dma.start_transfer from %arg0 : memref<2x4xf32> to %arg1 : memref<2x4xf32, strided<[8, 1], offset: 0>>
  return %0 : !dma.token
}
