// RUN: quidditch-opt %s --quidditch-convert-to-llvm | FileCheck %s

// CHECK-LABEL: @test
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_SIZE:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1_PTR:[[:alnum:]]+]]
func.func private @test(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>) -> !quidditch_snitch.dma_token {
  // CHECK: %[[ZERO:.*]] = llvm.mlir.zero
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ZERO]][%[[ARG0_SIZE]]]
  // CHECK: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
  // CHECK: %[[R:.*]] = llvm.call @snrt_dma_start_1d(%[[ARG1_PTR]], %[[ARG0_PTR]], %[[SIZE]])
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<?xf32> to %arg1 : memref<?xf32>
  // CHECK: return %[[R]]
  return %0 : !quidditch_snitch.dma_token
}

// CHECK-LABEL: @test2
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_SIZE:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1_ALIGNED_PTR:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1_OFFSET:[[:alnum:]]+]]
func.func private @test2(%arg0 : memref<?xf32>, %arg1 : memref<?xf32, strided<[1], offset: ?>>) -> !quidditch_snitch.dma_token {
  // CHECK: %[[ARG1_PTR:.*]] = llvm.getelementptr %[[ARG1_ALIGNED_PTR]][%[[ARG1_OFFSET]]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ARG0_SIZE]]]
  // CHECK: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
  // CHECK: %[[R:.*]] = llvm.call @snrt_dma_start_1d(%[[ARG1_PTR]], %[[ARG0_PTR]], %[[SIZE]])
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<?xf32> to %arg1 : memref<?xf32, strided<[1], offset: ?>>
  // CHECK: llvm.call @snrt_dma_start_1d(
  %1 = quidditch_snitch.start_dma_transfer from %arg1 : memref<?xf32, strided<[1], offset: ?>> to %arg0 : memref<?xf32>
  return %0 : !quidditch_snitch.dma_token
}

// CHECK-LABEL: @test3
func.func private @test3(%arg0 : memref<?x4xf32>, %arg1 : memref<?x4xf32, strided<[4, 1], offset: ?>>) -> !quidditch_snitch.dma_token {
  // CHECK: llvm.call @snrt_dma_start_1d(
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<?x4xf32> to %arg1 : memref<?x4xf32, strided<[4, 1], offset: ?>>
  return %0 : !quidditch_snitch.dma_token
}

// CHECK-LABEL: @dynamic_inner(
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_DIM1:[[:alnum:]]+]]
func.func private @dynamic_inner(%subview_3 : memref<1x?xf64, strided<[161, 1], offset: ?>>, %subview_5 : memref<1x?xf64, strided<[81, 1]>>) {
  // CHECK-DAG: %[[NULL:.*]] = llvm.mlir.zero
  // CHECK-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 :
  // CHECK: %[[SIZE:.*]] = llvm.mul %[[ARG0_DIM1]], %[[ONE]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][%[[SIZE]]]
  // CHECK: %[[BYTES:.*]] = llvm.ptrtoint %[[GEP]]
  // CHECK: call @snrt_dma_start_1d(
  // CHECK-SAME: %{{[[:alnum:]]+}}
  // CHECK-SAME: %{{[[:alnum:]]+}}
  // CHECK-SAME: %[[BYTES]]
  %12 = quidditch_snitch.start_dma_transfer from %subview_3 : memref<1x?xf64, strided<[161, 1], offset: ?>> to %subview_5 : memref<1x?xf64, strided<[81, 1]>>
  return
}

// CHECK-LABEL: @test4
func.func private @test4(%arg0 : memref<1x4xf32>, %arg1 : memref<1x4xf32, strided<[40, 1], offset: ?>>) -> !quidditch_snitch.dma_token {
  // CHECK: llvm.call @snrt_dma_start_1d(
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<1x4xf32> to %arg1 : memref<1x4xf32, strided<[40, 1], offset: ?>>
  return %0 : !quidditch_snitch.dma_token
}

// CHECK-LABEL: @test5
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_SIZE:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_STRIDE_N:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1_STRIDE_N:[[:alnum:]]+]]
func.func private @test5(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32, strided<[8, 1], offset: 0>>) -> !quidditch_snitch.dma_token {
  // CHECK: %[[ELEMENT_WIDTH:.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: %[[INNER_SIZE:.*]] = llvm.mul %[[ELEMENT_WIDTH]], %[[ELEMENT_WIDTH]]
  // CHECK: %[[ARG0_STRIDE:.*]] = llvm.mul %[[ARG0_STRIDE_N]], %[[ELEMENT_WIDTH]]
  // CHECK: %[[ARG1_STRIDE:.*]] = llvm.mul %[[ARG1_STRIDE_N]], %[[ELEMENT_WIDTH]]
  // CHECK: llvm.call @snrt_dma_start_2d(%[[ARG1_PTR]], %[[ARG0_PTR]], %[[INNER_SIZE]], %[[ARG1_STRIDE]], %[[ARG0_STRIDE]], %[[ARG0_SIZE]])
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<2x4xf32> to %arg1 : memref<2x4xf32, strided<[8, 1], offset: 0>>
  return %0 : !quidditch_snitch.dma_token
}

// CHECK-LABEL: @test6
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_SIZE:[[:alnum:]]+]]
// CHECK-SAME: %[[DIM1:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_STRIDE0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG0_STRIDE_N:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1_ALIGNED_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1_STRIDE0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1_STRIDE_N:[[:alnum:]]+]]
func.func private @test6(%arg0 : memref<3x2x4xf32>, %arg1 : memref<3x2x4xf32, strided<[16, 8, 1], offset: 2>>) -> !quidditch_snitch.dma_token {
  // CHECK-DAG: %[[ELEMENT_WIDTH:.*]] = llvm.mlir.constant(4 : i32)
  // CHECK-DAG: %[[ZERO32:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : index) : i32
  // CHECK-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 : {{.*}}) : i32
  // CHECK: %[[ARG1_PTR:.*]] = llvm.getelementptr %[[ARG1_ALIGNED_PTR]][2]
  // CHECK: %[[INNER_SIZE:.*]] = llvm.mul %[[ELEMENT_WIDTH]], %[[ELEMENT_WIDTH]]
  // CHECK: llvm.br ^[[BB1:.*]](%[[ZERO]], %[[ZERO32]]

  // CHECK: ^[[BB1]](%[[IV1:.*]]: i32, %[[IV2:.*]]: i32):
  // CHECK: %[[COND:.*]] = llvm.icmp "slt" %[[IV1]], %[[ARG0_SIZE]]
  // CHECK: llvm.cond_br %[[COND]], ^[[BODY:.*]], ^[[EXIT:[[:alnum:]]+]]

  // CHECK: ^[[BODY]]:
  // CHECK: %[[MUL:.*]] = llvm.mul %[[IV1]], %[[ARG0_STRIDE0]]
  // CHECK: %[[ARG0_OFFSET1:.*]] = llvm.add %[[MUL]], %[[ZERO32]]
  // CHECK: %[[ARG0_ADJUSTED:.*]] = llvm.getelementptr %[[ARG0_PTR]][%[[ARG0_OFFSET1]]]

  // CHECK: %[[MUL:.*]] = llvm.mul %[[IV1]], %[[ARG1_STRIDE0]]
  // CHECK: %[[ARG1_OFFSET1:.*]] = llvm.add %[[MUL]], %[[ZERO32]]
  // CHECK: %[[ARG1_ADJUSTED:.*]] = llvm.getelementptr %[[ARG1_PTR]][%[[ARG1_OFFSET1]]]

  // CHECK: %[[ARG0_STRIDE:.*]] = llvm.mul %[[ARG0_STRIDE_N]], %[[ELEMENT_WIDTH]]
  // CHECK: %[[ARG1_STRIDE:.*]] = llvm.mul %[[ARG1_STRIDE_N]], %[[ELEMENT_WIDTH]]
  // CHECK: %[[RES:.*]] = llvm.call @snrt_dma_start_2d(%[[ARG1_ADJUSTED]], %[[ARG0_ADJUSTED]], %[[INNER_SIZE]], %[[ARG1_STRIDE]], %[[ARG0_STRIDE]], %[[DIM1]])
  // CHECK: %[[INV:.*]] = llvm.add %[[IV1]], %[[ONE]]
  // CHECK: llvm.br ^[[BB1]](%[[INV]], %[[RES]]

  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<3x2x4xf32> to %arg1 : memref<3x2x4xf32, strided<[16, 8, 1], offset: 2>>
  // CHECK: return %[[IV2]]
  return %0 : !quidditch_snitch.dma_token
}


// CHECK-LABEL: @dynamic_strides
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_SIZE:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG0_STRIDE_N:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1_PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[ARG1_STRIDE_N:[[:alnum:]]+]]
func.func private @dynamic_strides(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32, strided<[?, 1], offset: 0>>) -> !quidditch_snitch.dma_token {
  // CHECK: %[[ELEMENT_WIDTH:.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: %[[INNER_SIZE:.*]] = llvm.mul %[[ELEMENT_WIDTH]], %[[ELEMENT_WIDTH]]
  // CHECK: %[[ARG0_STRIDE:.*]] = llvm.mul %[[ARG0_STRIDE_N]], %[[ELEMENT_WIDTH]]
  // CHECK: %[[ARG1_STRIDE:.*]] = llvm.mul %[[ARG1_STRIDE_N]], %[[ELEMENT_WIDTH]]
  // CHECK: llvm.call @snrt_dma_start_2d(%[[ARG1_PTR]], %[[ARG0_PTR]], %[[INNER_SIZE]], %[[ARG1_STRIDE]], %[[ARG0_STRIDE]], %[[ARG0_SIZE]])
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<2x4xf32> to %arg1 : memref<2x4xf32, strided<[?, 1], offset: 0>>
  return %0 : !quidditch_snitch.dma_token
}
