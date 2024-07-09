// RUN: quidditch-opt %s --quidditch-convert-snitch-to-llvm | FileCheck %s

// CHECK-LABEL: @test
func.func @test(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>) -> !quidditch_snitch.dma_token {
  // CHECK: %[[ARG0_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
  // CHECK: %[[ARG1_PTR:.*]] = llvm.extractvalue %{{.*}}[1]
  // CHECK: %[[ARG0_SIZE:.*]] = llvm.extractvalue %{{.*}}[3, 0]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ARG0_SIZE]]]
  // CHECK: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
  // CHECK: %[[R:.*]] = llvm.call @snrt_dma_start_1d(%[[ARG1_PTR]], %[[ARG0_PTR]], %[[SIZE]])
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<?xf32> to %arg1 : memref<?xf32>
  // CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[R]]
  // CHECK: return %[[C]]
  return %0 : !quidditch_snitch.dma_token
}

// CHECK-LABEL: @test2
func.func @test2(%arg0 : memref<?xf32>, %arg1 : memref<?xf32, strided<[1], offset: ?>>) -> !quidditch_snitch.dma_token {
  // CHECK: %[[ARG0_PTR:.*]] = llvm.extractvalue %[[ARG0:.*]][1]
  // CHECK: %[[ARG1_ALIGNED_PTR:.*]] = llvm.extractvalue %[[ARG1:.*]][1]
  // CHECK: %[[ARG1_OFFSET:.*]] = llvm.extractvalue %[[ARG1]][2]
  // CHECK: %[[ARG1_PTR:.*]] = llvm.getelementptr %[[ARG1_ALIGNED_PTR]][%[[ARG1_OFFSET]]]
  // CHECK: %[[ARG0_SIZE:.*]] = llvm.extractvalue %[[ARG0]][3, 0]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %{{.*}}[%[[ARG0_SIZE]]]
  // CHECK: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
  // CHECK: %[[R:.*]] = llvm.call @snrt_dma_start_1d(%[[ARG1_PTR]], %[[ARG0_PTR]], %[[SIZE]])
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<?xf32> to %arg1 : memref<?xf32, strided<[1], offset: ?>>
  // CHECK: llvm.call @snrt_dma_start_1d(
  %1 = quidditch_snitch.start_dma_transfer from %arg1 : memref<?xf32, strided<[1], offset: ?>> to %arg0 : memref<?xf32>
  return %0 : !quidditch_snitch.dma_token
}

// CHECK-LABEL: @test3
func.func @test3(%arg0 : memref<?x4xf32>, %arg1 : memref<?x4xf32, strided<[4, 1], offset: ?>>) -> !quidditch_snitch.dma_token {
  // CHECK: llvm.call @snrt_dma_start_1d(
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<?x4xf32> to %arg1 : memref<?x4xf32, strided<[4, 1], offset: ?>>
  return %0 : !quidditch_snitch.dma_token
}

// CHECK-LABEL: @test4
func.func @test4(%arg0 : memref<1x4xf32>, %arg1 : memref<1x4xf32, strided<[40, 1], offset: ?>>) -> !quidditch_snitch.dma_token {
  // CHECK: llvm.call @snrt_dma_start_1d(
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<1x4xf32> to %arg1 : memref<1x4xf32, strided<[40, 1], offset: ?>>
  return %0 : !quidditch_snitch.dma_token
}

// CHECK-LABEL: @test5
// CHECK-SAME: %[[ARG0_M:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1_M:[[:alnum:]]+]]
func.func @test5(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32, strided<[8, 1], offset: 0>>) -> !quidditch_snitch.dma_token {
  // CHECK: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0_M]]
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1_M]]
  // CHECK: %[[ARG0_PTR:.*]] = llvm.extractvalue %[[ARG0]][1]
  // CHECK: %[[ARG1_PTR:.*]] = llvm.extractvalue %[[ARG1]][1]

  // CHECK: %[[ELEMENT_WIDTH:.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: %[[CONT_ELEMENTS:.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: %[[INNER_SIZE:.*]] = llvm.mul %[[CONT_ELEMENTS]], %[[ELEMENT_WIDTH]]

  // CHECK: %[[ARG0_OFFSET:.*]] = llvm.mlir.zero
  // CHECK: %[[ARG0_ADJUSTED:.*]] = llvm.getelementptr %[[ARG0_PTR]][%[[ARG0_OFFSET]]]

  // CHECK: %[[ARG1_OFFSET:.*]] = llvm.mlir.zero
  // CHECK: %[[ARG1_ADJUSTED:.*]] = llvm.getelementptr %[[ARG1_PTR]][%[[ARG1_OFFSET]]]

  // CHECK: %[[ARG0_STRIDE:.*]] = llvm.extractvalue %[[ARG0]][4, 0]
  // CHECK: %[[ARG1_STRIDE:.*]] = llvm.extractvalue %[[ARG1]][4, 0]
  // CHECK: %[[DIM0:.*]] = llvm.extractvalue %[[ARG0]][3, 0]
  // CHECK: llvm.call @snrt_dma_start_2d(%[[ARG1_ADJUSTED]], %[[ARG0_ADJUSTED]], %[[INNER_SIZE]], %[[ARG1_STRIDE]], %[[ARG0_STRIDE]], %[[DIM0]])
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<2x4xf32> to %arg1 : memref<2x4xf32, strided<[8, 1], offset: 0>>
  return %0 : !quidditch_snitch.dma_token
}

// CHECK-LABEL: @test6
// CHECK-SAME: %[[ARG0_M:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1_M:[[:alnum:]]+]]
func.func @test6(%arg0 : memref<3x2x4xf32>, %arg1 : memref<3x2x4xf32, strided<[16, 8, 1], offset: 2>>) -> !quidditch_snitch.dma_token {
  // CHECK: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0_M]]
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1_M]]
  // CHECK: %[[ARG0_PTR:.*]] = llvm.extractvalue %[[ARG0]][1]
  // CHECK: %[[ARG1_ALIGNED_PTR:.*]] = llvm.extractvalue %[[ARG1]][1]
  // CHECK: %[[ARG1_OFFSET:.*]] = llvm.mlir.constant(2 : index)
  // CHECK: %[[ARG1_PTR:.*]] = llvm.getelementptr %[[ARG1_ALIGNED_PTR]][%[[ARG1_OFFSET]]]

  // CHECK: %[[ELEMENT_WIDTH:.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: %[[CONT_ELEMENTS:.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: %[[INNER_SIZE:.*]] = llvm.mul %[[CONT_ELEMENTS]], %[[ELEMENT_WIDTH]]

  // CHECK: %[[DIM0:.*]] = llvm.extractvalue %[[ARG0]][3, 0]
  // CHECK: %[[DIM0_INDEX:.*]] = builtin.unrealized_conversion_cast %[[DIM0]]

  // CHECK: %[[LOOP:.*]] = scf.for %[[IV:.*]] = %{{.*}} to %[[DIM0_INDEX]] step %{{.*}} iter_args({{.*}})

  // CHECK: %[[ARG0_OFFSET:.*]] = llvm.mlir.zero
  // CHECK: %[[IV_I32:.*]] = builtin.unrealized_conversion_cast %[[IV]]
  // CHECK: %[[ARG0_STRIDE0:.*]] = llvm.extractvalue %[[ARG0]][4, 0]
  // CHECK: %[[MUL:.*]] = llvm.mul %[[IV_I32]], %[[ARG0_STRIDE0]]
  // CHECK: %[[ARG0_OFFSET1:.*]] = llvm.add %[[ARG0_OFFSET]], %[[MUL]]
  // CHECK: %[[ARG0_ADJUSTED:.*]] = llvm.getelementptr %[[ARG0_PTR]][%[[ARG0_OFFSET1]]]

  // CHECK: %[[ARG1_OFFSET:.*]] = llvm.mlir.zero
  // CHECK: %[[IV_I32:.*]] = builtin.unrealized_conversion_cast %[[IV]]
  // CHECK: %[[ARG1_STRIDE0:.*]] = llvm.extractvalue %[[ARG1]][4, 0]
  // CHECK: %[[MUL:.*]] = llvm.mul %[[IV_I32]], %[[ARG1_STRIDE0]]
  // CHECK: %[[ARG1_OFFSET1:.*]] = llvm.add %[[ARG1_OFFSET]], %[[MUL]]
  // CHECK: %[[ARG1_ADJUSTED:.*]] = llvm.getelementptr %[[ARG1_PTR]][%[[ARG1_OFFSET1]]]

  // CHECK: %[[ARG0_STRIDE:.*]] = llvm.extractvalue %[[ARG0]][4, 1]
  // CHECK: %[[ARG1_STRIDE:.*]] = llvm.extractvalue %[[ARG1]][4, 1]
  // CHECK: %[[DIM1:.*]] = llvm.extractvalue %[[ARG0]][3, 1]
  // CHECK: %[[RES:.*]] = llvm.call @snrt_dma_start_2d(%[[ARG1_ADJUSTED]], %[[ARG0_ADJUSTED]], %[[INNER_SIZE]], %[[ARG1_STRIDE]], %[[ARG0_STRIDE]], %[[DIM1]])
  // CHECK: scf.yield %[[RES]]

  // CHECK: builtin.unrealized_conversion_cast %[[LOOP]]
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<3x2x4xf32> to %arg1 : memref<3x2x4xf32, strided<[16, 8, 1], offset: 2>>
  return %0 : !quidditch_snitch.dma_token
}


// CHECK-LABEL: @dynamic_strides
// CHECK-SAME: %[[ARG0_M:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1_M:[[:alnum:]]+]]
func.func @dynamic_strides(%arg0 : memref<2x4xf32>, %arg1 : memref<2x4xf32, strided<[?, 1], offset: 0>>) -> !quidditch_snitch.dma_token {
  // CHECK: %[[ARG0:.*]] = builtin.unrealized_conversion_cast %[[ARG0_M]]
  // CHECK: %[[ARG1:.*]] = builtin.unrealized_conversion_cast %[[ARG1_M]]
  // CHECK: %[[ARG0_PTR:.*]] = llvm.extractvalue %[[ARG0]][1]
  // CHECK: %[[ARG1_PTR:.*]] = llvm.extractvalue %[[ARG1]][1]

  // CHECK: %[[ELEMENT_WIDTH:.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: %[[CONT_ELEMENTS:.*]] = llvm.mlir.constant(4 : i32)
  // CHECK: %[[INNER_SIZE:.*]] = llvm.mul %[[CONT_ELEMENTS]], %[[ELEMENT_WIDTH]]

  // CHECK: %[[ARG0_OFFSET:.*]] = llvm.mlir.zero
  // CHECK: %[[ARG0_ADJUSTED:.*]] = llvm.getelementptr %[[ARG0_PTR]][%[[ARG0_OFFSET]]]

  // CHECK: %[[ARG1_OFFSET:.*]] = llvm.mlir.zero
  // CHECK: %[[ARG1_ADJUSTED:.*]] = llvm.getelementptr %[[ARG1_PTR]][%[[ARG1_OFFSET]]]

  // CHECK: %[[ARG0_STRIDE:.*]] = llvm.extractvalue %[[ARG0]][4, 0]
  // CHECK: %[[ARG1_STRIDE:.*]] = llvm.extractvalue %[[ARG1]][4, 0]
  // CHECK: %[[DIM0:.*]] = llvm.extractvalue %[[ARG0]][3, 0]
  // CHECK: llvm.call @snrt_dma_start_2d(%[[ARG1_ADJUSTED]], %[[ARG0_ADJUSTED]], %[[INNER_SIZE]], %[[ARG1_STRIDE]], %[[ARG0_STRIDE]], %[[DIM0]])
  %0 = quidditch_snitch.start_dma_transfer from %arg0 : memref<2x4xf32> to %arg1 : memref<2x4xf32, strided<[?, 1], offset: 0>>
  return %0 : !quidditch_snitch.dma_token
}
