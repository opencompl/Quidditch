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
