// RUN: quidditch-opt %s --quidditch-convert-to-llvm | FileCheck %s

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
func.func private @test(%arg0 : !dma.token, %arg1 : !dma.token) -> !dma.token {
  // CHECK: %[[TOKEN:.*]] = llvm.intr.umax(%[[ARG0]], %[[ARG1]])
  %token = dma.combine_tokens %arg0, %arg1
  // CHECK: llvm.return %[[TOKEN]]
  return %token : !dma.token
}

// CHECK-LABEL: @test_empty(
func.func private @test_empty() -> !dma.token {
  // CHECK: %[[TOKEN:.*]] = llvm.mlir.constant(0 :
  %token = dma.combine_tokens
  // CHECK: return %[[TOKEN]]
  return %token : !dma.token
}
