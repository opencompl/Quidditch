// RUN: quidditch-opt %s --quidditch-hoist-hal-ops-to-func | FileCheck %s

// TODO: The first argument is dead/redundant after hoisting.

// CHECK-LABEL: func @test(
// CHECK-SAME: %[[ARG0:.*]]: i32
// CHECK-SAME: %[[ARG1:.*]]: memref
// CHECK-SAME: attributes
// CHECK-SAME: llvm.bareptr
// CHECK-SAME: xdsl_generated
func.func @test() {
  %0 = hal.interface.constant.load[0] : i32
  // CHECK: arith.index_castui %[[ARG0]]
  %1 = arith.index_castui %0 : i32 to index
  %2 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%1) flags(ReadOnly) : memref<1x400xf64>
  return
}

// CHECK-LABEL: func @test$iree_to_xdsl()
// CHECK: %[[C:.*]] = hal.interface.constant.load[0] : i32
// CHECK: %[[CAST:.*]] = arith.index_castui %[[C]] : i32 to index
// CHECK: %[[MEMREF:.*]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%[[CAST]]) flags(ReadOnly)
// CHECK: call @test(%[[C]], %[[MEMREF]])
