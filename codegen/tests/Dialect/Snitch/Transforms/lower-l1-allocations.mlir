// RUN: quidditch-opt %s -p "builtin.module(func.func(quidditch-lower-l1-allocations))" | FileCheck %s

// CHECK-LABEL: @test()
// CHECK-SAME: -> (memref<32xf32>, memref<2x64xf64, strided<[65, 1]>>)
func.func @test() -> (memref<32xf32, #quidditch_snitch.l1_encoding>,
                      memref<2x64xf64, strided<[65, 1]>, #quidditch_snitch.l1_encoding>) {
  // CHECK: %[[VIEW:.*]] = quidditch_snitch.l1_memory_view
  // CHECK: %[[OFFSET:.*]] = arith.constant 0
  // CHECK: %[[VIEW0:.*]] = memref.view %[[VIEW]][%[[OFFSET]]][] : memref<{{.*}}xi8> to memref<32xf32>
  // CHECK: %[[ALLOCA0:.*]] = memref.reinterpret_cast %[[VIEW0]]
  // CHECK-SAME: offset: [0]
  // CHECK-SAME: sizes: [32]
  // CHECK-SAME: strides: [1]
  %0 = memref.alloca() : memref<32xf32, #quidditch_snitch.l1_encoding>
  // CHECK: %[[OFFSET:.*]] = arith.constant 128
  // CHECK: %[[VIEW1:.*]] = memref.view %[[VIEW]][%[[OFFSET]]][] : memref<{{.*}}xi8> to memref<129xf64>
  // CHECK: %[[ALLOCA1:.*]] = memref.reinterpret_cast %[[VIEW1]]
  // CHECK-SAME: offset: [0]
  // CHECK-SAME: sizes: [2, 64]
  // CHECK-SAME: strides: [65, 1]
  %1 = memref.alloca() : memref<2x64xf64, strided<[65, 1]>, #quidditch_snitch.l1_encoding>
  // CHECK: return %[[ALLOCA0]], %[[ALLOCA1]]
  return %0, %1 : memref<32xf32, #quidditch_snitch.l1_encoding>, memref<2x64xf64, strided<[65, 1]>, #quidditch_snitch.l1_encoding>
}
