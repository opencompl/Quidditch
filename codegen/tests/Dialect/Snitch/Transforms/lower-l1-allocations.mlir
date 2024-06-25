// RUN: quidditch-opt %s -p "builtin.module(func.func(quidditch-lower-l1-allocations))" | FileCheck %s

// CHECK-LABEL: @test()
// CHECK-SAME: -> (memref<32xf32>, memref<64xf64>)
func.func @test() -> (memref<32xf32, #quidditch_snitch.l1_encoding>,
                      memref<64xf64, #quidditch_snitch.l1_encoding>) {
  // CHECK: %[[VIEW:.*]] = quidditch_snitch.l1_memory_view
  // CHECK: %[[OFFSET:.*]] = arith.constant 0
  // CHECK: %[[ALLOCA0:.*]] = memref.view %[[VIEW]][%[[OFFSET]]][] : memref<{{.*}}xi8> to memref<32xf32>
  %0 = memref.alloca() : memref<32xf32, #quidditch_snitch.l1_encoding>
  // CHECK: %[[OFFSET:.*]] = arith.constant 128
  // CHECK: %[[ALLOCA1:.*]] = memref.view %[[VIEW]][%[[OFFSET]]][] : memref<{{.*}}xi8> to memref<64xf64>
  %1 = memref.alloca() : memref<64xf64, #quidditch_snitch.l1_encoding>
  // CHECK: return %[[ALLOCA0]], %[[ALLOCA1]]
  return %0, %1 : memref<32xf32, #quidditch_snitch.l1_encoding>, memref<64xf64, #quidditch_snitch.l1_encoding>
}
