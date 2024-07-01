// RUN: quidditch-opt %s --quidditch-convert-snitch-to-llvm | FileCheck %s

// CHECK-LABEL: @test
func.func @test(%arg0 : memref<32xi8, strided<[1], offset: ?>>) {
  // CHECK: %[[ARG0:.*]] = builtin.unrealized_conversion_cast
  // CHECK: %[[ALIGN_PTR:.*]] = llvm.extractvalue %[[ARG0]][1]
  // CHECK: %[[OFFSET:.*]] = llvm.extractvalue %[[ARG0]][2]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[ALIGN_PTR]][%[[OFFSET]]]
  // CHECK: llvm.call @name(%[[GEP]])
  quidditch_snitch.call_microkernel "name"(%arg0) : memref<32xi8, strided<[1], offset: ?>> [{
    "the assembly"
  }]
  return
}

// CHECK: llvm.func @name(!llvm.ptr)
// CHECK-SAME: hal.import.bitcode
// CHECK-SAME: quidditch_snitch.riscv_assembly = "the assembly"
