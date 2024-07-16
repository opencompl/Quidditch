// RUN: quidditch-opt %s --quidditch-convert-to-llvm | FileCheck %s

// CHECK-LABEL: @test
// CHECK-SAME: %[[ALLOC_PTR:[[:alnum:]]+]]
// CHECK-SAME: %[[ALIGN_PTR:[[:alnum:]]+]]
// CHECK-SAME: %[[OFFSET:[[:alnum:]]+]]
func.func private @test(%arg0 : memref<32xi8, strided<[1], offset: ?>>) {
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
