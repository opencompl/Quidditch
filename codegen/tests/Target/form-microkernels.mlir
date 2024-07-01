// RUN: quidditch-opt %s -p "builtin.module(func.func(quidditch-form-microkernels))" --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @linalgs_in_scf
func.func @linalgs_in_scf(%cond : i1) {
  %cst0 = arith.constant 0.0 : f32
  // CHECK: scf.if
  scf.if %cond {
    // CHECK: quidditch_snitch.memref.microkernel
    // CHECK-SAME: {
    // CHECK: linalg.fill
    // CHECK-NEXT: }
    %empty = memref.alloc() : memref<32xf32>
   linalg.fill ins(%cst0 : f32) outs(%empty : memref<32xf32>)
  }
  return
}
