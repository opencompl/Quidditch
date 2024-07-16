// RUN: quidditch-opt %s --quidditch-convert-to-llvm| FileCheck %s

// CHECK-LABEL: @test
func.func private @test() {
  // CHECK: llvm.inline_asm has_side_effects "fmv.x.w $0, fa0\0Amv $0, $0", "=r,~{memory}"
  quidditch_snitch.microkernel_fence
  return
}
