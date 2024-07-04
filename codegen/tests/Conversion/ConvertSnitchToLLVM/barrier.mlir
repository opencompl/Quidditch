// RUN: quidditch-opt %s --quidditch-convert-snitch-to-llvm | FileCheck %s

// CHECK-LABEL: @test
func.func @test() {
  // CHECK: llvm.inline_asm has_side_effects "csrr x0, 0x7C2", "~{memory}"
  quidditch_snitch.barrier
  return
}
