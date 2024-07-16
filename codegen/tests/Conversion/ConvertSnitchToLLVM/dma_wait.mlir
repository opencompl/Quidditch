// RUN: quidditch-opt %s --quidditch-convert-snitch-to-llvm | FileCheck %s

// CHECK-LABEL: @test
func.func @test(%arg0 : !quidditch_snitch.dma_token) {
  // CHECK: %[[ARG0:.*]] = builtin.unrealized_conversion_cast
  // CHECK: scf.while
  // CHECK-NEXT: %[[ID:.*]] = llvm.inline_asm has_side_effects ".insn r 0x2b, 0, 0b100, $0, zero, zero
  // CHECK-SAME: "=r"
  // CHECK-SAME: -> i32
  // CHECK: %[[COND:.*]] = llvm.icmp "ult" %[[ID]], %[[ARG0]]
  // CHECK: scf.condition(%[[COND]])
  quidditch_snitch.wait_for_dma_transfers %arg0 : !quidditch_snitch.dma_token
  return
}
