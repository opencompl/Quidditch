// RUN: quidditch-opt %s --quidditch-convert-to-llvm | FileCheck %s

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
func.func private @test(%arg0 : !dma.token) {
  // CHECK: llvm.br ^[[BODY:[[:alnum:]]+]]
  // CHECK: ^[[BODY]]:
  // CHECK-NEXT: %[[ID:.*]] = llvm.inline_asm has_side_effects ".insn r 0x2b, 0, 0b100, $0, zero, zero
  // CHECK-SAME: "=r"
  // CHECK-SAME: -> i32
  // CHECK: %[[COND:.*]] = llvm.icmp "ult" %[[ID]], %[[ARG0]]
  // CHECK: llvm.cond_br %[[COND]], ^[[BODY]], ^[[CONT:[[:alnum:]]+]]
  // CHECK: ^[[CONT]]:
  dma.wait_for_transfer %arg0
  // CHECK-NEXT: llvm.return
  return
}
