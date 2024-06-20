// RUN: quidditch-opt %s -p "builtin.module(hal.executable(hal.executable.variant(quidditch-disable-variant)))" | FileCheck %s

hal.executable @test {
  // CHECK-LABEL: hal.executable.variant public @static
  hal.executable.variant @static target(#hal.executable.target<"", "", {}>) {
    // CHECK-NEXT: hal.executable.condition
    // CHECK-NEXT: %[[FALSE:.*]] = arith.constant false
    // CHECK-NEXT: hal.return %[[FALSE]]

    builtin.module {
      func.func @test() attributes { quidditch_snitch.xdsl_compilation_failed } {
        return
      }
    }
  }
}
