// RUN: quidditch-opt %s -p "builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(quidditch-remove-trivial-loops)))))" --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @test
hal.executable @test {
  // CHECK-LABEL: hal.executable.variant public @static
  hal.executable.variant @static target(#hal.executable.target<"", "", {compute_cores = 8 : i32}>) {
    builtin.module {
      // CHECK-LABEL: func @test
      func.func @test() {
        %0 = quidditch_snitch.cluster_index
        %1 = arith.constant 8 : index
        // CHECK-NOT: scf.for
        scf.for %arg3 = %0 to %1 step %1 {
          "test.op"(%arg3) : (index) -> ()
          scf.yield
        }
        return
      }
    }
  }
}
