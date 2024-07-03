// RUN: quidditch-opt %s -p "builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(quidditch-lower-forall-op)))))" -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @test
hal.executable @test {
  // CHECK-LABEL: hal.executable.variant public @static
  hal.executable.variant @static target(#hal.executable.target<"", "", {compute_cores = 8 : i32}>) {
    builtin.module {
      // CHECK-LABEL: func @test
      // CHECK-SAME: %[[LB:[[:alnum:]]+]]
      // CHECK-SAME: %[[UB:[[:alnum:]]+]]
      // CHECK-SAME: %[[STEP:[[:alnum:]]+]]
      func.func @test(%lb : index, %ub : index, %step : index) {
        // CHECK: %[[ID:.*]] = quidditch_snitch.cluster_index
        // CHECK: %[[LB2:.*]] = arith.muli %[[ID]], %[[STEP]]
        // CHECK: %[[LB3:.*]] = arith.addi %[[LB]], %[[LB2]]
        // CHECK: %[[CORES:.*]] = arith.constant 8
        // CHECK: %[[STEP2:.*]] = arith.muli %[[STEP]], %[[CORES]]
        // CHECK: scf.for %[[IV:.*]] = %[[LB3]] to %[[UB]] step %[[STEP2]]
        scf.forall (%iter) = (%lb) to (%ub) step (%step) {
          // CHECK-NEXT: "test.op"(%[[IV]])
          "test.op"(%iter) : (index) -> ()
        }
        return
      }
    }
  }
}
