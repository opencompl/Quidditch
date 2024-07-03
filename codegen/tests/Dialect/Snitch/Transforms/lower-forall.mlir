// RUN: quidditch-opt %s -p "builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(quidditch-lower-forall-op)))))" --allow-unregistered-dialect | FileCheck %s

// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0 + d1 * d2)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<(d0) -> (d0 * 8)>

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
        // CHECK: %[[NLB:.*]] = affine.apply #[[$MAP1]](%[[LB]], %[[ID]], %[[STEP]])
        // CHECK: %[[NSTEP:.*]] = affine.apply #[[$MAP2]](%[[STEP]])
        // CHECK: scf.for %[[IV:.*]] = %[[NLB]] to %[[UB]] step %[[NSTEP]]
        scf.forall (%iter) = (%lb) to (%ub) step (%step) {
          // CHECK-NEXT: "test.op"(%[[IV]])
          "test.op"(%iter) : (index) -> ()
        }
        return
      }
    }
  }
}
