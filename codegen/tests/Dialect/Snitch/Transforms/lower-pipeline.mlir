// RUN: quidditch-opt %s --quidditch-lower-pipeline-op | FileCheck %s

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<() -> (0)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<([[D0:.*]]) -> (([[D0]] floordiv 40) mod 2)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<([[D0:.*]]) -> ([[D0]])>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<([[D0:.*]]) -> ([[D0]] - 40)>
// CHECK-DAG: #[[$MAP4:.*]] = affine_map<() -> (1200)>
// CHECK-DAG: #[[$MAP5:.*]] = affine_map<() -> (1160)>

// CHECK-LABEL: @test
func.func @test(
  %arg0 : index,
  %9 : memref<1200x400xf64, strided<[400, 1], offset: ?>>,
  %alloca : memref<1x1200xf64, #quidditch_snitch.l1_encoding>,
  %alloca2 : memref<1x100xf64, #quidditch_snitch.l1_encoding>,
  %out : memref<1x40xf64, #quidditch_snitch.l1_encoding>
) {
  // CHECK-DAG: %[[LB:.*]] = arith.constant 0
  // CHECK-DAG: %[[UB:.*]] = arith.constant 1200
  // CHECK-DAG: %[[STEP:.*]] = arith.constant 40
  %c0 = arith.constant 0 : index
  %c1200 = arith.constant 1200 : index
  %c40 = arith.constant 40 : index

  // CHECK: %[[ALLOCA0:.*]] = memref.alloca()
  // CHECK: %[[ALLOCA1:.*]] = memref.alloca()

  // Stage 0 ramp up.
  // CHECK: %[[IV:.*]] = affine.apply #[[$MAP0]]()
  // CHECK: memref.subview %{{.*}}[%[[IV]], %{{.*}}]
  // CHECK: %[[CYCLE:.*]] = affine.apply #[[$MAP1]](%[[IV]])
  // CHECK: %[[ALLOCA:.*]] = scf.index_switch %[[CYCLE]]
  // CHECK-NEXT: case 0
  // CHECK-NEXT: yield %[[ALLOCA0]]
  // CHECK: default
  // CHECK-NEXT: yield %[[ALLOCA1]]
  // CHECK: %[[TOKEN:.*]] = dma.start_transfer from %{{.*}} to %[[ALLOCA]]

  // Full pipeline.
  // CHECK: %[[NEW_LB:.*]] = arith.addi %[[LB]], %[[STEP]]
  // CHECK: %[[LAST:.*]]:2 = scf.for %[[IV:.*]] = %[[NEW_LB]] to %[[UB]] step %[[STEP]] iter_args(%[[YIELDED0:.*]] = %[[ALLOCA]], %[[YIELDED1:.*]] = %[[TOKEN]])
  quidditch_snitch.pipeline %c0 to %c1200 step %c40 {
  ^bb0(%arg1: index):
    // CHECK: %[[STAGE0_IV:.*]] = affine.apply #[[$MAP2]](%[[IV]])
    // CHECK: memref.subview %{{.*}}[%[[STAGE0_IV]], %{{.*}}]
    // CHECK: %[[NEXT_YIELDED:.*]] = scf.index_switch

    %subview_3 = memref.subview %9[%arg1, %arg0] [40, 100] [1, 1] : memref<1200x400xf64, strided<[400, 1], offset: ?>> to memref<40x100xf64, strided<[400, 1], offset: ?>>
    %alloca_4 = memref.alloca() {alignment = 64 : i64} : memref<40x100xf64, #quidditch_snitch.l1_encoding>
    %16 = dma.start_transfer from %subview_3 : memref<40x100xf64, strided<[400, 1], offset: ?>> to %alloca_4 : memref<40x100xf64, #quidditch_snitch.l1_encoding>
    quidditch_snitch.pipeline_yield %alloca_4, %16 : memref<40x100xf64, #quidditch_snitch.l1_encoding>, !dma.token
  }, {
  ^bb0(%arg1: index, %arg2: memref<40x100xf64, #quidditch_snitch.l1_encoding>, %arg3: !dma.token):
    // CHECK: %[[STAGE1_IV:.*]] = affine.apply #[[$MAP3]](%[[IV]])
    // CHECK: memref.subview %{{.*}}[0, %[[STAGE1_IV]]]
    // CHECK: wait_for_transfer %[[YIELDED1]]
    // CHECK: linalg.matmul_transpose_b ins(%{{.*}}, %[[YIELDED0]] : {{.*}})
    // CHECK: yield %[[NEXT_YIELDED]], %{{.*}} :

    %subview_3 = memref.subview %alloca[0, %arg1] [1, 40] [1, 1] : memref<1x1200xf64, #quidditch_snitch.l1_encoding> to memref<1x40xf64, strided<[1200, 1], offset: ?>, #quidditch_snitch.l1_encoding>
    dma.wait_for_transfer %arg3
    linalg.matmul_transpose_b
      ins(%alloca2, %arg2 : memref<1x100xf64, #quidditch_snitch.l1_encoding>, memref<40x100xf64, #quidditch_snitch.l1_encoding>)
      outs(%out : memref<1x40xf64, #quidditch_snitch.l1_encoding>)
  }
  // CHECK: %[[IV:.*]] = affine.apply #[[$MAP4]]()
  // CHECK: %[[STAGE1_IV:.*]] = affine.apply #[[$MAP5]]()
  // CHECK: memref.subview %{{.*}}[0, %[[STAGE1_IV]]]
  // CHECK: wait_for_transfer %[[LAST]]#1
  // CHECK: linalg.matmul_transpose_b ins(%{{.*}}, %[[LAST]]#0 : {{.*}})
  return
}
