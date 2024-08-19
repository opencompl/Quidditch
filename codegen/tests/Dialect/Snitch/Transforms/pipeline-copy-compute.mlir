// RUN: quidditch-opt %s -p "builtin.module(func.func(quidditch-pipeline-copy-compute))" | FileCheck %s

// CHECK-LABEL: @test
// CHECK-SAME: %[[ARG0:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG1:[[:alnum:]]+]]
// CHECK-SAME: %[[ARG2:[[:alnum:]]+]]
func.func @test(%arg0: index, %extracted_slice : tensor<1x100xf64>, %14 : tensor<1200x400xf64>) -> tensor<1x1200xf64> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C40:.*]] = arith.constant 40
  // CHECK-DAG: %[[C1200:.*]] = arith.constant 1200
  %c0 = arith.constant 0 : index
  %c40 = arith.constant 40 : index
  %c1200 = arith.constant 1200 : index
  // CHECK: %[[EMPTY:.*]] = tensor.empty()
  %arg1 = tensor.empty() : tensor<1x1200xf64>
  // CHECK: pipeline %[[C0]] to %[[C1200]] step %[[C40]] inits(%[[EMPTY]])
  %24 = scf.for %arg2 = %c0 to %c1200 step %c40 iter_args(%arg3 = %arg1) -> (tensor<1x1200xf64>) {
    // CHECK: ^{{.*}}(%[[IV:.*]]: index, %[[ITER:[[:alnum:]]+]]:
    // CHECK: %[[RESULT0:.*]], %[[TOKEN0:.*]] = quidditch_snitch.start_tensor_copy %[[ARG1]]
    // CHECK: %[[SLICE1:.*]] = tensor.extract_slice %[[ARG2]][%[[IV]], %[[ARG0]]]
    // CHECK: %[[RESULT1:.*]], %[[TOKEN1:.*]] = quidditch_snitch.start_tensor_copy %[[SLICE1]]
    // CHECK: %[[SLICE2:.*]] = tensor.extract_slice %[[ITER]][0, %[[IV]]]
    // CHECK: %[[RESULT2:.*]], %[[TOKEN2:.*]] = quidditch_snitch.start_tensor_copy %[[SLICE2]]
    // CHECK: pipeline_yield %[[ITER]], %[[RESULT0:.*]], %[[TOKEN0]], %[[SLICE1]], %[[RESULT1]], %[[TOKEN1]], %[[SLICE2]], %[[RESULT2]], %[[TOKEN2]]

    %extracted_slice_6 = tensor.extract_slice %14[%arg2, %arg0] [40, 100] [1, 1] : tensor<1200x400xf64> to tensor<40x100xf64>
    %extracted_slice_7 = tensor.extract_slice %arg3[0, %arg2] [1, 40] [1, 1] : tensor<1x1200xf64> to tensor<1x40xf64>
    %result_8, %token_9 = quidditch_snitch.start_tensor_copy %extracted_slice to L1 : tensor<1x100xf64> -> tensor<1x100xf64>
    %25 = quidditch_snitch.wait_for_tensor_copy of %extracted_slice : tensor<1x100xf64> to %result_8 using %token_9 -> tensor<1x100xf64>
    %result_10, %token_11 = quidditch_snitch.start_tensor_copy %extracted_slice_6 to L1 : tensor<40x100xf64> -> tensor<40x100xf64>
    %26 = quidditch_snitch.wait_for_tensor_copy of %extracted_slice_6 : tensor<40x100xf64> to %result_10 using %token_11 -> tensor<40x100xf64>
    %result_12, %token_13 = quidditch_snitch.start_tensor_copy %extracted_slice_7 to L1 : tensor<1x40xf64> -> tensor<1x40xf64>
    %27 = quidditch_snitch.wait_for_tensor_copy of %extracted_slice_7 : tensor<1x40xf64> to %result_12 using %token_13 -> tensor<1x40xf64>

    // CHECK: ^{{.*}}(
    // CHECK-SAME: %[[IV:[[:alnum:]]+]]
    // CHECK-SAME: %[[ITER:[[:alnum:]]+]]
    // CHECK-SAME: %[[RESULT0:[[:alnum:]]+]]
    // CHECK-SAME: %[[TOKEN0:[[:alnum:]]+]]
    // CHECK-SAME: %[[SLICE1:[[:alnum:]]+]]
    // CHECK-SAME: %[[RESULT1:[[:alnum:]]+]]
    // CHECK-SAME: %[[TOKEN1:[[:alnum:]]+]]
    // CHECK-SAME: %[[SLICE2:[[:alnum:]]+]]
    // CHECK-SAME: %[[RESULT2:[[:alnum:]]+]]
    // CHECK-SAME: %[[TOKEN2:[[:alnum:]]+]]
    // CHECK: %[[OPA:.*]] = quidditch_snitch.wait_for_tensor_copy of %[[ARG1]]
    // CHECK-SAME: to %[[RESULT0]]
    // CHECK-SAME: using %[[TOKEN0]]
    // CHECK: %[[OPB:.*]] = quidditch_snitch.wait_for_tensor_copy of %[[SLICE1]]
    // CHECK-SAME: to %[[RESULT1]]
    // CHECK-SAME: using %[[TOKEN1]]
    // CHECK: %[[OPC:.*]] = quidditch_snitch.wait_for_tensor_copy of %[[SLICE2]]
    // CHECK-SAME: to %[[RESULT2]]
    // CHECK-SAME: using %[[TOKEN2]]
    // CHECK: %[[RES:.*]] = linalg.matmul_transpose_b
    // CHECK-SAME: ins(%[[OPA]], %[[OPB]] :
    // CHECK-SAME: outs(%[[OPC]] :
    // CHECK: %[[YIELDED:.*]] = tensor.insert_slice %[[RES]] into %[[ITER]]
    // CHECK: pipeline_yield %[[YIELDED]]

    %28 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<workgroup_tiles = [0, 0, 100], l1_tiles = [0, 40], dual_buffer = true>} ins(%25, %26 : tensor<1x100xf64>, tensor<40x100xf64>) outs(%27 : tensor<1x40xf64>) -> tensor<1x40xf64>
    %inserted_slice = tensor.insert_slice %28 into %arg3[0, %arg2] [1, 40] [1, 1] : tensor<1x40xf64> into tensor<1x1200xf64>
    scf.yield %inserted_slice : tensor<1x1200xf64>
  }
  return %24 : tensor<1x1200xf64>
}
