func.func @main$async_dispatch_0_matmul_transpose_b_1x400x161_f64() attributes {translation_info = #iree_codegen.translation_info<None>} {
  %c40 = arith.constant 40 : index
  %c400 = arith.constant 400 : index
  %cst = arith.constant 0.000000e+00 : f64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c515200 = arith.constant 515200 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x161xf64>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<400x161xf64>>
  %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c515200) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x400xf64>>
  %3 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<1x400xf64>>
  %4 = flow.dispatch.tensor.load %3, offsets = [0, 0], sizes = [1, 400], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<1x400xf64>> -> tensor<1x400xf64>
  %5 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1, 161], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x161xf64>> -> tensor<1x161xf64>
  %6 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [400, 161], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<400x161xf64>> -> tensor<400x161xf64>
  %7 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [1, 400], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<1x400xf64>> -> tensor<1x400xf64>
  %8 = bufferization.alloc_tensor() {memory_space = #quidditch_snitch.l1_encoding} : tensor<1x400xf64>
  %result, %token = quidditch_snitch.start_tensor_copy %8 to L1 : tensor<1x400xf64>
  %9 = quidditch_snitch.wait_for_tensor_copy %token, %result : tensor<1x400xf64>
  %10 = scf.forall (%arg0) = (0) to (400) step (50) shared_outs(%arg1 = %9) -> (tensor<1x400xf64>) {
    %extracted_slice = tensor.extract_slice %arg1[0, %arg0] [1, 50] [1, 1] : tensor<1x400xf64> to tensor<1x50xf64>
    %17 = linalg.fill ins(%cst : f64) outs(%extracted_slice : tensor<1x50xf64>) -> tensor<1x50xf64>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %17 into %arg1[0, %arg0] [1, 50] [1, 1] : tensor<1x50xf64> into tensor<1x400xf64>
    }
  }

  %result_0, %token_1 = quidditch_snitch.start_tensor_copy %5 to L1 : tensor<1x161xf64>
  %11 = quidditch_snitch.wait_for_tensor_copy %token_1, %result_0 : tensor<1x161xf64>

  %first_slice = tensor.extract_slice %6[0, 0] [40, 161] [1, 1] : tensor<400x161xf64> to tensor<40x161xf64>
  %first_result, %first_token = quidditch_snitch.start_tensor_copy %first_slice to L1 : tensor<40x161xf64>
  %storage = bufferization.alloc_tensor() {memory_space = #quidditch_snitch.l1_encoding} : tensor<40x161xf64>

  %12:4 = scf.for %arg0 = %c0 to %c400 step %c40 iter_args(%arg1 = %10, %iter_token = %first_token, %iter_tensor = %first_result, %next_storage = %storage) -> (tensor<1x400xf64>, !quidditch_snitch.dma_token, tensor<40x161xf64>, tensor<40x161xf64>) {
    %next = arith.addi %arg0, %c40 : index
    %next_slice = tensor.extract_slice %6[%next, 0] [40, 161] [1, 1] : tensor<400x161xf64> to tensor<40x161xf64>
    %next_result = linalg.copy ins(%next_slice : tensor<40x161xf64>) outs(%next_storage : tensor<40x161xf64>) -> tensor<40x161xf64>
    %next_token = quidditch_snitch.completed_token

    %extracted_slice_8 = tensor.extract_slice %arg1[0, %arg0] [1, 40] [1, 1] : tensor<1x400xf64> to tensor<1x40xf64>
    %result_11, %token_12 = quidditch_snitch.start_tensor_copy %extracted_slice_8 to L1 : tensor<1x40xf64>
    %17 = quidditch_snitch.wait_for_tensor_copy %iter_token, %iter_tensor : tensor<40x161xf64>
    %18 = quidditch_snitch.wait_for_tensor_copy %token_12, %result_11 : tensor<1x40xf64>
    %19 = scf.forall (%arg2) = (0) to (40) step (5) shared_outs(%arg3 = %18) -> (tensor<1x40xf64>) {
      %extracted_slice_13 = tensor.extract_slice %17[%arg2, 0] [5, 161] [1, 1] : tensor<40x161xf64> to tensor<5x161xf64>
      %extracted_slice_14 = tensor.extract_slice %arg3[0, %arg2] [1, 5] [1, 1] : tensor<1x40xf64> to tensor<1x5xf64>
      %20 = linalg.matmul_transpose_b {lowering_config = #quidditch_snitch.lowering_config<l1_tiles = [0, 40]>} ins(%11, %extracted_slice_13 : tensor<1x161xf64>, tensor<5x161xf64>) outs(%extracted_slice_14 : tensor<1x5xf64>) -> tensor<1x5xf64>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %20 into %arg3[0, %arg2] [1, 5] [1, 1] : tensor<1x5xf64> into tensor<1x40xf64>
      }
    }
    %inserted_slice = tensor.insert_slice %19 into %arg1[0, %arg0] [1, 40] [1, 1] : tensor<1x40xf64> into tensor<1x400xf64>
    scf.yield %inserted_slice, %next_token, %next_result, %17 : tensor<1x400xf64>, !quidditch_snitch.dma_token, tensor<40x161xf64>, tensor<40x161xf64>
  }



  %result_2, %token_3 = quidditch_snitch.start_tensor_copy %12#0 to L1 : tensor<1x400xf64>
  %13 = quidditch_snitch.wait_for_tensor_copy %token_3, %result_2 : tensor<1x400xf64>
  %result_4, %token_5 = quidditch_snitch.start_tensor_copy %7 to L1 : tensor<1x400xf64>
  %14 = quidditch_snitch.wait_for_tensor_copy %token_5, %result_4 : tensor<1x400xf64>
  %result_6, %token_7 = quidditch_snitch.start_tensor_copy %4 to L1 : tensor<1x400xf64>
  %15 = quidditch_snitch.wait_for_tensor_copy %token_7, %result_6 : tensor<1x400xf64>
  %16 = scf.forall (%arg0) = (0) to (400) step (50) shared_outs(%arg1 = %15) -> (tensor<1x400xf64>) {
    %extracted_slice = tensor.extract_slice %13[0, %arg0] [1, 50] [1, 1] : tensor<1x400xf64> to tensor<1x50xf64>
    %extracted_slice_8 = tensor.extract_slice %14[0, %arg0] [1, 50] [1, 1] : tensor<1x400xf64> to tensor<1x50xf64>
    %extracted_slice_9 = tensor.extract_slice %arg1[0, %arg0] [1, 50] [1, 1] : tensor<1x400xf64> to tensor<1x50xf64>
    %17 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %extracted_slice_8 : tensor<1x50xf64>, tensor<1x50xf64>) outs(%extracted_slice_9 : tensor<1x50xf64>) {
    ^bb0(%in: f64, %in_10: f64, %out: f64):
      %18 = arith.addf %in, %in_10 : f64
      linalg.yield %18 : f64
    } -> tensor<1x50xf64>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %17 into %arg1[0, %arg0] [1, 50] [1, 1] : tensor<1x50xf64> into tensor<1x400xf64>
    }
  }
  flow.dispatch.tensor.store %16, %3, offsets = [0, 0], sizes = [1, 400], strides = [1, 1] : tensor<1x400xf64> -> !flow.dispatch.tensor<writeonly:tensor<1x400xf64>>
  return
}
