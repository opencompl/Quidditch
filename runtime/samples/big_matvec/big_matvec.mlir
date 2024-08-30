builtin.module @big_matvec {
    func.func @test32(%arg0: tensor<1x400xf64>, %arg1: tensor<320x400xf64>) -> tensor<1x320xf64> {
      %init = tensor.empty() : tensor<1x320xf64>
      %out = linalg.matmul_transpose_b {
            lowering_config = #quidditch_snitch.lowering_config<
                l1_tiles = [0, 32, 80],
                l1_tiles_interchange = [2, 0, 1],
                dual_buffer = true
            >
        }
        ins(%arg0, %arg1 : tensor<1x400xf64>, tensor<320x400xf64>)
        outs(%init : tensor<1x320xf64>) -> tensor<1x320xf64>
      %out2 = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ], iterator_types = ["parallel", "parallel"]
      } ins(%out : tensor<1x320xf64>) outs(%out : tensor<1x320xf64>) {
      ^bb0(%element : f64, %outs : f64):
        %bias = arith.constant 5.0 : f64
        %added = arith.addf %element, %bias : f64
        linalg.yield %added : f64
      } -> tensor<1x320xf64>
      func.return %out2 : tensor<1x320xf64>
    }

    func.func @test40(%arg0: tensor<1x400xf64>, %arg1: tensor<320x400xf64>) -> tensor<1x320xf64> {
      %init = tensor.empty() : tensor<1x320xf64>
      %out = linalg.matmul_transpose_b {
            lowering_config = #quidditch_snitch.lowering_config<
                l1_tiles = [0, 40, 80],
                l1_tiles_interchange = [2, 0, 1],
                dual_buffer = true
            >
        }
        ins(%arg0, %arg1 : tensor<1x400xf64>, tensor<320x400xf64>)
        outs(%init : tensor<1x320xf64>) -> tensor<1x320xf64>
      %out2 = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ], iterator_types = ["parallel", "parallel"]
      } ins(%out : tensor<1x320xf64>) outs(%out : tensor<1x320xf64>) {
      ^bb0(%element : f64, %outs : f64):
        %bias = arith.constant 5.0 : f64
        %added = arith.addf %element, %bias : f64
        linalg.yield %added : f64
      } -> tensor<1x320xf64>
      func.return %out2 : tensor<1x320xf64>
    }

    func.func @test64(%arg0: tensor<1x400xf64>, %arg1: tensor<320x400xf64>) -> tensor<1x320xf64> {
      %init = tensor.empty() : tensor<1x320xf64>
      %out = linalg.matmul_transpose_b {
            lowering_config = #quidditch_snitch.lowering_config<
                l1_tiles = [0, 64, 80],
                l1_tiles_interchange = [2, 0, 1],
                dual_buffer = true
            >
        }
        ins(%arg0, %arg1 : tensor<1x400xf64>, tensor<320x400xf64>)
        outs(%init : tensor<1x320xf64>) -> tensor<1x320xf64>
      %out2 = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ], iterator_types = ["parallel", "parallel"]
      } ins(%out : tensor<1x320xf64>) outs(%out : tensor<1x320xf64>) {
      ^bb0(%element : f64, %outs : f64):
        %bias = arith.constant 5.0 : f64
        %added = arith.addf %element, %bias : f64
        linalg.yield %added : f64
      } -> tensor<1x320xf64>
      func.return %out2 : tensor<1x320xf64>
    }

    func.func @test32_100(%arg0: tensor<1x400xf64>, %arg1: tensor<320x400xf64>) -> tensor<1x320xf64> {
      %init = tensor.empty() : tensor<1x320xf64>
      %out = linalg.matmul_transpose_b {
            lowering_config = #quidditch_snitch.lowering_config<
                l1_tiles = [0, 32, 100],
                l1_tiles_interchange = [2, 0, 1],
                dual_buffer = true
            >
        }
        ins(%arg0, %arg1 : tensor<1x400xf64>, tensor<320x400xf64>)
        outs(%init : tensor<1x320xf64>) -> tensor<1x320xf64>
      %out2 = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ], iterator_types = ["parallel", "parallel"]
      } ins(%out : tensor<1x320xf64>) outs(%out : tensor<1x320xf64>) {
      ^bb0(%element : f64, %outs : f64):
        %bias = arith.constant 5.0 : f64
        %added = arith.addf %element, %bias : f64
        linalg.yield %added : f64
      } -> tensor<1x320xf64>
      func.return %out2 : tensor<1x320xf64>
    }

    func.func @test40_100(%arg0: tensor<1x400xf64>, %arg1: tensor<320x400xf64>) -> tensor<1x320xf64> {
      %init = tensor.empty() : tensor<1x320xf64>
      %out = linalg.matmul_transpose_b {
            lowering_config = #quidditch_snitch.lowering_config<
                l1_tiles = [0, 40, 100],
                l1_tiles_interchange = [2, 0, 1],
                dual_buffer = true
            >
        }
        ins(%arg0, %arg1 : tensor<1x400xf64>, tensor<320x400xf64>)
        outs(%init : tensor<1x320xf64>) -> tensor<1x320xf64>
      %out2 = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
        ], iterator_types = ["parallel", "parallel"]
      } ins(%out : tensor<1x320xf64>) outs(%out : tensor<1x320xf64>) {
      ^bb0(%element : f64, %outs : f64):
        %bias = arith.constant 5.0 : f64
        %added = arith.addf %element, %bias : f64
        linalg.yield %added : f64
      } -> tensor<1x320xf64>
      func.return %out2 : tensor<1x320xf64>
    }
}
