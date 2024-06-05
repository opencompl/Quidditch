builtin.module @test_simple_add {
    func.func @add(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
      %init = tensor.empty() : tensor<4xf64>
      %out = linalg.generic
              {indexing_maps = [affine_map<(d0) -> (d0)>,
                                affine_map<(d0) -> (d0)>,
                                affine_map<(d0) -> (d0)>],
               iterator_types = ["parallel"]}
               ins(%arg0, %arg1 : tensor<4xf64>, tensor<4xf64>)
               outs(%init : tensor<4xf64>) {
      ^bb0(%in: f64 , %in_1: f64, %out: f64):
        %o = arith.addf %in, %in_1 : f64
        linalg.yield %o : f64
      } -> tensor<4xf64>
      func.return %out : tensor<4xf64>
    }
}
