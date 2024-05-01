builtin.module @test_simple_add {
    func.func @add(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<128xf32> {
      %init = tensor.empty() : tensor<128xf32>
      %out = linalg.generic
              {indexing_maps = [affine_map<(d0) -> (d0)>,
                                affine_map<(d0) -> (d0)>,
                                affine_map<(d0) -> (d0)>],
               iterator_types = ["parallel"]}
               ins(%arg0, %arg1 : tensor<128xf32>, tensor<128xf32>)
               outs(%init : tensor<128xf32>) {
      ^bb0(%in: f32 , %in_1: f32, %out: f32):
        %o = arith.addf %in, %in_1 : f32
        linalg.yield %o : f32
      } -> tensor<128xf32>
      func.return %out : tensor<128xf32>
    }
}
