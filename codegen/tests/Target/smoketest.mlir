// RUN: iree-compile %s --iree-hal-target-backends=quidditch --iree-input-demote-f64-to-f32=0 \
// RUN:   --iree-quidditch-static-library-output-path=%t.o \
// RUN:   --iree-quidditch-xdsl-opt-path=%xdsl-opt \
// RUN:   --iree-quidditch-toolchain-root=%quidditch-toolchain-root \
// RUN:   --output-format=vm-c \
// RUN:   --iree-vm-target-index-bits=32 \
// RUN:   -o /dev/null

builtin.module @test_simple_add {
    func.func @add(%arg0: tensor<128xf64>, %arg1: tensor<128xf64>) -> tensor<128xf64> {
      %init = tensor.empty() : tensor<128xf64>
      %out = linalg.generic
              {indexing_maps = [affine_map<(d0) -> (d0)>,
                                affine_map<(d0) -> (d0)>,
                                affine_map<(d0) -> (d0)>],
               iterator_types = ["parallel"]}
               ins(%arg0, %arg1 : tensor<128xf64>, tensor<128xf64>)
               outs(%init : tensor<128xf64>) {
      ^bb0(%in: f64 , %in_1: f64, %out: f64):
        %o = arith.addf %in, %in_1 : f64
        linalg.yield %o : f64
      } -> tensor<128xf64>
      func.return %out : tensor<128xf64>
    }
}
