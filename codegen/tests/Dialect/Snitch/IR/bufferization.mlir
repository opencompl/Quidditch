// RUN: quidditch-opt %s --one-shot-bufferize | FileCheck %s

// CHECK-LABEL: @pipeline_op(
func.func @pipeline_op(%arg0_dim : index) -> tensor<?xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[C10:.*]] = arith.constant 10
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %arg0 = tensor.empty(%arg0_dim) : tensor<?xf32>

  // CHECK: pipeline %[[C0]] to %[[C10]] step %[[C1]] {
  %t = quidditch_snitch.pipeline %c0 to %c10 step %c1 inits(%arg0) -> tensor<?xf32> {
  // CHECK: ^{{.*}}(%[[IV:.*]]: index):
  ^bb0(%iv: index, %tensor: tensor<?xf32>):
    // CHECK: quidditch_snitch.pipeline_yield
    quidditch_snitch.pipeline_yield %tensor : tensor<?xf32>
  }, {
  // CHECK: ^{{.*}}(%[[IV:.*]]: index, %{{.*}}: memref<?xf32{{.*}}>):
  ^bb0(%iv: index, %tensor: tensor<?xf32>):
    quidditch_snitch.pipeline_yield %tensor : tensor<?xf32>
  // CHECK-NEXT: }
  }
  return %t : tensor<?xf32>
}

// CHECK: func @microkernel(
func.func @microkernel(%arg0 : tensor<32xf32>) -> tensor<32xf32> {
  // CHECK: %[[ARG0:.*]] = bufferization.to_memref
  // CHECK: %[[INIT:.*]] = memref.alloc()
  %init = tensor.empty() : tensor<32xf32>
  // CHECK: quidditch_snitch.memref.microkernel(%[[ARG0]], %[[INIT]])
  %0 = quidditch_snitch.tensor.microkernel -> tensor<32xf32> {
    // CHECK-NEXT: ^{{.*}}(%[[ARG1:.*]]: memref<{{.*}}>, %[[ARG2:.*]]: memref<{{.*}}>):

    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[ARG1]] : {{.*}}) outs(%[[ARG2]] : {{.*}})
    %1 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%arg0 : tensor<32xf32>) outs(%init : tensor<32xf32>) {
    ^bb0(%in : f32, %out : f32):
      %o = arith.addf %in, %in : f32
      linalg.yield %o : f32
    } -> tensor<32xf32>

    quidditch_snitch.microkernel_yield %1 : tensor<32xf32>
  }
  // CHECK: %[[RET:.*]] = bufferization.to_tensor %[[INIT]]
  // CHECK: return %[[RET]]
  return %0 : tensor<32xf32>
}

// CHECK: func @sync_tensor(
func.func @sync_tensor() -> tensor<32xf32> {
  // CHECK: %[[MEMREF:.*]] = memref.alloc
  %arg0 = bufferization.alloc_tensor() : tensor<32xf32>

  // CHECK: quidditch_snitch.microkernel_fence
  // CHECK: %[[R:.*]] = bufferization.to_tensor %[[MEMREF]]
  %r = quidditch_snitch.sync_tensor %arg0 : tensor<32xf32>

  // CHECK: return %[[R]]
  return %r : tensor<32xf32>
}
