// RUN: quidditch-opt %s -p "builtin.module(quidditch-specialize-dma-code)" | FileCheck %s

// CHECK-LABEL: @test(
// CHECK-SAME: %[[A:[[:alnum:]]+]]: memref<32xf32>
// CHECK-SAME: %[[B:[[:alnum:]]+]]: memref<32xf32>
// CHECK-SAME: attributes
// CHECK-SAME: quidditch_snitch.dma_specialization = @[[DMA_SPECIALIZATION:([[:alnum:]]|\$|_)+]]
func.func @test(%a : memref<32xf32>, %b : memref<32xf32>, %cond : i1) {
  %view = quidditch_snitch.l1_memory_view -> memref<512xi8>
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  // CHECK: memref.view
  // CHECK-NEXT: memref.view
  %a_l1 = memref.view %view[%c0][] : memref<512xi8> to memref<32xf32>
  %b_l1 = memref.view %view[%c256][] : memref<512xi8> to memref<32xf32>

  // CHECK-NEXT: dma.completed_token
  // CHECK-NEXT: dma.completed_token
  // CHECK-NEXT: quidditch_snitch.barrier
  dma.start_transfer from %a : memref<32xf32> to %a_l1 : memref<32xf32>
  %t = dma.start_transfer from %b : memref<32xf32> to %b_l1 : memref<32xf32>
  dma.wait_for_transfer %t

  // CHECK-NEXT: microkernel
  // CHECK: }
  quidditch_snitch.memref.microkernel(%a_l1, %b_l1) : memref<32xf32>, memref<32xf32> {
  ^bb0(%arg0 : memref<32xf32>, %arg1 : memref<32xf32>):
    linalg.abs ins(%arg0 : memref<32xf32>) outs(%arg1 : memref<32xf32>)
  }
  quidditch_snitch.microkernel_fence

  // CHECK-NEXT: quidditch_snitch.microkernel_fence
  // CHECK-NEXT: quidditch_snitch.barrier
  // CHECK-NEXT: dma.completed_token
  %t2 = dma.start_transfer from %b_l1 : memref<32xf32> to %b : memref<32xf32>
  // CHECK-NEXT: quidditch_snitch.barrier
  dma.wait_for_transfer %t2


  // CHECK: scf.if
  %r:2 = scf.if %cond -> (!dma.token, index) {
    // CHECK-NEXT: %[[C:.*]] = dma.completed_token
    %t3 = dma.start_transfer from %b_l1 : memref<32xf32> to %b : memref<32xf32>
    // CHECK-NEXT: %[[I:.*]] = quidditch_snitch.compute_core_index
    %i = quidditch_snitch.compute_core_index
    // CHECK-NEXT: yield %[[C]], %[[I]]
    scf.yield %t3, %i : !dma.token, index
  } else {
    // CHECK-NEXT: else
    // CHECK-NEXT: %[[C:.*]] = dma.completed_token
    %c = dma.completed_token
    // CHECK-NEXT: %[[I:.*]] = arith.constant
    %i = arith.constant 1 : index
    // CHECK-NEXT: yield %[[C]], %[[I]]
    scf.yield %c, %i : !dma.token, index
  }
  // CHECK: quidditch_snitch.barrier
  dma.wait_for_transfer %r#0
  // CHECK-NEXT: return
  return
}

// CHECK: @[[DMA_SPECIALIZATION]](
// CHECK-SAME: %[[A:[[:alnum:]]+]]: memref<32xf32>
// CHECK-SAME: %[[B:[[:alnum:]]+]]: memref<32xf32>

// CHECK: memref.view
// CHECK-NEXT: memref.view

// CHECK-NEXT: dma.start_transfer
// CHECK-NEXT: dma.start_transfer
// CHECK-NEXT: dma.wait_for_transfer
// CHECK-NEXT: quidditch_snitch.barrier

// CHECK-NEXT: quidditch_snitch.barrier
// CHECK-NEXT: dma.start_transfer
// CHECK-NEXT: dma.wait_for_transfer
// CHECK-NEXT: quidditch_snitch.barrier

// CHECK-NEXT: scf.if
// CHECK-NEXT: dma.start_transfer
// CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0
// CHECK-NEXT: yield %{{.*}}, %[[ZERO]] :
// CHECK-NEXT: else
// CHECK-NEXT: completed_token
// CHECK-NEXT: arith.constant
// CHECK-NEXT: yield
// CHECK: dma.wait_for_transfer
// CHECK-NEXT: quidditch_snitch.barrier

// CHECK-NEXT: return
