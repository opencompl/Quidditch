// RUN: quidditch-opt %s -p "builtin.module(quidditch-specialize-dma-code)" | FileCheck %s

// CHECK-LABEL: @test(
// CHECK-SAME: %[[A:[[:alnum:]]+]]: memref<32xf32>
// CHECK-SAME: %[[B:[[:alnum:]]+]]: memref<32xf32>
// CHECK-SAME: attributes
// CHECK-SAME: quidditch_snitch.dma_specialization = @[[DMA_SPECIALIZATION:([[:alnum:]]|\$|_)+]]
func.func @test(%a : memref<32xf32>, %b : memref<32xf32>) {
  %view = quidditch_snitch.l1_memory_view -> memref<512xi8>
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  // CHECK: memref.view
  // CHECK-NEXT: memref.view
  %a_l1 = memref.view %view[%c0][] : memref<512xi8> to memref<32xf32>
  %b_l1 = memref.view %view[%c256][] : memref<512xi8> to memref<32xf32>

  // CHECK-NEXT: quidditch_snitch.microkernel_fence
  // CHECK-NEXT: quidditch_snitch.barrier
  // CHECK-NEXT: quidditch_snitch.microkernel_fence
  // CHECK-NEXT: quidditch_snitch.barrier
  // CHECK-NEXT: quidditch_snitch.barrier
  quidditch_snitch.start_dma_transfer from %a : memref<32xf32> to %a_l1 : memref<32xf32>
  %t = quidditch_snitch.start_dma_transfer from %b : memref<32xf32> to %b_l1 : memref<32xf32>
  quidditch_snitch.wait_for_dma_transfers %t : !quidditch_snitch.dma_token

  // CHECK-NEXT: microkernel
  // CHECK: }
  quidditch_snitch.memref.microkernel(%a_l1, %b_l1) : memref<32xf32>, memref<32xf32> {
  ^bb0(%arg0 : memref<32xf32>, %arg1 : memref<32xf32>):
    linalg.abs ins(%arg0 : memref<32xf32>) outs(%arg1 : memref<32xf32>)
  }

  // CHECK-NEXT: quidditch_snitch.microkernel_fence
  // CHECK-NEXT: quidditch_snitch.barrier
  // CHECK-NEXT: quidditch_snitch.barrier
  // CHECK-NEXT: return
  %t2 = quidditch_snitch.start_dma_transfer from %b_l1 : memref<32xf32> to %b : memref<32xf32>
  quidditch_snitch.wait_for_dma_transfers %t2 : !quidditch_snitch.dma_token
  return
}

// CHECK: @[[DMA_SPECIALIZATION]](
// CHECK-SAME: %[[A:[[:alnum:]]+]]: memref<32xf32>
// CHECK-SAME: %[[B:[[:alnum:]]+]]: memref<32xf32>

// CHECK: memref.view
// CHECK-NEXT: memref.view

// CHECK-NEXT: quidditch_snitch.barrier
// CHECK-NEXT: quidditch_snitch.start_dma_transfer
// CHECK-NEXT: quidditch_snitch.barrier
// CHECK-NEXT: quidditch_snitch.start_dma_transfer
// CHECK-NEXT: quidditch_snitch.wait_for_dma_transfers
// CHECK-NEXT: quidditch_snitch.barrier

// CHECK-NEXT: quidditch_snitch.barrier
// CHECK-NEXT: quidditch_snitch.start_dma_transfer
// CHECK-NEXT: quidditch_snitch.wait_for_dma_transfers
// CHECK-NEXT: quidditch_snitch.barrier

// CHECK-NEXT: return
