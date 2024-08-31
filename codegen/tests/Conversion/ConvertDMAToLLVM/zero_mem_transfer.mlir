// RUN: quidditch-opt %s --quidditch-convert-to-llvm | FileCheck %s

// CHECK-LABEL: @test(
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[DIM0:[[:alnum:]]+]]
func.func private @test(%arg0 : memref<?xf32>) -> !dma.token {
  // CHECK-DAG: %[[NULL:.*]] = llvm.mlir.zero
  // CHECK-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 :
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[NULL]][%[[DIM0]]]
  // CHECK: %[[SIZE:.*]] = llvm.ptrtoint %[[GEP]]
  // CHECK: %[[LOOP:.*]] = llvm.udiv %[[SIZE]], %[[ZERO_MEM_SIZE:[[:alnum:]]+]]
  // CHECK: call @snrt_dma_start_2d(%[[PTR]], %[[ZERO_MEM:.*]], %[[ZERO_MEM_SIZE]], %[[ZERO_MEM_SIZE]], %[[ZERO]], %[[LOOP]])
  // CHECK: %[[OFFSET:.*]] = llvm.mul %[[LOOP]], %[[ZERO_MEM_SIZE]]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[PTR]][%[[OFFSET]]]
  // CHECK: %[[REM:.*]] = llvm.urem %[[SIZE]], %[[ZERO_MEM_SIZE]]
  // CHECK: %[[TOKEN:.*]] = llvm.call @snrt_dma_start_1d(%[[GEP]], %[[ZERO_MEM]], %[[REM]])
  %0 = dma.start_zero_mem_transfer %arg0 : memref<?xf32>
  // CHECK: return %[[TOKEN]]
  return %0 : !dma.token
}

// CHECK-LABEL: @test1(
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[PTR:[[:alnum:]]+]]
// CHECK-SAME: %{{[[:alnum:]]+}}
// CHECK-SAME: %[[DIM0:[[:alnum:]]+]]
// CHECK-SAME: %[[DIM1:[[:alnum:]]+]]
// CHECK-SAME: %[[DIM2:[[:alnum:]]+]]
// CHECK-SAME: %[[STRIDE0:[[:alnum:]]+]]
// CHECK-SAME: %[[STRIDE1:[[:alnum:]]+]]
func.func private @test1(%arg0 : memref<?x?x?xf32, strided<[?, ?, 1]>>) -> !dma.token {
  // CHECK-DAG: %[[ZERO_INDEX:.*]] = llvm.mlir.constant(0 : index)
  // CHECK-DAG: %[[ZERO_I32:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-DAG: %[[ONE:.*]] = llvm.mlir.constant(1 :
  // CHECK-DAG: %[[NULL:.*]] = llvm.mlir.zero
  // CHECK: llvm.br ^[[LOOP0:.*]](%[[ZERO_INDEX]], %[[ZERO_I32]] :

  // CHECK: ^[[LOOP0]](%[[IV0:.*]]: {{.*}}, %[[TOKEN0:.*]]: {{.*}}):
  // CHECK: %[[CMP0:.*]] = llvm.icmp "slt" %[[IV0]], %[[DIM0]]
  // CHECK: llvm.cond_br %[[CMP0]], ^[[BODY0:.*]], ^[[EXIT0:[[:alnum:]]+]]

  // CHECK: ^[[BODY0]]:
  // CHECK: llvm.br ^[[LOOP1:.*]](%[[ZERO_INDEX]], %[[TOKEN0]] :

  // CHECK: ^[[LOOP1]](%[[IV1:.*]]: {{.*}}, %[[TOKEN1:.*]]: {{.*}}):
  // CHECK: %[[CMP1:.*]] = llvm.icmp "slt" %[[IV1]], %[[DIM1]]
  // CHECK: llvm.cond_br %[[CMP1]], ^[[BODY1:.*]], ^[[EXIT1:[[:alnum:]]+]]

  // CHECK: ^[[BODY1]]:
  // memref.subview lowering:
  // CHECK: %[[MUL0:.*]] = llvm.mul %[[IV0]], %[[STRIDE0]]
  // CHECK: %[[MUL1:.*]] = llvm.mul %[[IV1]], %[[STRIDE1]]
  // CHECK: %[[OFFSET:.*]] = llvm.add %[[MUL0]], %[[MUL1]]

  // Check the subview is used as destination. The lowering here is already tested above.
  // CHECK: %[[GEP:.*]] = llvm.getelementptr %[[PTR]][%[[OFFSET]]]
  // CHECK: call @snrt_dma_start_2d(%[[GEP]],

  // CHECK: %[[NEXT_TOKEN:.*]] = llvm.call @snrt_dma_start_1d(
  // CHECK: %[[INC1:.*]] = llvm.add %[[IV1]], %[[ONE]]
  // CHECK: llvm.br ^[[LOOP1]](%[[INC1]], %[[NEXT_TOKEN]] :

  // CHECK: ^[[EXIT1]]:
  // CHECK: %[[INC0:.*]] = llvm.add %[[IV0]], %[[ONE]]
  // CHECK: llvm.br ^[[LOOP0]](%[[INC0]], %[[TOKEN1]] :

  // CHECK: ^[[EXIT0]]:
  %0 = dma.start_zero_mem_transfer %arg0 : memref<?x?x?xf32, strided<[?, ?, 1]>>
  // CHECK: return %[[TOKEN0]]
  return %0 : !dma.token
}
