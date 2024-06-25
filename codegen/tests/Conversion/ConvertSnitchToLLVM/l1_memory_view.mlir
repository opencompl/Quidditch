// RUN: quidditch-opt %s --quidditch-convert-snitch-to-llvm | FileCheck %s

// CHECK-LABEL: @test
func.func @test() -> memref<32xi8> {
  // CHECK-DAG: %[[DIM:.*]] = llvm.mlir.constant(32 : index)
  // CHECK-DAG: %[[STRIDE:.*]] = llvm.mlir.constant(1 : index)
  // CHECK: %[[L1_ADDRESS:.*]] = llvm.mlir.constant({{[0-9]+}} : i32)
  // CHECK: %[[PTR:.*]] = llvm.inttoptr %[[L1_ADDRESS]]
  // CHECK: %[[UNDEF:.*]] = llvm.mlir.undef
  // CHECK: %[[DESC1:.*]] = llvm.insertvalue %[[PTR]], %[[UNDEF]][0]
  // CHECK: %[[DESC2:.*]] = llvm.insertvalue %[[PTR]], %[[DESC1]][1]
  // CHECK: %[[OFFSET:.*]] = llvm.mlir.constant(0 : index)
  // CHECK: %[[DESC3:.*]] = llvm.insertvalue %[[OFFSET]], %[[DESC2]][2]
  // CHECK: %[[DESC4:.*]] = llvm.insertvalue %[[DIM]], %[[DESC3]][3, 0]
  // CHECK: %[[DESC5:.*]] = llvm.insertvalue %[[STRIDE]], %[[DESC4]][4, 0]
  // CHECK: builtin.unrealized_conversion_cast %[[DESC5]]
  %0 = quidditch_snitch.l1_memory_view -> memref<32xi8>
  return %0 : memref<32xi8>
}
