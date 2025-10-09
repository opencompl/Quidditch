builtin.module @test_tiny_matmul {
  func.func @tiny_matmul(%lhs: tensor<3x3xf64>, %rhs: tensor<3x3xf64>, %acc: tensor<3x3xf64>) -> tensor<3x3xf64> {
  %result = linalg.matmul // changing linalg.add to linalg.matmul, causes a RESOURCE_EXHAUSTED error
    ins(%lhs, %rhs: tensor<3x3xf64>, tensor<3x3xf64>)
    outs(%acc: tensor<3x3xf64>)
  -> tensor<3x3xf64>
  return %result: tensor<3x3xf64>
  }
}
