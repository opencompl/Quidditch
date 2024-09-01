// RUN: quidditch-opt %s --verify-roundtrip

func.func @test(%arg0 : tensor<?x4xf64>) -> (tensor<?x4xf64>, !dma.token) {
  %0:2 = dma.start_tensor_copy of %arg0 to #quidditch_snitch.l1_encoding -> tensor<?x4xf64>
  return %0#0, %0#1 : tensor<?x4xf64>, !dma.token
}
