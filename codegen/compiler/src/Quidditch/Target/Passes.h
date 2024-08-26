
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace quidditch {

enum class TilingLevel {
  /// Performs tiling within a workgroup to fit all tensors required for the
  /// root operation into L1.
  L1,
  /// Performs tiling and distribution of compute operations to all compute
  /// cores within a workgroup.
  Thread
};

#define GEN_PASS_DECL
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch
