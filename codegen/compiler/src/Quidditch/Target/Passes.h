
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace quidditch {

enum class TilingLevel {
  Reduction,
  Thread
};

#define GEN_PASS_DECL
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch
