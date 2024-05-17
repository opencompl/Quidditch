
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace quidditch {
#define GEN_PASS_DECL
#include "Quidditch/Passes.h.inc"
} // namespace quidditch
