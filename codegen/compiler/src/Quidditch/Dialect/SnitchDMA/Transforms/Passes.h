
#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace quidditch::SnitchDMA {
#define GEN_PASS_DECL
#include "Quidditch/Dialect/SnitchDMA/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch
