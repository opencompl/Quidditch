
#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "SnitchDMATypes.h"

#define GET_OP_CLASSES
#include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMAOps.h.inc"

namespace quidditch::SnitchDMA {
class QueueResource : public mlir::SideEffects::Resource::Base<QueueResource> {
public:
  llvm::StringRef getName() override;
};
} // namespace quidditch::SnitchDMA
