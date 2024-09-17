#include "SnitchDMATypes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "SnitchDMADialect.h"

#define GET_TYPEDEF_CLASSES
#include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMATypes.cpp.inc"
