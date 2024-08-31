
#pragma once

namespace mlir {
class DialectRegistry;
}

namespace quidditch::dma {
void registerDMACoreSpecializationOpInterface(mlir::DialectRegistry &registry);
}
