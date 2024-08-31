#include <iree/compiler/Tools/init_dialects.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "Quidditch/Conversion/Passes.h"
#include "Quidditch/Dialect/DMA/Extensions/DMACoreSpecializationOpInterfaceImpl.h"
#include "Quidditch/Dialect/DMA/IR/DMADialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h"
#include "Quidditch/Target/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace quidditch {
#define GEN_PASS_REGISTRATION
#include "Quidditch/Target/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "Quidditch/Conversion/Passes.h.inc"
namespace Snitch {
#define GEN_PASS_REGISTRATION
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace Snitch
} // namespace quidditch

using namespace mlir;

int main(int argc, char **argv) {

  // Be lazy and support all upstream dialects as input dialects.
  DialectRegistry registry;
  quidditch::dma::registerDMACoreSpecializationOpInterface(registry);
  iree_compiler::registerAllDialects(registry);
  registry.insert<quidditch::Snitch::QuidditchSnitchDialect,
                  quidditch::dma::DMADialect>();

  quidditch::registerPasses();
  quidditch::registerConversionPasses();
  quidditch::Snitch::registerTransformsPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerTransformsPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}
