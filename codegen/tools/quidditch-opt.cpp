#include <iree/compiler/Tools/init_dialects.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include <Quidditch/Dialect/Snitch/QuidditchSnitchDialect.h>
#include <Quidditch/Target/Passes.h>

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace quidditch {
#define GEN_PASS_REGISTRATION
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;

int main(int argc, char **argv) {

  // Be lazy and support all upstream dialects as input dialects.
  DialectRegistry registry;
  iree_compiler::registerAllDialects(registry);
  registry.insert<quidditch::Snitch::QuidditchSnitchDialect>();

  quidditch::registerPasses();
  mlir::bufferization::registerBufferizationPasses();
  mlir::registerTransformsPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}
