#include "Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_FORMMICROKERNELSPASS
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch

using namespace mlir;
using namespace quidditch::Snitch;

namespace {
class FormMicrokernels
    : public quidditch::Snitch::impl::FormMicrokernelsPassBase<
          FormMicrokernels> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

void FormMicrokernels::runOnOperation() {
  FunctionOpInterface func = getOperation();

  func.walk([](linalg::LinalgOp linalgOp) {
    if (!linalgOp.hasPureTensorSemantics())
      return;

    auto builder = OpBuilder(linalgOp);
    auto kernelOp = builder.create<TensorMicrokernelOp>(
        linalgOp.getLoc(), linalgOp->getResultTypes());
    for (auto [oldResult, newResult] :
         llvm::zip_equal(linalgOp->getResults(), kernelOp.getResults())) {
      oldResult.replaceAllUsesWith(
          builder.create<SyncTensorOp>(linalgOp.getLoc(), newResult));
    }

    Block *block = &kernelOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(block);

    linalgOp->remove();
    builder.insert(linalgOp);
    builder.create<MicrokernelYieldOp>(linalgOp->getLoc(),
                                       linalgOp->getResults());
  });
}
