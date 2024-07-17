#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_SPECIALIZEDMACODEPASS
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch

namespace {
class SpecializeDMACode
    : public quidditch::Snitch::impl::SpecializeDMACodePassBase<
          SpecializeDMACode> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;

private:
};

} // namespace

using namespace mlir;
using namespace quidditch::Snitch;

static void removeComputeOps(FunctionOpInterface dmaCode) {
  dmaCode->walk([&](Operation *operation) {
    if (isa<MemRefMicrokernelOp, MicrokernelFenceOp>(operation))
      operation->erase();
  });
}

static void removeDmaCode(FunctionOpInterface computeCode) {
  SmallVector<Operation *> toDelete;
  computeCode->walk([&](Operation *operation) {
    if (isa<WaitForDMATransfersOp>(operation))
      operation->erase();
    if (isa<StartDMATransferOp>(operation)) {
      OpBuilder builder(operation);
      operation->replaceAllUsesWith(
          builder.create<CompletedTokenOp>(operation->getLoc()));
      operation->erase();
    }
  });
}

static void insertBarriers(FunctionOpInterface function) {
  function->walk([](Operation *operation) {
    OpBuilder builder(operation->getContext());
    if (isa<WaitForDMATransfersOp>(operation)) {
      // Barrier needs to be after the wait to signal to compute ops the
      // transfer is done.
      builder.setInsertionPointAfter(operation);
    } else if (isa<StartDMATransferOp>(operation)) {
      // Barrier needs to be before the transfer for compute ops to signal
      // that a computation is done.
      // TODO: This is overly conservative and could be optimized somewhere.
      builder.setInsertionPoint(operation);
      builder.create<MicrokernelFenceOp>(operation->getLoc());
    } else
      return;

    builder.create<BarrierOp>(operation->getLoc());
  });
}

void SpecializeDMACode::runOnOperation() {
  auto *dialect = getContext().getLoadedDialect<QuidditchSnitchDialect>();
  SymbolTable table(getOperation());
  auto toSpecialize =
      llvm::to_vector(getOperation().getOps<FunctionOpInterface>());
  for (FunctionOpInterface function : toSpecialize) {
    if (function.isDeclaration())
      continue;

    insertBarriers(function);

    FunctionOpInterface clone = function.clone();
    clone.setName((clone.getName() + "$dma").str());
    table.insert(clone, std::next(function->getIterator()));
    dialect->getDmaSpecializationAttrHelper().setAttr(
        function, FlatSymbolRefAttr::get(clone));

    removeComputeOps(clone);
    removeDmaCode(function);
  }
}
