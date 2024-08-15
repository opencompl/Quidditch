#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
    if (auto index = dyn_cast<ComputeCoreIndexOp>(operation)) {
      OpBuilder builder(operation);
      // Make the DMA core follow the control flow of the first compute core.
      // This whole pass runs under the assumption that any operation that is
      // run on either the DMA core or compute cores are in non-divergent
      // control flow. Making the DMA core follow any compute cores control
      // flow is therefore safe to do.
      // This is mainly required for barriers within a `scf.forall`.
      operation->replaceAllUsesWith(
          builder.create<arith::ConstantIndexOp>(operation->getLoc(), 0));
      operation->erase();
    }
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
