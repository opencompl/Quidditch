#include <memory>
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

namespace quidditch {
#define GEN_PASS_DEF_AVOIDBANKCONFLICTSPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace quidditch;

namespace {

class AvoidBankConflicts final
    : public quidditch::impl::AvoidBankConflictsPassBase<AvoidBankConflicts> {
  void runOnOperation() override;
};
} // namespace

static FailureOr<memref::AllocaOp>
findBaseAllocation(TypedValue<MemRefType> memRef) {
//  SmallVector<TypedValue<MemRefType>> workList = {memRef};
//  while (!workList.empty()) {
//    TypedValue<MemRefType> current = workList.pop_back_val();
//    if (auto alloca = dyn_cast<memref::AllocaOp>(current))
//      return alloca;
//
//    auto next =
//        TypeSwitch<Operation *, TypedValue<MemRefType>>(current.getDefiningOp())
//            .Case([](OffsetSizeAndStrideOpInterface
//                         offsetSizeAndStrideOpInterface) {
//
//            })
//            .Default({});
//  }
  return failure();
}

void AvoidBankConflicts::runOnOperation() {
  getOperation()->walk([&](TilingInterface computeOp) {
    SmallVector<FailureOr<memref::AllocaOp>> baseAllocas;
    for (Value operand : computeOp->getOperands())
      if (auto memRefValue = dyn_cast<TypedValue<MemRefType>>(operand)) {
        if (isa_and_nonnull<Snitch::L1EncodingAttr>(
                memRefValue.getType().getMemorySpace()))
          baseAllocas.push_back(findBaseAllocation(memRefValue));
        else
          baseAllocas.push_back(failure());
      }
  });
}
