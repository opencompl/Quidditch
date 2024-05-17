#include "Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace quidditch {
#define GEN_PASS_DEF_FILTERFORXDSLPASS
#include "Quidditch/Passes.h.inc"
} // namespace quidditch

using namespace mlir;

namespace {
class FilterForxDSL
    : public quidditch::impl::FilterForxDSLPassBase<FilterForxDSL> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

void FilterForxDSL::runOnOperation() {
  getOperation()->walk([&](Operation* op) {
    if (isa<memref::AssumeAlignmentOp>(op))
      op->erase();
  });
}
