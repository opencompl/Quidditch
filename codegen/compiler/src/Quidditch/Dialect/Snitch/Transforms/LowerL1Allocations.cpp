#include "Passes.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace quidditch::Snitch {
#define GEN_PASS_DEF_LOWERL1ALLOCATIONSPASS
#include "Quidditch/Dialect/Snitch/Transforms/Passes.h.inc"
} // namespace quidditch::Snitch

namespace {
class LowerL1Allocations
    : public quidditch::Snitch::impl::LowerL1AllocationsPassBase<
          LowerL1Allocations> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;
using namespace quidditch::Snitch;

void LowerL1Allocations::runOnOperation() {
  SmallVector<memref::AllocaOp> allocs;
  getOperation()->walk([&](memref::AllocaOp allocOp) {
    if (!isa_and_nonnull<L1EncodingAttr>(allocOp.getType().getMemorySpace()))
      return;
    if (!allocOp.getDynamicSizes().empty()) {
      // Note: There is no reason for this being unsupported other than we don't
      // need it right now and its extra work.
      allocOp->emitOpError(
          "L1 allocations with dynamic size is currently unsupported");
      signalPassFailure();
      return;
    }

    allocs.push_back(allocOp);
  });
  if (allocs.empty())
    return;

  auto builder = OpBuilder::atBlockBegin(&getOperation().front());
  auto l1Memory = builder.create<L1MemoryViewOp>(
      getOperation().getLoc(),
      MemRefType::get({l1MemoryBytes}, builder.getI8Type()));
  uint64_t offset = 0;
  for (memref::AllocaOp allocOp : allocs) {
    builder.setInsertionPoint(allocOp);
    MemRefType memRefType = allocOp.getType();
    // Note: This assumes bitWidth == alignment == size.
    uint64_t bitWidth = memRefType.getElementTypeBitWidth();
    if (std::optional<uint64_t> alignment = allocOp.getAlignment())
      offset = llvm::alignTo(offset, *alignment);
    else
      offset = llvm::alignTo(offset, llvm::divideCeil(bitWidth, 8));

    auto byteShift =
        builder.create<arith::ConstantIndexOp>(allocOp.getLoc(), offset);
    // Get rid of the memory space at this point in the pipeline.
    auto viewOp = builder.create<memref::ViewOp>(
        allocOp.getLoc(),
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        memRefType.getLayout()),
        l1Memory, byteShift,
        /*sizes=*/ValueRange());
    allocOp->replaceAllUsesWith(viewOp);

    uint64_t memRefSize = llvm::divideCeil(bitWidth, 8);
    for (uint64_t size : memRefType.getShape())
      memRefSize *= size;

    offset += memRefSize;
    if (offset >= l1MemoryBytes) {
      auto diagEmit =
          assertCompiled ? &Operation::emitError : &Operation::emitWarning;
      ((*getOperation()).*
       diagEmit)("kernel does not fit into L1 memory and cannot be compiled");
      if (assertCompiled) {
        signalPassFailure();
        return;
      }

      auto *dialect = getContext().getLoadedDialect<QuidditchSnitchDialect>();
      dialect->getXdslCompilationFailedAttrHelper().setAttr(
          getOperation(), UnitAttr::get(&getContext()));

      // The function is in an invalid state now that we cannot lower. Work
      // around this by erasing the body completely.
      getOperation().setPrivate();
      getOperation().getBlocks().clear();
      return;
    }
  }

  // Change any leftover memory space occurrences.
  AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](MemRefType memRefType) -> std::optional<MemRefType> {
        if (!memRefType.getMemorySpace())
          return std::nullopt;

        return MemRefType::get(memRefType.getShape(),
                               memRefType.getElementType(),
                               memRefType.getLayout());
      });
  replacer.recursivelyReplaceElementsIn(getOperation(), /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);
}
