#include "Passes.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace quidditch {
#define GEN_PASS_DEF_DISABLEQUIDDITCHVARIANTPASS
#include "Quidditch/Target/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {
class DisableQuidditchVariant
    : public quidditch::impl::DisableQuidditchVariantPassBase<
          DisableQuidditchVariant> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

void DisableQuidditchVariant::runOnOperation() {
  // The following code makes the assumption that it is run before linkage,
  // meaning that one hal.executable.variant contains exactly one kernel.
  IREE::HAL::ExecutableVariantOp operation = getOperation();
  ModuleOp module = operation.getInnerModule();

  // If xDSL failed to compile the kernel, then disable this variant
  // permanently. While the code will later be replaced by the linker using
  // LLVM code, the dispatch code needs to also be informed to make sure that
  // the right workgroup sizes are used for the kernel.
  for (auto func : module.getOps<LLVM::LLVMFuncOp>())
    if (func->hasAttr("xdsl_generated") && func->hasAttr("riscv_assembly"))
      return;

  OpBuilder builder(&getContext());
  operation.createConditionOp(builder);
  Value falseC = builder.create<arith::ConstantOp>(operation->getLoc(),
                                                   builder.getBoolAttr(false));
  builder.create<IREE::HAL::ReturnOp>(operation->getLoc(), falseC);
}
