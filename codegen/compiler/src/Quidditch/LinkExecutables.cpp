#include "Passes.h"

#include "iree/compiler/Codegen/Utils/LinkingUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace quidditch {
#define GEN_PASS_DEF_LINKEXECUTABLESPASS
#include "Quidditch/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {
class LinkExecutables
    : public quidditch::impl::LinkExecutablesPassBase<LinkExecutables> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

void LinkExecutables::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());

  auto sourceExecutableOps =
      llvm::to_vector<8>(moduleOp.getOps<IREE::HAL::ExecutableOp>());
  if (sourceExecutableOps.size() <= 1)
    return;

  // TODO: This is a huge hack to workaround the linking process wanting a
  //  unified condition for the resulting `hal.executable.variant`. We instead
  //  want a condition per-kernel, aka per `hal.executable.variant` prior to
  //  linking. We currently achieve this by using different
  //  `hal.executable.condition` ops which are processed prior to linking in the
  //  stream-to-hal conversion. Remove the condition now to satisfy the linker
  //  requirement.
  for (IREE::HAL::ExecutableOp executable : sourceExecutableOps)
    for (auto variant : executable.getOps<IREE::HAL::ExecutableVariantOp>())
      if (IREE::HAL::ExecutableConditionOp condition = variant.getConditionOp())
        condition.erase();

  // Guess a module name, if needed, to make the output files readable.
  std::string moduleName = guessModuleName(moduleOp, "quidditch_module");

  // Create our new "linked" hal.executable.
  std::string linkedExecutableName =
      llvm::formatv("{0}_linked_{1}", moduleName, "quidditch");
  auto linkedExecutableOp = moduleBuilder.create<IREE::HAL::ExecutableOp>(
      moduleOp.getLoc(), linkedExecutableName);
  linkedExecutableOp.setVisibility(sourceExecutableOps.front().getVisibility());
  auto executableBuilder =
      OpBuilder::atBlockBegin(&linkedExecutableOp.getBlock());

  IREE::HAL::ExecutableVariantOp llvmVariant;
  IREE::HAL::ExecutableVariantOp quidditchVariant;

  // Gather all unique executable targets - we may have multiple.
  SetVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs =
      gatherExecutableTargets(sourceExecutableOps);
  for (auto [index, attr] : llvm::enumerate(executableTargetAttrs)) {
    // Add our hal.executable.variant with an empty module.
    std::string linkedVariantName =
        executableTargetAttrs.size() == 1
            ? attr.getSymbolNameFragment()
            : llvm::formatv("{0}_{1}", attr.getSymbolNameFragment(), index);
    auto linkedTargetOp =
        executableBuilder.create<IREE::HAL::ExecutableVariantOp>(
            moduleOp.getLoc(), linkedVariantName, attr);
    auto targetBuilder = OpBuilder::atBlockBegin(&linkedTargetOp.getBlock());
    targetBuilder.create<mlir::ModuleOp>(moduleOp.getLoc());

    if (attr.getBackend() == "llvm-cpu")
      llvmVariant = linkedTargetOp;
    else if (attr.getBackend() == "quidditch")
      quidditchVariant = linkedTargetOp;

    auto mergeModuleFn = [](mlir::ModuleOp sourceInnerModule,
                            mlir::ModuleOp linkedInnerModule,
                            DenseMap<StringRef, Operation *> &symbolMap) {
      return mergeModuleInto(sourceInnerModule, linkedInnerModule, symbolMap);
    };

    // Try linking together all executables in moduleOp.
    if (failed(linkExecutablesInto(moduleOp, sourceExecutableOps,
                                   linkedExecutableOp, linkedTargetOp,
                                   mergeModuleFn))) {
      return signalPassFailure();
    }
  }

  if (!quidditchVariant || !llvmVariant)
    return;

  std::size_t xDSLFunctionsReplaced = 0;
  std::size_t xDSLFunctionsEncountered = 0;
  // Replace quidditch functions that xDSL could not compile with LLVM
  // implementations.
  auto builder =
      OpBuilder::atBlockEnd(quidditchVariant.getInnerModule().getBody());
  SymbolTable quidditchTable(quidditchVariant.getInnerModule());
  SymbolTable llvmTable(llvmVariant.getInnerModule());
  for (auto func : llvm::to_vector(
           quidditchVariant.getInnerModule().getOps<LLVM::LLVMFuncOp>())) {
    StringRef symName = func.getSymName();
    if (!symName.consume_back("$iree_to_xdsl"))
      continue;

    xDSLFunctionsEncountered++;

    auto attr = func->getAttrOfType<ArrayAttr>("xdsl_kernels");
    if (!attr)
      continue;

    SmallVector<Operation *> toErase;
    bool failed = false;
    for (FlatSymbolRefAttr ref : attr.getAsRange<FlatSymbolRefAttr>()) {
      // xDSL kernel is considered having failed to compile if it does not
      // exist or does not have a body yet no 'riscv_assembly' either.
      auto xdslFuncOp = quidditchTable.lookup<LLVM::LLVMFuncOp>(ref.getAttr());
      if (xdslFuncOp && xdslFuncOp->hasAttrOfType<StringAttr>("riscv_assembly"))
        continue;

      if (xdslFuncOp && !xdslFuncOp.empty())
        continue;

      failed = true;
      if (xdslFuncOp)
        toErase.push_back(xdslFuncOp);
    }
    if (!failed)
      continue;

    auto llvmFuncOp = llvmTable.lookup<LLVM::LLVMFuncOp>(symName);
    if (!llvmFuncOp)
      continue;

    for (Operation *op : toErase)
      op->erase();

    xDSLFunctionsReplaced++;

    func.erase();
    LLVM::LLVMFuncOp clone = llvmFuncOp.clone();
    builder.insert(clone);
  }

  if (xDSLFunctionsReplaced)
    emitWarning(moduleOp.getLoc(), "Replaced ")
        << xDSLFunctionsReplaced << " out of " << xDSLFunctionsEncountered
        << " kernels with LLVM implementations as they failed to compile";
}
