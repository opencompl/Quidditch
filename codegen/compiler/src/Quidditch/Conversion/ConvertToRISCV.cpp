#include "Passes.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"

namespace quidditch {
#define GEN_PASS_DEF_CONVERTTORISCVPASS
#include "Quidditch/Conversion/Passes.h.inc"
} // namespace quidditch

using namespace mlir;
using namespace quidditch::Snitch;

namespace {
class ConvertToRISCV
    : public quidditch::impl::ConvertToRISCVPassBase<ConvertToRISCV> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;

private:
  FailureOr<StringAttr> convertToRISCVAssembly(MemRefMicrokernelOp kernelOp,
                                               StringAttr kernelName);
};
} // namespace

static bool canUseBarepointerCC(Type type) {
  auto memRef = dyn_cast<MemRefType>(type);
  if (!memRef)
    return true;
  if (isa<UnrankedMemRefType>(memRef))
    return false;

  int64_t offset = 0;
  SmallVector<int64_t, 4> strides;
  if (failed(getStridesAndOffset(memRef, strides, offset)))
    return false;

  for (int64_t stride : strides)
    if (ShapedType::isDynamic(stride))
      return false;

  return !ShapedType::isDynamic(offset);
}

FailureOr<StringAttr>
ConvertToRISCV::convertToRISCVAssembly(MemRefMicrokernelOp kernelOp,
                                       StringAttr kernelName) {
  if (!llvm::all_of(kernelOp.getBody().getArgumentTypes(),
                    canUseBarepointerCC)) {
    auto emit = assertCompiled ? &MemRefMicrokernelOp::emitError
                               : &MemRefMicrokernelOp::emitWarning;

    (kernelOp.*emit)("function inputs ")
        << kernelOp.getBody().getArgumentTypes()
        << " do not support bare-pointer calling convention required by "
           "xDSL.";
    return failure();
  }

  OpBuilder builder(&getContext());
  OwningOpRef<func::FuncOp> tempFuncOp = builder.create<func::FuncOp>(
      kernelOp.getLoc(), kernelName,
      builder.getFunctionType(kernelOp.getBody().getArgumentTypes(),
                              kernelOp.getResultTypes()));
  IRMapping mapping;
  kernelOp.getBody().cloneInto(&tempFuncOp->getBody(), mapping);
  builder.setInsertionPointToEnd(&tempFuncOp->getBody().back());
  tempFuncOp->getBody().back().getTerminator()->erase();

  SmallVector<Value> returns;
  for (Value value : kernelOp.getYieldOp().getResults())
    returns.push_back(mapping.lookupOrDefault(value));
  builder.create<func::ReturnOp>(kernelOp.getLoc(), returns);

  SmallString<64> stdinFile;
  int stdinFd;
  if (llvm::sys::fs::createTemporaryFile("xdsl-in", "mlir", stdinFd, stdinFile))
    return failure();

  llvm::FileRemover stdinFileRemove(stdinFile);
  {
    llvm::raw_fd_ostream ss(stdinFd, /*shouldClose=*/true);
    tempFuncOp->print(ss, OpPrintingFlags().useLocalScope());
  }

  SmallString<64> stdoutFile;
  if (llvm::sys::fs::createTemporaryFile("xdsl-out", "S", stdoutFile))
    return failure();

  llvm::FileRemover stdoutFileRemove(stdoutFile);

  SmallString<64> stderrFile;
  if (llvm::sys::fs::createTemporaryFile("xdsl-diag", "S", stderrFile))
    return failure();

  llvm::FileRemover stderrFileRemove(stderrFile);

  std::optional<llvm::StringRef> redirects[3] = {/*stdin=*/stdinFile.str(),
                                                 /*stdout=*/stdoutFile.str(),
                                                 /*stderr=*/stderrFile.str()};
  int ret = llvm::sys::ExecuteAndWait(
      xDSLOptPath,
      {xDSLOptPath, "-p",
       "convert-linalg-to-memref-stream,"
       "test-optimise-memref-stream," // NOLINT(*-suspicious-missing-comma)
       "test-lower-memref-stream-to-snitch-stream,"
       "test-lower-snitch-stream-to-asm",
       "-t", "riscv-asm"},
      std::nullopt, redirects);
  if (ret != 0) {
    auto diagEmit =
        assertCompiled ? &Operation::emitError : &Operation::emitWarning;

    InFlightDiagnostic diag =
        ((kernelOp)->*diagEmit)("Failed to translate kernel with xDSL");

    if (llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
            llvm::MemoryBuffer::getFile(stderrFile, /*IsText=*/true))
      diag.attachNote() << "stderr:\n" << buffer.get()->getBuffer();

    return diag;
  }

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(stdoutFile, /*IsText=*/true);
  if (!buffer)
    return kernelOp.emitError("failed to open ") << stdoutFile;

  return StringAttr::get(&getContext(), (*buffer)->getBuffer());
}

void ConvertToRISCV::runOnOperation() {
  ModuleOp module = getOperation();
  SymbolTable symbolTable(module);
  auto *dialect = getContext().getLoadedDialect<QuidditchSnitchDialect>();

  std::size_t kernelIndex = 0;
  module.walk([&](MemRefMicrokernelOp kernelOp) {
    auto parentFuncOp = kernelOp->getParentOfType<func::FuncOp>();
    auto kernelName = StringAttr::get(
        &getContext(), llvm::formatv("{0}$xdsl_kernel{1}",
                                     parentFuncOp.getSymName(), kernelIndex++)
                           .str());

    FailureOr<StringAttr> riscvAssembly =
        convertToRISCVAssembly(kernelOp, kernelName);
    if (failed(riscvAssembly)) {
      if (assertCompiled) {
        signalPassFailure();
        return WalkResult::interrupt();
      }

      auto containedFunc = kernelOp->getParentOfType<func::FuncOp>();
      dialect->getXdslCompilationFailedAttrHelper().setAttr(
          containedFunc, UnitAttr::get(&getContext()));
      // The function is in an invalid state now that we cannot lower. Work
      // around this by erasing the body completely.
      containedFunc.setPrivate();
      containedFunc.getBody().getBlocks().clear();
      return WalkResult::interrupt();
    }

    auto builder = OpBuilder::atBlockEnd(module.getBody());

    auto kernelDecl = builder.create<func::FuncOp>(
        kernelOp.getLoc(), kernelName,
        builder.getFunctionType(kernelOp.getBody().getArgumentTypes(),
                                kernelOp.getResultTypes()));

    kernelDecl.setVisibility(SymbolTable::Visibility::Private);
    // Required to tell the conversion pass to LLVM that this is actually a
    // call into the same linkage unit, and does not have to be rewritten to a
    // HAL module call.
    kernelDecl->setAttr("hal.import.bitcode", UnitAttr::get(&getContext()));
    kernelDecl->setAttr("llvm.bareptr", UnitAttr::get(&getContext()));

    dialect->getRiscvAssemblyAttrHelper().setAttr(kernelDecl, *riscvAssembly);

    builder.setInsertionPoint(kernelOp);
    auto callOp = builder.create<func::CallOp>(kernelOp.getLoc(), kernelDecl,
                                               kernelOp.getInputs());
    kernelOp.replaceAllUsesWith(callOp);
    kernelOp.erase();
    return WalkResult::advance();
  });
}
