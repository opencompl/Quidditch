#include "Passes.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace quidditch {
#define GEN_PASS_DEF_CONVERTTORISCVPASS
#include "Quidditch/Passes.h.inc"
} // namespace quidditch

using namespace mlir;

namespace {
class ConvertToRISCV
    : public quidditch::impl::ConvertToRISCVPassBase<ConvertToRISCV> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

void ConvertToRISCV::runOnOperation() {
  func::FuncOp func = getOperation();

  if (!func->hasAttr("xdsl_generated"))
    return;

  SmallString<64> stdinFile;
  int stdinFd;
  if (llvm::sys::fs::createTemporaryFile("xdsl-in", "mlir", stdinFd,
                                         stdinFile)) {
    signalPassFailure();
    return;
  }
  llvm::FileRemover stdinFileRemove(stdinFile);
  {
    llvm::raw_fd_ostream ss(stdinFd, /*shouldClose=*/true);
    func.print(ss, OpPrintingFlags().printGenericOpForm().useLocalScope());
  }

  SmallString<64> stdoutFile;
  if (llvm::sys::fs::createTemporaryFile("xdsl-out", "S", stdoutFile)) {
    signalPassFailure();
    return;
  }
  llvm::FileRemover stdoutFileRemove(stdoutFile);

  SmallString<64> stderrFile;
  if (llvm::sys::fs::createTemporaryFile("xdsl-diag", "S", stderrFile)) {
    signalPassFailure();
    return;
  }
  llvm::FileRemover stderrFileRemove(stderrFile);

  std::optional<llvm::StringRef> redirects[3] = {/*stdin=*/stdinFile.str(),
                                                 /*stdout=*/stdoutFile.str(),
                                                 /*stderr=*/stderrFile.str()};
  int ret = llvm::sys::ExecuteAndWait(
      xDSLOptPath,
      {xDSLOptPath, "-p",
       "convert-linalg-to-memref-stream,memref-streamify,convert-"
       "memref-stream-to-loops,scf-for-loop-flatten,"
       "arith-add-fastmath,loop-hoist-memref,lower-affine,convert-memref-"
       "stream-to-snitch,convert-func-to-"
       "riscv-func,convert-memref-to-riscv,convert-arith-to-riscv,"
       "convert-scf-to-riscv-scf,dce,reconcile-unrealized-casts,test-"
       "lower-snitch-stream-to-asm",
       "-t", "riscv-asm"},
      std::nullopt, redirects);
  if (ret != 0) {
    auto diagEmit =
        assertCompiled ? &Operation::emitError : &Operation::emitWarning;

    InFlightDiagnostic diag =
        ((func.getOperation())->*diagEmit)("Failed to translate ")
            .append(func.getSymName(), " with xDSL");

    if (llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
            llvm::MemoryBuffer::getFile(stderrFile, /*IsText=*/true))
      diag.attachNote() << "stderr:\n" << buffer.get()->getBuffer();

    if (assertCompiled) {
      signalPassFailure();
      return;
    }
  }

  // Function body no longer needed.
  func.getBody().getBlocks().clear();
  func.setVisibility(SymbolTable::Visibility::Private);
  // Required to tell the conversion pass to LLVM that this is actually a
  // call into the same linkage unit and does not have to be rewritten to a
  // HAL module call.
  func->setAttr("hal.import.bitcode", UnitAttr::get(&getContext()));

  if (ret != 0)
    return;

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(stdoutFile, /*IsText=*/true);
  if (!buffer) {
    func.emitOpError("failed to open ") << stdoutFile;
    signalPassFailure();
    return;
  }

  func->setAttr("riscv_assembly",
                StringAttr::get(&getContext(), (*buffer)->getBuffer()));
}
