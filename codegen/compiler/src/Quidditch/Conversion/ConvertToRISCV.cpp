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

static Type transformType(Type type) {
  auto memRefType = dyn_cast<MemRefType>(type);
  if (!memRefType)
    return type;

  auto strided = dyn_cast_or_null<StridedLayoutAttr>(memRefType.getLayout());
  if (!strided)
    return type;

  auto strideReplacement =
      StridedLayoutAttr::get(type.getContext(), 0, strided.getStrides());
  if (strideReplacement.isIdentity())
    return MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                           nullptr, memRefType.getMemorySpace());

  return MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                         strideReplacement, memRefType.getMemorySpace());
}

FailureOr<StringAttr>
ConvertToRISCV::convertToRISCVAssembly(MemRefMicrokernelOp kernelOp,
                                       StringAttr kernelName) {
  if (!llvm::all_of(kernelOp.getBody().getArgumentTypes(),
                    CallMicrokernelOp::supportsArgumentType)) {
    auto emit = assertCompiled ? &MemRefMicrokernelOp::emitError
                               : &MemRefMicrokernelOp::emitWarning;

    (kernelOp.*emit)("function inputs ")
        << kernelOp.getBody().getArgumentTypes()
        << " do not support bare-pointer calling convention required by "
           "xDSL.";
    return failure();
  }

  SmallVector<Type> argumentTypes =
      llvm::map_to_vector(kernelOp.getBody().getArgumentTypes(), transformType);

  OpBuilder builder(&getContext());
  OwningOpRef<func::FuncOp> tempFuncOp =
      builder.create<func::FuncOp>(kernelOp.getLoc(), kernelName,
                                   builder.getFunctionType(argumentTypes, {}));
  IRMapping mapping;
  kernelOp.getBody().cloneInto(&tempFuncOp->getBody(), mapping);
  for (BlockArgument argument : tempFuncOp->getArguments())
    argument.setType(transformType(argument.getType()));

  builder.setInsertionPointToEnd(&tempFuncOp->getBody().back());

  builder.create<func::ReturnOp>(kernelOp.getLoc());

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
       "arith-add-fastmath,"
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

      auto builder = IRRewriter(kernelOp);
      builder.inlineBlockBefore(&kernelOp.getBody().front(), kernelOp,
                                kernelOp.getInputs());
      kernelOp.erase();
      return WalkResult::advance();
    }

    auto builder = OpBuilder(kernelOp);
    // Assigning names here for deterministic names when lowered.
    builder.create<CallMicrokernelOp>(kernelOp.getLoc(), kernelName,
                                      kernelOp.getInputs(), *riscvAssembly);
    kernelOp.erase();
    return WalkResult::advance();
  });
}
