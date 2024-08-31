#include "DMACoreSpecializationOpInterfaceImpl.h"

#include "Quidditch/Dialect/DMA/IR/DMADialect.h"
#include "Quidditch/Dialect/DMA/IR/DMAOps.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchInterfaces.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace quidditch::dma;
using namespace quidditch::Snitch;

namespace {

//===----------------------------------------------------------------------===//
// StartTransferOp::DMACoreSpecializationOpInterface
//===----------------------------------------------------------------------===//

struct StartTransferOpImpl
    : CoreSpecializationOpInterface::ExternalModel<StartTransferOpImpl,
                                                   StartTransferOp> {
  void replaceWithNoop(Operation *op, RewriterBase &rewriter) const {
    rewriter.replaceOpWithNewOp<CompletedTokenOp>(op);
  }
};

struct StartTransferOpDMAImpl
    : DMACoreSpecializationOpInterface::ExternalModel<StartTransferOpDMAImpl,
                                                      StartTransferOp> {};

//===----------------------------------------------------------------------===//
// StartZeroMemTransferOp::DMACoreSpecializationOpInterface
//===----------------------------------------------------------------------===//

struct StartZeroMemTransferOpImpl
    : CoreSpecializationOpInterface::ExternalModel<StartZeroMemTransferOpImpl,
                                                   StartZeroMemTransferOp> {
  void replaceWithNoop(Operation *op, RewriterBase &rewriter) const {
    rewriter.replaceOpWithNewOp<CompletedTokenOp>(op);
  }

  // bool needsSynchronization(Operation *op) const { return true; }
};

struct StartZeroMemTransferOpDMAImpl
    : DMACoreSpecializationOpInterface::ExternalModel<
          StartZeroMemTransferOpDMAImpl, StartZeroMemTransferOp> {};

//===----------------------------------------------------------------------===//
// WaitForTransfersOpImpl::DMACoreSpecializationOpInterface
//===----------------------------------------------------------------------===//

struct WaitForTransfersOpImpl
    : CoreSpecializationOpInterface::ExternalModel<WaitForTransfersOpImpl,
                                                   WaitForTransfersOp> {
  void replaceWithNoop(Operation *op, RewriterBase &rewriter) const {
    rewriter.eraseOp(op);
  }

  bool needsSynchronization(Operation *op) const { return true; }
};

struct WaitForTransfersOpDMAImpl
    : DMACoreSpecializationOpInterface::ExternalModel<WaitForTransfersOpDMAImpl,
                                                      WaitForTransfersOp> {};

} // namespace

void quidditch::dma::registerDMACoreSpecializationOpInterface(
    mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *context, DMADialect *dialect) {
#define REGISTER_IMPLS(Op) Op::attachInterface<Op##Impl, Op##DMAImpl>(*context)
    REGISTER_IMPLS(StartTransferOp);
    REGISTER_IMPLS(StartZeroMemTransferOp);
    REGISTER_IMPLS(WaitForTransfersOp);
  });
}
