#include "Passes.h"

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.h"
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchOps.h"

namespace quidditch {
#define GEN_PASS_DEF_CONVERTSNITCHTOLLVMPASS
#include "Quidditch/Conversion/Passes.h.inc"
} // namespace quidditch

namespace {
class ConvertSnitchToLLVM
    : public quidditch::impl::ConvertSnitchToLLVMPassBase<ConvertSnitchToLLVM> {
public:
  using Base::Base;

protected:
  void runOnOperation() override;
};
} // namespace

using namespace mlir;
using namespace quidditch::Snitch;

namespace {
struct L1MemoryViewOpLowering : ConvertOpToLLVMPattern<L1MemoryViewOp> {
  using ConvertOpToLLVMPattern<L1MemoryViewOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(L1MemoryViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value size;

    this->getMemRefDescriptorSizes(op->getLoc(), op.getType(),
                                   adaptor.getOperands(), rewriter, sizes,
                                   strides, size);

    // TODO: This is horribly hardcoded when it shouldn't be.
    Value l1Address = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(0x10000000));
    Value allocatedPtr = rewriter.create<LLVM::IntToPtrOp>(
        op->getLoc(), rewriter.getType<LLVM::LLVMPointerType>(), l1Address);

    auto memRefDescriptor =
        this->createMemRefDescriptor(op->getLoc(), op.getType(), allocatedPtr,
                                     allocatedPtr, sizes, strides, rewriter);

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, {memRefDescriptor});
    return success();
  }
};

/// Returns true if this MemRef type is known to have a fully contiguous layout.
bool isContiguous(MemRefType memRefType) {
  MemRefLayoutAttrInterface layout = memRefType.getLayout();
  if (!layout || layout.isIdentity())
    return true;

  // It is impossible to statically determine contiguity with dynamic strides.
  auto strided = dyn_cast<StridedLayoutAttr>(layout);
  if (!strided || llvm::any_of(strided.getStrides(), ShapedType::isDynamic))
    return false;

  // Calculate what the strides would be if it had an identity layout and check
  // that they match.
  ArrayRef<int64_t> shape = memRefType.getShape();
  ArrayRef<int64_t> strides = strided.getStrides();
  std::uint64_t currentIdentityStride = 1;
  for (auto [dim, stride] : llvm::zip_equal(llvm::reverse(shape.drop_front()),
                                            strides.drop_front())) {
    if (currentIdentityStride != stride)
      return false;

    if (ShapedType::isDynamic(dim))
      return false;
    currentIdentityStride *= dim;
  }
  return currentIdentityStride == strided.getStrides().front();
}

struct StartDMATransferOp1DLowering
    : ConvertOpToLLVMPattern<StartDMATransferOp> {

  LLVM::LLVMFuncOp dmaStart1DFunc;

  StartDMATransferOp1DLowering(LLVM::LLVMFuncOp dmaStart1DFunc,
                               const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter, /*benefit=*/2),
        dmaStart1DFunc(dmaStart1DFunc) {}

  LogicalResult match(StartDMATransferOp op) const override {
    return success(isContiguous(op.getSource().getType()) &&
                   isContiguous(op.getDest().getType()));
  }

  void rewrite(StartDMATransferOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override {
    MemRefDescriptor sourceDescriptor(adaptor.getSource());
    MemRefDescriptor destDescriptor(adaptor.getDest());

    Value source = sourceDescriptor.bufferPtr(
        rewriter, op->getLoc(), *getTypeConverter(), op.getSource().getType());
    Value dest = destDescriptor.bufferPtr(
        rewriter, op->getLoc(), *getTypeConverter(), op.getSource().getType());

    MemRefType sourceMemRef = op.getSource().getType();
    SmallVector<Value> dynamicSizes;
    for (std::int64_t dim : sourceMemRef.getShape())
      if (ShapedType::isDynamic(dim))
        dynamicSizes.push_back(
            sourceDescriptor.size(rewriter, op->getLoc(), dim));

    SmallVector<Value> sizes;
    SmallVector<Value> strides;
    Value totalSize;
    getMemRefDescriptorSizes(
        op->getLoc(),
        // Offsets are not considered an identity layout.
        // Get rid of the layout entirely for the size calculation.
        MemRefType::get(sourceMemRef.getShape(), sourceMemRef.getElementType(),
                        nullptr, sourceMemRef.getMemorySpace()),
        dynamicSizes, rewriter, sizes, strides, totalSize);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, dmaStart1DFunc,
                                              ValueRange{
                                                  dest,
                                                  source,
                                                  totalSize,
                                              });
  }
};

struct StartDMATransferOp2DLowering
    : ConvertOpToLLVMPattern<StartDMATransferOp> {

  LLVM::LLVMFuncOp dmaStart2DFunc;

  StartDMATransferOp2DLowering(LLVM::LLVMFuncOp dmaStart2DFunc,
                               const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter), dmaStart2DFunc(dmaStart2DFunc) {}

  LogicalResult
  matchAndRewrite(StartDMATransferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getSource().getType().getRank() != 2)
      return failure();

    MemRefDescriptor sourceDescriptor(adaptor.getSource());
    MemRefDescriptor destDescriptor(adaptor.getDest());

    Value source = sourceDescriptor.bufferPtr(
        rewriter, op->getLoc(), *getTypeConverter(), op.getSource().getType());
    Value dest = destDescriptor.bufferPtr(
        rewriter, op->getLoc(), *getTypeConverter(), op.getSource().getType());

    Value size = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(),
        rewriter.getI32IntegerAttr(llvm::divideCeil(
            op.getSource().getType().getElementTypeBitWidth(), 8)));
    size = rewriter.create<LLVM::MulOp>(
        op->getLoc(), size, sourceDescriptor.size(rewriter, op->getLoc(), 0));

    Value sourceStride = sourceDescriptor.stride(rewriter, op->getLoc(), 1);
    Value destStride = destDescriptor.stride(rewriter, op->getLoc(), 1);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, dmaStart2DFunc,
        ValueRange{dest, source, size, destStride, sourceStride,
                   sourceDescriptor.size(rewriter, op->getLoc(), 1)});
    return success();
  }
};

struct WaitForDMATransfersOpLowering
    : ConvertOpToLLVMPattern<WaitForDMATransfersOp> {

  LLVM::LLVMFuncOp waitFunc;

  WaitForDMATransfersOpLowering(LLVM::LLVMFuncOp waitFunc,
                                const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter), waitFunc(waitFunc) {}

  LogicalResult
  matchAndRewrite(WaitForDMATransfersOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: This should wait for only a specific transfer not all.
    //    for (Value token : adaptor.getTokens())
    //      rewriter.create<LLVM::CallOp>(op->getLoc(), waitFunc, token);
    rewriter.create<LLVM::CallOp>(op->getLoc(), waitFunc, ValueRange());
    rewriter.eraseOp(op);
    return success();
  }
};

struct BarrierOpLowering : ConvertOpToLLVMPattern<BarrierOp> {

  LLVM::LLVMFuncOp barrierFunc;

  BarrierOpLowering(LLVM::LLVMFuncOp barrierFunc,
                    const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter), barrierFunc(barrierFunc) {}

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, barrierFunc, ValueRange());
    return success();
  }
};
} // namespace

void ConvertSnitchToLLVM::runOnOperation() {

  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(&getContext(),
                             dataLayoutAnalysis.getAtOrAbove(getOperation()));
  // TODO: This is horribly hardcoded when it shouldn't be.
  options.overrideIndexBitwidth(32);
  LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);
  typeConverter.addConversion([](DMATokenType token) {
    return IntegerType::get(token.getContext(), 32);
  });

  auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
  auto ptrType = builder.getType<LLVM::LLVMPointerType>();
  IntegerType i32 = builder.getI32Type();
  IntegerType sizeT = i32;
  auto dmaStart1D = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_dma_start_1d",
      LLVM::LLVMFunctionType::get(i32,
                                  ArrayRef<Type>{ptrType, ptrType, sizeT}));
  dmaStart1D->setAttr("hal.import.bitcode", builder.getUnitAttr());

  auto dmaStart2D = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_dma_start_2d",
      LLVM::LLVMFunctionType::get(
          i32, ArrayRef<Type>{ptrType, ptrType, sizeT, sizeT, sizeT, sizeT}));
  dmaStart2D->setAttr("hal.import.bitcode", builder.getUnitAttr());

  // TODO: This should wait for only a specific transfer not all.
  //       This is currently bugged in the snitch_cluster repo and potentially
  //       the hardware.
  auto dmaWait = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_dma_wait_all",
      LLVM::LLVMFunctionType::get(builder.getType<LLVM::LLVMVoidType>(),
                                  ArrayRef<Type>{}));
  dmaWait->setAttr("hal.import.bitcode", builder.getUnitAttr());

  auto barrier = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_cluster_hw_barrier",
      LLVM::LLVMFunctionType::get(builder.getType<LLVM::LLVMVoidType>(),
                                  ArrayRef<Type>{}));
  barrier->setAttr("hal.import.bitcode", builder.getUnitAttr());

  RewritePatternSet patterns(&getContext());
  patterns.insert<L1MemoryViewOpLowering>(typeConverter);
  patterns.insert<StartDMATransferOp1DLowering>(dmaStart1D, typeConverter);
  patterns.insert<StartDMATransferOp2DLowering>(dmaStart2D, typeConverter);
  patterns.insert<WaitForDMATransfersOpLowering>(dmaWait, typeConverter);
  patterns.insert<BarrierOpLowering>(barrier, typeConverter);

  LLVMConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](auto) { return true; });
  target.addIllegalDialect<QuidditchSnitchDialect>();
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
