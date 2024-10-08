#include "ConvertDMAToLLVM.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include "Quidditch/Dialect/DMA/IR/DMAOps.h"
#include "Quidditch/Dialect/SnitchDMA/IR/SnitchDMAOps.h"

using namespace mlir;
using namespace quidditch;
using namespace quidditch::dma;

/// Returns the number of potentially non-contiguous outer dimensions of
/// 'memRefType'. The remaining inner dimensions (i.e. all dimensions at index
/// 'NonContiguousOuterDims' to the MemRef rank) are known to be contiguous.
/// Returns failure if the layout attribute of the MemRef is unsupported.
static FailureOr<size_t> getNumNonContiguousOuterDims(MemRefType memRefType) {
  auto stridesAttr =
      dyn_cast_or_null<StridedLayoutAttr>(memRefType.getLayout());
  if (!stridesAttr) {
    if (memRefType.getLayout() && !memRefType.getLayout().isIdentity())
      return failure();

    // No layout or identity layouts are by definition fully contiguous.
    return 0;
  }

  int64_t innerSize = 1;
  ArrayRef<int64_t> shape = memRefType.getShape();
  ArrayRef<int64_t> strides = stridesAttr.getStrides();
  for (; !shape.empty();
       shape = shape.drop_back(), strides = strides.drop_back()) {
    int64_t dim = shape.back();
    // Unit dims can be dropped alongside the corresponding stride of that dim.
    if (dim == 1)
      continue;

    int64_t stride = strides.back();
    if (ShapedType::isDynamic(stride))
      break;

    if (innerSize != stride)
      break;

    // Note: Dim may be dynamic with the value -1. This intentionally will only
    // fail the 'if' above later if the outer dims are non-zero.
    innerSize *= dim;
  }

  return shape.size();
}

/// Returns true if this MemRef type is known to have a fully contiguous layout.
/// TODO: Could be upstreamed next to
/// 'memref::isStaticShapeAndContiguousRowMajor'
static bool isContiguous(MemRefType memRefType) {
  return getNumNonContiguousOuterDims(memRefType) == 0;
}

namespace {
struct StartTransferOpLowering : ConvertOpToLLVMPattern<StartTransferOp> {

  LLVM::LLVMFuncOp dmaStart1DFunc;
  LLVM::LLVMFuncOp dmaStart2DFunc;

  StartTransferOpLowering(LLVM::LLVMFuncOp dmaStart1DFunc,
                          LLVM::LLVMFuncOp dmaStart2DFunc,
                          const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter), dmaStart1DFunc(dmaStart1DFunc),
        dmaStart2DFunc(dmaStart2DFunc) {}

  LogicalResult
  matchAndRewrite(StartTransferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned rank = op.getSource().getType().getRank();
    if (rank != 1 && rank != 2)
      return failure();

    MemRefDescriptor sourceDescriptor(adaptor.getSource());
    MemRefDescriptor destDescriptor(adaptor.getDest());

    Value source = sourceDescriptor.bufferPtr(
        rewriter, op->getLoc(), *getTypeConverter(), op.getSource().getType());
    Value dest = destDescriptor.bufferPtr(
        rewriter, op->getLoc(), *getTypeConverter(), op.getDest().getType());

    Value elementSize = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(),
        rewriter.getI32IntegerAttr(llvm::divideCeil(
            op.getSource().getType().getElementTypeBitWidth(), 8)));
    Value innerSize = rewriter.create<LLVM::MulOp>(
        op->getLoc(), sourceDescriptor.size(rewriter, op->getLoc(), rank - 1),
        elementSize);
    if (rank == 1) {
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, dmaStart1DFunc,
                                                ValueRange{
                                                    dest,
                                                    source,
                                                    innerSize,
                                                });
    } else {
      Value sourceStride = rewriter.create<LLVM::MulOp>(
          op->getLoc(), sourceDescriptor.stride(rewriter, op->getLoc(), 0),
          elementSize);
      Value destStride = rewriter.create<LLVM::MulOp>(
          op->getLoc(), destDescriptor.stride(rewriter, op->getLoc(), 0),
          elementSize);
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          op, dmaStart2DFunc,
          ValueRange{
              dest,
              source,
              innerSize,
              destStride,
              sourceStride,
              sourceDescriptor.size(rewriter, op->getLoc(), 0),
          });
    }
    return success();
  }
};

// TODO: These should not be hardcoded.
constexpr unsigned zeroMemSize = 0x10000;
constexpr unsigned zeroMemAddress = 0x10030000;

struct StartContiguousZeroMemTransferOpOpLowering
    : ConvertOpToLLVMPattern<StartZeroMemTransferOp> {

  LLVM::LLVMFuncOp dmaStart1DFunc;
  LLVM::LLVMFuncOp dmaStart2DFunc;

  StartContiguousZeroMemTransferOpOpLowering(LLVM::LLVMFuncOp dmaStart1DFunc,
                                             LLVM::LLVMFuncOp dmaStart2DFunc,
                                             const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter, /*benefit=*/2),
        dmaStart1DFunc(dmaStart1DFunc), dmaStart2DFunc(dmaStart2DFunc) {}

  LogicalResult
  matchAndRewrite(StartZeroMemTransferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isContiguous(op.getFilled().getType()))
      return failure();

    Value zeroPointer = rewriter.create<LLVM::IntToPtrOp>(
        op->getLoc(), rewriter.getType<LLVM::LLVMPointerType>(),
        rewriter.create<LLVM::ConstantOp>(
            op->getLoc(), rewriter.getI32IntegerAttr(zeroMemAddress)));
    Value zeroMemSizeValue = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), rewriter.getI32IntegerAttr(zeroMemSize));

    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value size;

    auto filledDesc = MemRefDescriptor(adaptor.getFilled());

    MemRefType memRefType = op.getFilled().getType();
    SmallVector<Value> dynamicSizes;
    for (auto [index, shape] : llvm::enumerate(memRefType.getShape()))
      if (ShapedType::isDynamic(shape))
        dynamicSizes.push_back(filledDesc.size(rewriter, op->getLoc(), index));

    // Function does not support strided layout, even if it is contiguous.
    // Lie about it and remove it.
    // TODO: Consider fixing this upstream.
    // TODO: Make a clone method of `MemRefType` that changes just the layout.
    this->getMemRefDescriptorSizes(
        op->getLoc(),
        MemRefType::get(memRefType.getShape(), memRefType.getElementType()),
        dynamicSizes, rewriter, sizes, strides, size);

    Value zero =
        createIndexAttrConstant(rewriter, op->getLoc(), getIndexType(), 0);
    Value bufferPointer = filledDesc.bufferPtr(rewriter, op->getLoc(),
                                               *getTypeConverter(), memRefType);
    Value times2D =
        rewriter.create<LLVM::UDivOp>(op->getLoc(), size, zeroMemSizeValue);
    // Note: This call would not be legal as a 'start_dma_transfer' call as
    // MemRefs do not allow internal aliasing, which the below does via the
    // stride of 0.
    rewriter.create<LLVM::CallOp>(op->getLoc(), dmaStart2DFunc,
                                  ValueRange{bufferPointer, zeroPointer,
                                             zeroMemSizeValue, zeroMemSizeValue,
                                             zero, times2D});
    Value offset =
        rewriter.create<LLVM::MulOp>(op->getLoc(), times2D, zeroMemSizeValue);
    bufferPointer = rewriter.create<LLVM::GEPOp>(
        op->getLoc(), bufferPointer.getType(), rewriter.getI8Type(),
        bufferPointer, offset);
    Value rest =
        rewriter.create<LLVM::URemOp>(op->getLoc(), size, zeroMemSizeValue);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, dmaStart1DFunc, ValueRange{bufferPointer, zeroPointer, rest});
    return success();
  }
};

struct StartZeroMemTransferOpOpLowering
    : ConvertOpToLLVMPattern<StartZeroMemTransferOp> {

  using ConvertOpToLLVMPattern<StartZeroMemTransferOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StartZeroMemTransferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType memRefType = op.getFilled().getType();

    FailureOr<size_t> nonContiguousDims =
        getNumNonContiguousOuterDims(memRefType);
    if (failed(nonContiguousDims) || nonContiguousDims == 0)
      return failure();

    SmallVector<OpFoldResult> sizes =
        memref::getMixedSizes(rewriter, op->getLoc(), op.getFilled());

    SmallVector<Value> lowerBounds;
    SmallVector<Value> upperBounds;
    SmallVector<Value> steps;
    Value zeroIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    Value oneIndex = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 1);
    for (size_t index : llvm::seq(*nonContiguousDims)) {
      lowerBounds.push_back(zeroIndex);
      steps.push_back(oneIndex);
      upperBounds.push_back(getValueOrCreateConstantIndexOp(
          rewriter, op->getLoc(), sizes[index]));
    }

    // Loop over every non-contiguous dimension to zero every contiguous
    // inner subview.
    Value completedToken = rewriter.create<CompletedTokenOp>(op->getLoc());
    scf::LoopNest loopNest = scf::buildLoopNest(
        rewriter, op->getLoc(), lowerBounds, upperBounds, steps, completedToken,
        [&](OpBuilder &builder, Location loc, ValueRange ivs,
            ValueRange iterArgs) -> scf::ValueVector {
          SmallVector<OpFoldResult> offsets = ivs;
          SmallVector<OpFoldResult> subSizes(*nonContiguousDims,
                                             rewriter.getIndexAttr(1));
          for (unsigned i :
               llvm::seq<unsigned>(*nonContiguousDims, memRefType.getRank())) {
            offsets.push_back(rewriter.getIndexAttr(0));
            subSizes.push_back(sizes[i]);
          }
          SmallVector<OpFoldResult> strides(memRefType.getRank(),
                                            rewriter.getIndexAttr(1));

          Value subMemRef = rewriter.create<memref::SubViewOp>(
              loc, op.getFilled(), offsets, subSizes, strides);
          return {
              builder.create<StartZeroMemTransferOp>(op->getLoc(), subMemRef)};
        });

    Type tokenType = typeConverter->convertType(op.getType());
    rewriter.replaceOp(
        op, typeConverter->materializeTargetConversion(
                rewriter, op->getLoc(), tokenType, loopNest.results.front()));
    return success();
  }
};

struct WaitForTransferOpLowering : ConvertOpToLLVMPattern<WaitForTransferOp> {

  using ConvertOpToLLVMPattern<WaitForTransferOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(WaitForTransferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block *prev = op->getBlock();
    Block *body = rewriter.splitBlock(prev, op->getIterator());
    Block *after = rewriter.splitBlock(body, op->getNextNode()->getIterator());
    rewriter.setInsertionPointToEnd(prev);
    rewriter.create<LLVM::BrOp>(op->getLoc(), body);

    rewriter.setInsertionPointToEnd(body);
    Value lastCompleted = rewriter.create<SnitchDMA::StatOp>(op->getLoc());
    Value notDone =
        rewriter.create<LLVM::ICmpOp>(op->getLoc(), LLVM::ICmpPredicate::ult,
                                      lastCompleted, adaptor.getToken());
    rewriter.create<LLVM::CondBrOp>(op->getLoc(), notDone, body, after);

    rewriter.setInsertionPointToStart(after);
    rewriter.eraseOp(op);
    return success();
  }
};

struct CompletedTokenOpLowering : ConvertOpToLLVMPattern<CompletedTokenOp> {

  using ConvertOpToLLVMPattern<CompletedTokenOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CompletedTokenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(
        op, typeConverter->convertType(op.getType()), 0);
    return success();
  }
};

struct CombineTokensOpLowering : ConvertOpToLLVMPattern<CombineTokensOp> {

  using ConvertOpToLLVMPattern<CombineTokensOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CombineTokensOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getTokens().empty()) {
      rewriter.replaceOpWithNewOp<CompletedTokenOp>(op);
      return success();
    }

    // TODO: Note that this lowering only works for Snitch's single channel DMA!
    Value current = adaptor.getTokens().front();
    for (Value iter : llvm::drop_begin(adaptor.getTokens()))
      current = rewriter.create<LLVM::UMaxOp>(op->getLoc(), current, iter);

    rewriter.replaceOp(op, current);
    return success();
  }
};

struct StatOpLowering : ConvertOpToLLVMPattern<SnitchDMA::StatOp> {
  using ConvertOpToLLVMPattern<SnitchDMA::StatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SnitchDMA::StatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::InlineAsmOp>(
        op, /*res=*/rewriter.getI32Type(),
        /*operands=*/ValueRange(),
        // dmstati $0, 0
        // opcode6=0x2b, func3=0, func7=0b100, rd=$0, rs1=zero,
        // rs2=imm5(0)
        ".insn r 0x2b, 0, 0b100, $0, zero, zero\n",
        /*constraints=*/"=r",
        /*has_side_effects=*/true, /*is_align_stack=*/false,
        /*asm_dialect=*/nullptr, /*operand_attrs=*/nullptr);
    return success();
  }
};

} // namespace

void quidditch::populateDMAToLLVMConversionPatterns(
    mlir::ModuleOp moduleOp, LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns) {

  typeConverter.addConversion(
      [](TokenType token) { return IntegerType::get(token.getContext(), 32); });

  auto builder = OpBuilder::atBlockEnd(moduleOp.getBody());
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

  patterns.insert<CompletedTokenOpLowering, WaitForTransferOpLowering,
                  StartZeroMemTransferOpOpLowering, StatOpLowering,
                  CombineTokensOpLowering>(typeConverter);
  patterns.insert<StartTransferOpLowering>(dmaStart1D, dmaStart2D,
                                           typeConverter);
  patterns.insert<StartContiguousZeroMemTransferOpOpLowering>(
      dmaStart1D, dmaStart2D, typeConverter);
}
