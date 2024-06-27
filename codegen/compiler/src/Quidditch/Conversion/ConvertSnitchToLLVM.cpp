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

  auto builder = OpBuilder::atBlockEnd(getOperation().getBody());
  auto ptrType = builder.getType<LLVM::LLVMPointerType>();
  IntegerType i32 = builder.getI32Type();
  IntegerType sizeT = i32;
  auto barrier = builder.create<LLVM::LLVMFuncOp>(
      builder.getUnknownLoc(), "snrt_cluster_hw_barrier",
      LLVM::LLVMFunctionType::get(builder.getType<LLVM::LLVMVoidType>(),
                                  ArrayRef<Type>{}));
  barrier->setAttr("hal.import.bitcode", builder.getUnitAttr());

  RewritePatternSet patterns(&getContext());
  patterns.insert<L1MemoryViewOpLowering>(typeConverter);
  patterns.insert<BarrierOpLowering>(barrier, typeConverter);

  LLVMConversionTarget target(getContext());
  target.markUnknownOpDynamicallyLegal([](auto) { return true; });
  target.addIllegalDialect<QuidditchSnitchDialect>();
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
