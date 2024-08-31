#ifndef QUIDDITCH_DIALECT_SNITCH_DMAOPS
#define QUIDDITCH_DIALECT_SNITCH_DMAOPS

include "Quidditch/Dialect/DMA/IR/DMADialect.td"
include "Quidditch/Dialect/DMA/IR/DMATypes.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class DMA_Op<string mnemonic, list<Trait> traits = []> :
  Op<DMA_Dialect, mnemonic, traits>;

def DMA_StartTensorCopyOp : DMA_Op<"start_tensor_copy",
  [Pure, AllRanksMatch<["copy", "result"]>,
   DeclareOpInterfaceMethods<BufferizableOpInterface,
    ["resultBufferizesToMemoryWrite", "bufferizesToMemoryRead",
     "bufferizesToMemoryWrite", "getAliasingValues", "getBufferType",
      "bufferize", "bufferizesToAllocation"]>]> {

  let description = [{
    Operation starting a copy of a tensor to another memory space, optionally
    adding padding and returning it as a new tensor.
    The contained values of the resulting tensor is in an unspecified state.
    See `wait_for_tensor_copy` to transform the tensor value into a state
    equal to `$copy`.

    The operation may optionally add padding at the end of each dimension of
    the tensor. Zero is used as the padding value.
    The dimensions of the result tensor are computed using
    `dims(copy)[i] + high_pad[i]`.

    This operation is a noop if `$copy` is already in the given memory space,
    no padding is added, and bufferization can elide the copy.
  }];

  let arguments = (ins AnyRankedTensor:$copy,
                       AnyAttr:$memory_space,
                       Variadic<Index>:$high_pad,
                       OptionalAttr<DenseI64ArrayAttr>:$static_high_pad,
                       UnitAttr:$undef_padding
  );

  let results = (outs
    AnyRankedTensor:$result,
    DMA_TokenType:$token
  );

  let assemblyFormat = [{
    `of` $copy `to` $memory_space
    ( `pad` `with` (`undef` $undef_padding^) : (`zero`)? `by`
      custom<DynamicIndexList>($high_pad, $static_high_pad)^)?
    custom<TensorCopyTypes>(ref($static_high_pad), type($copy), type($result))
    attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$copy, "mlir::Attribute":$memorySpace), [{
      build($_builder, $_state, copy.getType(),
            $_builder.getType<TokenType>(), copy, memorySpace,
            /*high_pad=*/mlir::ValueRange(), /*static_high_pad=*/nullptr);
    }]>
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
  private:
    std::optional<bool>
    elidesAllocation(const mlir::bufferization::BufferizationOptions &options = {},
                      llvm::SmallVector<mlir::Value> *invocationStack = nullptr);
  public:

    bool hasPadding() {
      return static_cast<bool>(getStaticHighPadAttr());
    }

    llvm::SmallVector<mlir::OpFoldResult> getMixedHighPad();
  }];

  let hasFolder = 1;
}

def DMA_WaitForTensorCopyOp : DMA_Op<"wait_for_tensor_copy",
  [AllTypesMatch<["transfer_tensor", "result"]>, Pure,
   DeclareOpInterfaceMethods<BufferizableOpInterface,
    ["bufferizesToMemoryRead", "bufferizesToMemoryWrite", "getAliasingValues",
     "bufferize", "mustBufferizeInPlace", "isNotConflicting"]>]> {

  let description = [{
    Operation asserting that a previous `start_tensor_copy` operation has finished.
    Unless `token` is the result of an `completed_token` operation,
    `transfer_tensor` and `token` must at runtime be a token and tensor yielded
    by a `start_tensor_copy` operation and `copy` the original tensor used in
    `start_tensor_copy`.

    Once this operation returns, the returned tensor's values are guaranteed
    equal to the `copy` operand and in the memory space specified in
    `start_tensor_copy`.

    Note: The additional `copy` operand is given as it is effectively read by
    this operation.
    This additionally guarantees that the bufferization frame work does not
    perform a write to the underlying buffer of `copy` while the transfer is
    in progress.
  }];

  let arguments = (ins
    AnyRankedTensor:$transfer_tensor,
    DMA_TokenType:$token,
    AnyRankedTensor:$copy
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    `of` $copy `:` type($copy) `to` $transfer_tensor `using` $token `->` type($transfer_tensor) attr-dict
  }];

  let hasFolder = 1;
}

def DMA_StartTransferOp : DMA_Op<"start_transfer",
  [MemoryEffects<[MemWrite]>, SameOperandsElementType, SameOperandsShape]> {

  let description = [{
    Operation performing a DMA transfer from one MemRef to another.
    The shapes (including dynamic ones at runtime) of both MemRefs must be
    identical with different strides and offsets allowed.

    The DMA operation is likely (but not guaranteed) to run asynchronous and
    its completion only guaranteed by executing the `wait_for_transfers`
    operation with the token returned by this operation or a later one.
  }];

  let arguments = (ins
    Arg<Non0RankedMemRefOf<[AnyType]>, "source", [MemRead]>:$source,
    Arg<Non0RankedMemRefOf<[AnyType]>, "destination", [MemWrite]>:$dest
  );

  let results = (outs DMA_TokenType:$token);

  let assemblyFormat = [{
    `from` $source `:` type($source) `to` $dest `:` type($dest) attr-dict
  }];

  let hasFolder = 1;
}

def DMA_StartZeroMemTransferOp : DMA_Op<"start_zero_mem_transfer",
  [MemoryEffects<[MemWrite]>]> {

  let description = [{

  }];

  let arguments = (ins
    Arg<Non0RankedMemRefOf<[AnyType]>, "zeroed buffer", [MemWrite]>:$filled
  );

  let results = (outs DMA_TokenType:$token);

  let assemblyFormat = [{
    $filled `:` type($filled) attr-dict
  }];
}

def DMA_WaitForTransfersOp : DMA_Op<"wait_for_transfers"> {

  let description = [{
    Operation awaiting for DMA transfers denoted by its tokens to be finished.
  }];

  let arguments = (ins
    Variadic<DMA_TokenType>:$tokens
  );

  let assemblyFormat = [{
    ($tokens^ `:` type($tokens))? attr-dict
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def DMA_CompletedTokenOp
  : DMA_Op<"completed_token", [Pure, ConstantLike]> {

  let description = [{
    Op returning a special value representing a completed DMA transfer.
    Passing this token to `wait_for_transfers` will always return immediately.
  }];

  let results = (outs DMA_TokenType:$token);

  let assemblyFormat = [{
    attr-dict
  }];

  let hasFolder = 1;
}

#endif
