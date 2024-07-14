#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHOPS
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHOPS

include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.td"
include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchTypes.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class QuidditchSnitch_Op<string mnemonic, list<Trait> traits = []> :
  Op<QuidditchSnitch_Dialect, mnemonic, traits>;

def QuidditchSnitch_MemRefMicrokernelOp
  : QuidditchSnitch_Op<"memref.microkernel", [IsolatedFromAbove, SingleBlock,
      NoTerminator]> {

  let description = [{
    Operation denoting a region of operations as a microkernel.
    The region is `IsolatedFromAbove` making all inputs to the microkernel explicit.
    A later compilation step turns the "uncompiled" microkernel into a compiled
    `call_microkernel` operation.

    Operations within the Microkernel may be executed asynchronous.
    Side-effects of operations are therefore only guaranteed to be visible
    after a subsequent invocation of `microkernel_fence`.
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `` `(` $inputs `)` (`:` type($inputs)^)? $body attr-dict
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    mlir::Block* createEntryBlock();
  }];
}

def QuidditchSnitch_CallMicrokernelOp
  : QuidditchSnitch_Op<"call_microkernel"> {

  let description = [{
    Operation denoting a call to a compiled microkernel.
    The compiled artifact is available as the `riscv_assembly` attribute.
    `name` is used as a hint for the symbol name of the microkernel and has no
    semantics.

    The Microkernel may be executed asynchronous.
    Side-effects of operations are therefore only guaranteed to be visible
    after a subsequent invocation of `microkernel_fence`.
  }];

  let arguments = (ins StrAttr:$name, Variadic<AnyType>:$inputs, StrAttr:$riscv_assembly);

  let assemblyFormat = [{
    $name `(` $inputs `)` (`:` type($inputs)^)? `[``{`
    custom<RISCVAssembly>($riscv_assembly)
    `}` `]` attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static bool supportsArgumentType(mlir::Type type);
  }];
}

def QuidditchSnitch_MicrokernelFenceOp : QuidditchSnitch_Op<"microkernel_fence"> {

  let description = [{
    Execution of this operation guarantees that the side-effects of all
    previous microkernel invocations are visible as soon as this operation
    returns.
  }];

  let assemblyFormat = [{
    attr-dict
  }];
}

def QuidditchSnitch_StartTensorCopyOp : QuidditchSnitch_Op<"start_tensor_copy",
  [AllTypesMatch<["copy", "result"]>, Pure,
   DeclareOpInterfaceMethods<BufferizableOpInterface,
    ["resultBufferizesToMemoryWrite", "bufferizesToMemoryRead",
     "bufferizesToMemoryWrite", "getAliasingValues", "getBufferType",
      "bufferize", "bufferizesToAllocation"]>]> {

  let description = [{
    Operation starting a copy of a tensor to L1 memory space returning it as
    a new tensor.
    The contained values of the tensor in an unspecified state.
    See `wait_for_tensor_copy` to transform the tensor value into a state
    equal to `$copy`.

    This operation is a noop if `$copy` and `$result` are already in L1 and
    bufferization can elide the copy.
  }];

  let arguments = (ins AnyRankedTensor:$copy);

  let results = (outs
    AnyRankedTensor:$result,
    QuidditchSnitch_DMATokenType:$token
  );

  let assemblyFormat = [{
    $copy `to` `L1` `:` type($copy) attr-dict
  }];
}

def QuidditchSnitch_WaitForTensorCopyOp : QuidditchSnitch_Op<"wait_for_tensor_copy",
  [AllTypesMatch<["transfer_tensor", "result", "copy"]>, Pure,
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
    equal to the `copy` operand and in L1 memory.

    Note: The additional `copy` operand is given as it is effectively read by
    this operation.
    This additionally guarantees that the bufferization frame work does not
    perform a write to the underlying buffer of `copy` while the transfer is
    in progress.
  }];

  let arguments = (ins
    AnyRankedTensor:$transfer_tensor,
    QuidditchSnitch_DMATokenType:$token,
    AnyRankedTensor:$copy
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    `of` $copy `to` $transfer_tensor `using` $token `:` type($transfer_tensor) attr-dict
  }];
}

def FlatI8MemRef : ConfinedType<MemRefOf<[I8]>, [HasStaticShapePred,
  HasAnyRankOfPred<[1]>], "one-dimensional i8 MemRef of a static size">;

def QuidditchSnitch_L1MemoryViewOp : QuidditchSnitch_Op<"l1_memory_view",
  [Pure]> {
  let results = (outs FlatI8MemRef:$result);

  let assemblyFormat = [{
    `->` type($result) attr-dict
  }];
}

def QuidditchSnitch_StartDMATransferOp : QuidditchSnitch_Op<"start_dma_transfer",
  [MemoryEffects<[MemWrite]>, SameOperandsElementType, SameOperandsShape]> {

  let description = [{
    Operation performing a DMA transfer from one MemRef to another.
    At least one of the two MemRefs must be in L1 memory.
    The shapes (including dynamic ones at runtime) of both MemRefs must be
    identical with different strides and offsets allowed.

    The DMA operation is likely (but not guaranteed) to run asynchronous and
    its completion only guaranteed by executing the `wait_for_dma_transfers`
    operation with the token returned by this operation or a later one.
  }];

  let arguments = (ins
    Arg<Non0RankedMemRefOf<[AnyType]>, "source", [MemRead]>:$source,
    Arg<Non0RankedMemRefOf<[AnyType]>, "destination", [MemWrite]>:$dest
  );

  let results = (outs QuidditchSnitch_DMATokenType:$token);

  let assemblyFormat = [{
    `from` $source `:` type($source) `to` $dest `:` type($dest) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}

def QuidditchSnitch_WaitForDMATransfersOp
  : QuidditchSnitch_Op<"wait_for_dma_transfers"> {

  let description = [{
    Operation awaiting for DMA transfers denoted by its tokens to be finished.
  }];

  let arguments = (ins
    Variadic<QuidditchSnitch_DMATokenType>:$tokens
  );

  let assemblyFormat = [{
    $tokens `:` type($tokens) attr-dict
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def QuidditchSnitch_CompletedTokenOp
  : QuidditchSnitch_Op<"completed_token", [Pure]> {

  let description = [{
    Op returning a special value representing a completed DMA transfer.
    Passing this token to `wait_for_dma_transfers` will always return immediately.
  }];

  let results = (outs QuidditchSnitch_DMATokenType:$token);

  let assemblyFormat = [{
    attr-dict
  }];
}

def QuidditchSnitch_BarrierOp : QuidditchSnitch_Op<"barrier"> {
  let assemblyFormat = [{
    attr-dict
  }];
}

def QuidditchSnitch_ClusterIndexOp
  : QuidditchSnitch_Op<"cluster_index", [Pure]> {

  let results = (outs Index:$result);

  let assemblyFormat = [{
    attr-dict
  }];
}

#endif
