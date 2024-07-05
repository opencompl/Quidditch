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

def QuidditchSnitch_CopyTensorOp : QuidditchSnitch_Op<"copy_tensor",
  [SameOperandsAndResultType, NoMemoryEffect,
   DeclareOpInterfaceMethods<BufferizableOpInterface,
    ["resultBufferizesToMemoryWrite", "bufferizesToMemoryRead",
     "bufferizesToMemoryWrite", "getAliasingValues", "getBufferType",
      "bufferize", "bufferizesToAllocation"]>]> {

  let description = [{
    Operation performing a copy of a tensor to another memory spaces.
    If `$transfers_to_l1` is true, then the op ensures that the resulting
    tensor is in L1 memory.
    Otherwise, the output tensor is guaranteed to be in L3 memory.
    This operation is a noop if `$copy` and `$result` are already in the same
    memory space and bufferization can elide the copy.
  }];

  // TODO: Not a big fan of the UnitAttr. This should be an enum.
  let arguments = (ins AnyRankedTensor:$copy, UnitAttr:$transfers_to_l1);

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $copy `to` ( `L1` $transfers_to_l1^) : (`L3`)? `:` type($copy) attr-dict
  }];

  let hasFolder = 1;
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
