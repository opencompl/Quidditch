#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHOPS
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHOPS

include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.td"
include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchTypes.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class QuidditchSnitch_Op<string mnemonic, list<Trait> traits = []> :
  Op<QuidditchSnitch_Dialect, mnemonic, traits>;

def QuidditchSnitch_TensorMicrokernelOp : QuidditchSnitch_Op<"tensor.microkernel",
  [SingleBlock, NoRegionArguments,
   DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getRegionInvocationBounds"]>,
   DeclareOpInterfaceMethods<BufferizableOpInterface, ["bufferize",
    "getAliasingOpOperands", "getBufferType"]>]> {

  let description = [{

  }];

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    (`->` type($results)^ )? $body attr-dict
  }];

  let extraClassDeclaration = [{
    MicrokernelYieldOp getYieldOp() {
      return llvm::cast<MicrokernelYieldOp>(getBody().back().getTerminator());
    }
  }];
}

def QuidditchSnitch_MicrokernelYieldOp
  : QuidditchSnitch_Op<"microkernel_yield", [Pure, Terminator,
    ParentOneOf<["TensorMicrokernelOp", "MemRefMicrokernelOp"]>, ReturnLike,
    DeclareOpInterfaceMethods<BufferizableOpInterface,
      ["bufferize", "bufferizesToMemoryRead", "bufferizesToMemoryWrite",
       "getAliasingValues", "mustBufferizeInPlace"]>]> {
  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = [{
    $results (`:` type($results)^)? attr-dict
  }];
}

def QuidditchSnitch_MemRefMicrokernelOp
  : QuidditchSnitch_Op<"memref.microkernel", [IsolatedFromAbove, SingleBlock,
      DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands", "getRegionInvocationBounds"]>]> {

  let description = [{

  }];

  let arguments = (ins Variadic<AnyType>:$inputs);

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `` `(` $inputs `)` `:` functional-type($inputs, $results) $body attr-dict
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{

    MicrokernelYieldOp getYieldOp();

    mlir::Block* createEntryBlock();
  }];
}

def QuidditchSnitch_CopyL1TensorOp : QuidditchSnitch_Op<"copy_l1_tensor",
  [SameOperandsAndResultType, NoMemoryEffect,
   DeclareOpInterfaceMethods<BufferizableOpInterface,
    ["resultBufferizesToMemoryWrite", "bufferizesToMemoryRead",
     "bufferizesToMemoryWrite", "getAliasingValues", "getBufferType",
      "bufferize"]>]> {

  let description = [{
    Operation performing copies of tensors between memory spaces.
    If `$transfer_to_l1` is true, then the op ensures that the resulting tensor
    is in L1.
    Otherwise, the output tensor is guaranteed to be in L3 memory.
    This operation is a noop if `$copy` and `$result` are already in the same
    memory space.
  }];

  // TODO: Not a big fan of the UnitAttr. This should be an enum.
  let arguments = (ins AnyStaticShapeTensor:$copy, UnitAttr:$transfer_to_l1);

  let results = (outs AnyStaticShapeTensor:$result);

  let assemblyFormat = [{
    $copy `to` ( `L1` $transfer_to_l1^) : (`L3`)? `:` type($copy) attr-dict
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

  // TODO: In reality what the constraint here is, is that both of them must either be contiguous or if a dimension is
  //       not contiguous in one of them, all dimensions prior to that must be contiguous (i.e. of equal size).
  //       Can support higher dimensions and funkier strides once needed.
  let arguments = (ins
    Arg<MemRefRankOf<[AnyType], [1, 2]>, "source", [MemRead]>:$source,
    Arg<MemRefRankOf<[AnyType], [1, 2]>, "destination", [MemWrite]>:$dest
  );

  let results = (outs QuidditchSnitch_DMATokenType:$token);

  let assemblyFormat = [{
    `from` $source `:` type($source) `to` $dest `:` type($dest) attr-dict
  }];
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
}

def QuidditchSnitch_BarrierOp : QuidditchSnitch_Op<"barrier"> {
  let assemblyFormat = [{
    attr-dict
  }];
}

#endif
