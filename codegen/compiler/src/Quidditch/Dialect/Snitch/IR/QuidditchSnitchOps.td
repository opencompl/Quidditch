#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHOPS
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHOPS

include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.td"
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

def FlatI8MemRef : ConfinedType<MemRefOf<[I8]>, [HasStaticShapePred,
  HasAnyRankOfPred<[1]>], "one-dimensional i8 MemRef of a static size">;

def QuidditchSnitch_L1MemoryViewOp : QuidditchSnitch_Op<"l1_memory_view",
  [Pure]> {
  let results = (outs FlatI8MemRef:$result);

  let assemblyFormat = [{
    `->` type($result) attr-dict
  }];
}

#endif
