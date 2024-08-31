#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHOPS
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHOPS

include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.td"
include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchInterfaces.td"
include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchTypes.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class QuidditchSnitch_Op<string mnemonic, list<Trait> traits = []> :
  Op<QuidditchSnitch_Dialect, mnemonic, traits>;

def QuidditchSnitch_TensorMicrokernelOp : QuidditchSnitch_Op<"tensor.microkernel",
  [SingleBlock, NoRegionArguments, RecursivelySpeculatable, RecursiveMemoryEffects,
   DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getRegionInvocationBounds"]>,
   DeclareOpInterfaceMethods<BufferizableOpInterface, ["bufferize",
    "getAliasingOpOperands", "getBufferType"]>]> {

  let description = [{
    Pre-bufferization version of `memref.microkernel`.
    Unlike `memref.microkernel` it is not isolated from above and may also
    return tensor operations as outputs via `microkernel_yield`.

    Like `memref.microkernel`, operations within the kernel may be executing
    asynchronously and cannot be used directly.
    A `sync_tensor` operation must be used to make any result tensor of this
    operation available.
    Failing to do so results in unspecified values within the tensor.
  }];

  let results = (outs Variadic<AnyRankedTensor>:$results);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    (`->` type($results)^ )? $body attr-dict
  }];

  let extraClassDeclaration = [{
    MicrokernelYieldOp getYieldOp() {
      return llvm::cast<MicrokernelYieldOp>(getBody().back().getTerminator());
    }

    mlir::Block* createEntryBlock();
  }];
}

def QuidditchSnitch_MicrokernelYieldOp
  : QuidditchSnitch_Op<"microkernel_yield", [Pure, Terminator,
    HasParent<"TensorMicrokernelOp">, ReturnLike,
    DeclareOpInterfaceMethods<BufferizableOpInterface,
      ["bufferize", "bufferizesToMemoryRead", "bufferizesToMemoryWrite",
       "getAliasingValues", "mustBufferizeInPlace"]>]> {
  let arguments = (ins Variadic<AnyRankedTensor>:$results);

  let assemblyFormat = [{
    $results (`:` type($results)^)? attr-dict
  }];
}

def QuidditchSnitch_SyncTensorOp : QuidditchSnitch_Op<"sync_tensor",
  [AllTypesMatch<["result", "input"]>, Pure,
    DeclareOpInterfaceMethods<BufferizableOpInterface,
     ["bufferizesToMemoryRead", "bufferizesToMemoryWrite", "getAliasingValues",
      "bufferize", "mustBufferizeInPlace"]>]> {

  let description = [{
    Performs synchronization of a tensor returned by a `tensor.microkernel`
    operation.
    The resulting tensor is guaranteed to consist of the results of any
    operations performed by the `tensor.microkernel` operation.
  }];

  let arguments = (ins
    AnyRankedTensor:$input
  );

  let results = (outs
    AnyRankedTensor:$result
  );

  let assemblyFormat = [{
    $input `:` type($result) attr-dict
  }];
}

def QuidditchSnitch_MemRefMicrokernelOp
  : QuidditchSnitch_Op<"memref.microkernel", [IsolatedFromAbove, SingleBlock,
      NoTerminator, QuidditchSnitch_ComputeCoreSpecializationOpInterface]> {

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

    void replaceWithNoop(mlir::RewriterBase& rewriter);
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

def QuidditchSnitch_MicrokernelFenceOp : QuidditchSnitch_Op<"microkernel_fence",
  [QuidditchSnitch_ComputeCoreSpecializationOpInterface]> {

  let description = [{
    Execution of this operation guarantees that the side-effects of all
    previous microkernel invocations are visible as soon as this operation
    returns.
  }];

  let assemblyFormat = [{
    attr-dict
  }];

  let extraClassDeclaration = [{
    bool needsSynchronization() {
      return true;
    }

    void replaceWithNoop(mlir::RewriterBase& rewriter);
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

def QuidditchSnitch_BarrierOp : QuidditchSnitch_Op<"barrier"> {
  let assemblyFormat = [{
    attr-dict
  }];
}

def QuidditchSnitch_ComputeCoreIndexOp
  : QuidditchSnitch_Op<"compute_core_index", [Pure,
      QuidditchSnitch_ComputeCoreSpecializationOpInterface]> {

  let description = [{
    Returns the index of the compute core within a given cluster.
    This is guaranteed to return a number between 0 and exclusive
    `compute_cores` where `compute_cores` is an `IntegerAttr` in
    the surrounding `ExecutableTargetAttr`.
  }];

  let results = (outs Index:$result);

  let assemblyFormat = [{
    attr-dict
  }];

  let extraClassDeclaration = [{
    void replaceWithNoop(mlir::RewriterBase& rewriter);
  }];
}

def QuidditchSnitch_PipelineOp : QuidditchSnitch_Op<"pipeline",
  [AllTypesMatch<["init_args", "results"]>,
   RecursivelySpeculatable, RecursiveMemoryEffects,
   SingleBlockImplicitTerminator<"quidditch::Snitch::PipelineYieldOp">,
   InferTypeOpAdaptor,
   DeclareOpInterfaceMethods<LoopLikeOpInterface,
    ["moveOutOfLoop"]>,
   DeclareOpInterfaceMethods<RegionBranchOpInterface,
    ["getEntrySuccessorOperands"]>,
   DeclareOpInterfaceMethods<BufferizableOpInterface,
    ["bufferizesToMemoryRead", "bufferizesToMemoryWrite", "getAliasingValues",
     "bufferize", "getBufferType", "isWritable", "getAliasingOpOperands",
     "mustBufferizeInPlace"]>
]> {

  let description = [{
    Op representing a loop consisting of different pipelined stages.
    Every stage in the pipeline is a region containing at least one block
    argument of type `index` which is the induction variable.
    The entry region may additionally have input tensors initialized by
    `init_args` if not yet bufferized.
    Stages are able to explicitly transfer data from one stage to another
    using `pipeline_yield` which are then passed onto the block arguments of
    the next stage following the induction variable.

    No guarantee is given regarding the order of side effects within a stage
    except:
    * For a given IV, `Stage[j]` is executed after `Stage[j-1]`.
    * For a given stage, IV `i` is executed after IV `i-1`.

    Note: Resource allocations performed within a stage may be multiplied by a
    lowering to support concurrently running stages.
  }];

  let arguments = (ins
    Index:$lower_bound,
    Index:$upper_bound,
    Index:$step,
    Variadic<AnyRankedTensor>:$init_args
  );

  let results = (outs
    Variadic<AnyRankedTensor>:$results
  );

  let regions = (region VariadicRegion<SizedRegion<1>>:$stages);

  let assemblyFormat = [{
    $lower_bound `to` $upper_bound `step` $step (`inits` `(`$init_args^ `)`
    `->` type($results) )? $stages attr-dict-with-keyword
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    mlir::BlockArgument getTiedEntryIterArg(mlir::OpOperand& operand);

    mlir::BlockArgument getTiedEntryIterArg(mlir::OpResult operand);

    mlir::OpResult getTiedResult(mlir::OpOperand& operand);

    mlir::OpOperand& getTiedYielded(mlir::OpResult result);

    mlir::OpOperand& getTiedYielded(mlir::BlockArgument argument);

    mlir::OpOperand& getTiedInit(mlir::BlockArgument argument);
  }];
}

def QuidditchSnitch_PipelineYieldOp : QuidditchSnitch_Op<"pipeline_yield",
  [Pure, HasParent<"PipelineOp">, Terminator, ReturnLike,
   DeclareOpInterfaceMethods<BufferizableOpInterface,
   ["bufferizesToMemoryRead", "bufferizesToMemoryWrite", "getAliasingValues",
    "bufferize", "mustBufferizeInPlace"]>
]> {
  let arguments = (ins
    Variadic<AnyType>:$results
  );

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, /*results=*/mlir::ValueRange());
    }]>
  ];

  let assemblyFormat = [{
    ($results `:` type($results)^)? attr-dict
  }];

  let extraClassDeclaration = [{
      mlir::BlockArgument getTiedBlockArgument(mlir::OpOperand& operand);
  }];
}

#endif
