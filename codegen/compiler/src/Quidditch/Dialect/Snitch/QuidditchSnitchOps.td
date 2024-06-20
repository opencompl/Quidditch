#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHOPS
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHOPS

include "Quidditch/Dialect/Snitch/QuidditchSnitchDialect.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class QuidditchSnitch_Op<string mnemonic, list<Trait> traits = []> :
  Op<QuidditchSnitch_Dialect, mnemonic, traits>;

// TODO: We likely want this in DPS and operating on tensors depending on the
// phase of our compiler where we introduce these.
def QuidditchSnitch_XDSLKernelOp : QuidditchSnitch_Op<"xdsl_kernel",
  [IsolatedFromAbove, SingleBlock, NoTerminator]> {

  let description = [{

  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    `` `(` $inputs `)` `:` type($inputs) $body attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{

    mlir::Block* createEntryBlock();
  }];
}

#endif
