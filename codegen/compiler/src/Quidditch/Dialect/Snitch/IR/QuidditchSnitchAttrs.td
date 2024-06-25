#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHATTRS
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHATTRS

include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.td"
include "mlir/IR/AttrTypeBase.td"

class QuidditchSnitch_Attr<string name, list<Trait> traits = []> :
  AttrDef<QuidditchSnitch_Dialect, name, traits>;

def QuidditchSnitch_L1EncodingAttr : QuidditchSnitch_Attr<"L1Encoding"> {
  let mnemonic = "l1_encoding";

  let description = [{
    Attribute used as both encoding on a tensor and memory space on a memref
    to denote either of these being in L1 memory.
  }];
}

#endif
