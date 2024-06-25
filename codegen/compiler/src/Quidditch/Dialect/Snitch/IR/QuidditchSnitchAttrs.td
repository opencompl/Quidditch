#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHATTRS
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHATTRS

include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.td"
include "mlir/IR/AttrTypeBase.td"

class QuidditchSnitch_Attr<string name, list<Trait> traits = []> :
  AttrDef<QuidditchSnitch_Dialect, name, traits>;

def QuidditchSnitch_L1EncodingAttr : QuidditchSnitch_Attr<"L1Encoding"> {
  let mnemonic = "l1_encoding";

  let description = [{
    Attribute used as memory space on a memref to denote it being in L1 memory.
  }];
}

#endif
