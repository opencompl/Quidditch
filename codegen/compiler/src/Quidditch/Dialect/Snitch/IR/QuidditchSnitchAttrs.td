#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHATTRS
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHATTRS

include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class QuidditchSnitch_Attr<string name, list<Trait> traits = []> :
  AttrDef<QuidditchSnitch_Dialect, name, traits>;

def QuidditchSnitch_L1EncodingAttr : QuidditchSnitch_Attr<"L1Encoding"> {
  let mnemonic = "l1_encoding";

  let description = [{
    Attribute used as memory space on a memref to denote it being in L1 memory.
  }];
}

def QuidditchSnitch_LoweringConfigAttr : QuidditchSnitch_Attr<"LoweringConfig",
  [DeclareAttrInterfaceMethods<IREECodegen_LoweringConfigAttrInterface, [
    "getWorkgroupTileSizes",
  ]>]> {
  let mnemonic = "lowering_config";

  let description = [{
    Attribute attached to compute operations to describe parameters used in the
    lowering process (e.g. tile sizes).
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$workgroup_tiles
  );

  let assemblyFormat = [{
    struct(params)
  }];
}

#endif
