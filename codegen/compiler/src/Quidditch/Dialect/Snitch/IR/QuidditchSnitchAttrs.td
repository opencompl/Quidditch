#ifndef QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHATTRS
#define QUIDDITCH_DIALECT_SNITCH_QUIDDITCHSNITCHATTRS

include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchDialect.td"
include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class QuidditchSnitch_Attr<string name, list<Trait> traits = []> :
  AttrDef<QuidditchSnitch_Dialect, name, traits>;

def QuidditchSnitch_CompletedTokenAttr : QuidditchSnitch_Attr<"CompletedToken"> {

  let mnemonic = "completed_token";

  let description = [{
    Attribute representing an instance of a `!quidditch_snitch.dma_token`
    signaling a complete transfer.
  }];
}

def QuidditchSnitch_L1EncodingAttr : QuidditchSnitch_Attr<"L1Encoding"> {
  let mnemonic = "l1_encoding";

  let description = [{
    Attribute used as memory space on a memref to denote it being in L1 memory.
  }];
}

def TileSizeList : OptionalArrayRefParameter<"int64_t"> {
  let parser = [{
    [&]() -> mlir::FailureOr<llvm::SmallVector<int64_t>> {
      if ($_parser.parseLSquare())
        return mlir::failure();

      if (succeeded($_parser.parseOptionalRSquare()))
        return llvm::SmallVector<int64_t>();

      auto result
        = mlir::FieldParser<llvm::SmallVector<int64_t>>::parse($_parser);
      if (mlir::failed(result) || $_parser.parseRSquare())
        return mlir::failure();

      return result;
    }()
  }];

  let printer = [{
    $_printer << "[" << $_self << "]"
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
    TileSizeList:$workgroup_tiles,
    TileSizeList:$l1_tiles,
    DefaultValuedParameter<"bool", "false">:$dual_buffer
  );

  let builders = [
    AttrBuilder<(ins
      CArg<"llvm::ArrayRef<int64_t>", "{}">:$workgroupTiles,
      CArg<"llvm::ArrayRef<int64_t>", "{}">:$l1Tiles,
      CArg<"bool", "false">:$dualBuffer
    )>
  ];

  let skipDefaultBuilders = 1;

  let assemblyFormat = [{
    `<` struct(params) `>`
  }];
}

#endif
