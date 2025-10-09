

#include <stdio.h>
#include <string.h>
#include <fstream> // to open tiling scheme file
#include <sstream>
#include <string>        // for string compare
#include <unordered_map> // to store parsed tiling schemes in a hash table
#include "Quidditch/Dialect/Snitch/IR/QuidditchSnitchAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h" // to parse tiling scheme
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

namespace quidditch {
// define a struct that stores
// tile size and loop interchange information
// for an iree dispatch's root operation
struct TilingScheme {
  bool valid = false;
  std::vector<std::vector<int>> tiles;
  std::vector<std::vector<int>> order;
  bool dualBuffer = false;
  std::string errs = "";
  // member funcs
  TilingScheme() = default;
  std::string str();
  bool getTiles_flat(llvm::SmallVector<int64_t> &out);
  bool getOrder_flat(llvm::SmallVector<int64_t> &out);
  bool getDualBuffer() { return dualBuffer; }
  // overloaded output operator
  friend std::stringstream &operator<<(std::stringstream &ss,
                                       const struct TilingScheme &ts);
};

// define a table that maps
// each iree dispatch function name to its tiling scheme.
typedef std::unordered_map<std::string, struct quidditch::TilingScheme>
    TileInfoTbl;
// function to read a json file and puts its contents in a TileInfo Table
TileInfoTbl *fillTileInfoTable(TileInfoTbl *tbl, const std::string &filePath,
                               std::string &errs);
// json parsing helper functions
struct TilingScheme parseTilingScheme(llvm::json::Value v, std::string &errs);
bool parseTilingSchemes(TileInfoTbl *tbl, llvm::StringRef fileContent,
                        std::string &errs);
bool parseListOfListOfInts(llvm::json::Object *obj, std::string listName,
                           std::vector<std::vector<int>> &out,
                           std::string &errs);
bool parseListOfInts(llvm::json::Object *obj, std::string listName,
                     std::vector<int> &out, std::string &errs);
bool parseBool(llvm::json::Object *obj, std::string listName, bool &out,
               std::string &errs);
} // namespace quidditch
