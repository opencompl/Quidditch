#include "TilingScheme.h"

// using namespace quidditch;
using namespace mlir;
using namespace mlir::iree_compiler;

// Tiling Scheme Functions defined below
namespace quidditch {

TileInfoTbl *fillTileInfoTable(TileInfoTbl *tbl, const std::string &filePath,
                               std::string &errs) {
  TileInfoTbl *result = tbl;
  // try to open file
  std::ifstream ifs(filePath);
  if (!ifs.is_open()) {
    std::stringstream ss;
    ss << "\nTiling Scheme File does not exist or cannot be opened.\n"
       << "Troublesome file path is " << filePath << "\n";
    errs = ss.str();
    return 0;
  }
  // try to read file
  std::stringstream ss;
  ss << ifs.rdbuf();
  if (ss.str().length() == 0) {
    errs = "\nTiling Scheme file cannot have content length of 0\n";
    ifs.close();
    return 0;
  }
  // try to parse list of schemes
  if (!parseTilingSchemes(tbl, StringRef(ss.str()), errs)) {
    result = 0;
  }
  ifs.close();
  return result;
}

bool parseTilingSchemes(TileInfoTbl *tbl, llvm::StringRef fileContent,
                        std::string &errs) {
  // try to parse
  llvm::Expected<llvm::json::Value> maybeParsed =
      llvm::json::parse(fileContent);
  if (!maybeParsed) {
    std::stringstream ss;
    ss << "\nError when parsing JSON file contents: "
       << llvm::toString(maybeParsed.takeError()) << "\n";
    errs = ss.str();
    return false;
  }
  // try to get the top level json object
  if (!maybeParsed->getAsObject()) {
    errs = "\nError: top-level value is not a JSON object\n";
    return false;
  }
  llvm::json::Object *O = maybeParsed->getAsObject();
  // make sure object has at least one field
  if (O->empty()) {
    errs = "\nError: top-level JSON object is empty\n";
    return false;
  }
  // try to parse each tiling scheme from function name key
  std::stringstream ss;
  for (const auto &func : *O) {
    struct TilingScheme ts = parseTilingScheme(func.getSecond(), errs);
    if (!ts.valid) {
      return false;
    } else {
      //  only insert the tiling scheme if an entry does not exist already
      // (NO OVERRIDING KEY-VALUE pairs in the table!!)
      auto search = tbl->find(func.getFirst().str());
      if (search == tbl->end()) {
        tbl->insert(std::pair(func.getFirst().str(), ts));
        ss << func.getFirst().str() << ":\n";
        ss << ts;
      }
    }
  }
  errs = ss.str();
  return true;
}

struct TilingScheme parseTilingScheme(llvm::json::Value v, std::string &errs) {

  struct TilingScheme ts;
  auto O = v.getAsObject();
  if (!O) {
    errs = "RHS of key_value pair is not a JSON object!";
    return ts;
  }
  bool read_tile_sizes = parseListOfListOfInts(O, "tile-sizes", ts.tiles, errs);
  bool read_loop_order = parseListOfListOfInts(O, "loop-order", ts.order, errs);
  bool read_dual_buffer = parseBool(O, "dual-buffer", ts.dualBuffer, errs);
  ts.valid = read_tile_sizes && read_loop_order && read_dual_buffer;
  return ts;
}

// TODO: call parseListOfInts inside parseListOfListOfInts
bool parseListOfListOfInts(llvm::json::Object *obj, std::string listName,
                           std::vector<std::vector<int>> &out,
                           std::string &errs) {
  llvm::json::Value *bnds = obj->get(StringRef(listName));
  if (!bnds) {
    std::stringstream ss;
    ss << "\nError: field labeled '" << listName << "' does not exist \n ";
    errs = ss.str();
    return false;
  }

  if (!bnds->getAsArray()) { // getAsArray returns a (const json::Array *)
    std::stringstream ss;
    ss << "\nError: field labeled '" << listName << "' is not a JSON array \n ";
    errs = ss.str();
    return false;
  }
  llvm::json::Path::Root Root("Try-to-parse-integer");
  for (const auto &Item :
       *(bnds->getAsArray())) { // loop over a json::Array type
    if (!Item.getAsArray()) {
      std::stringstream ss;
      ss << "\nError: elt of '" << listName << "' is not also a JSON array \n ";
      errs = ss.str();
      return false;
    }
    std::vector<int> sublist;
    int bound;
    for (const auto &elt :
         *(Item.getAsArray())) { // loop over a json::Array type
      if (!fromJSON(elt, bound, Root)) {
        std::stringstream ss;
        ss << llvm::toString(Root.getError()) << "\n";
        errs = ss.str();
        return false;
      }
      sublist.push_back(bound);
    }
    out.push_back(sublist);
  }
  return true;
}

bool parseListOfInts(llvm::json::Object *obj, std::string listName,
                     std::vector<int> &out, std::string &errs) {
  llvm::json::Value *bnds = obj->get(StringRef(listName));
  if (!bnds) {
    std::stringstream ss;
    ss << "\nError: field labeled '" << listName << "' does not exist \n ";
    errs = ss.str();
    return false;
  }
  if (!bnds->getAsArray()) { // getAsArray returns a (const json::Array *)
    std::stringstream ss;
    ss << "\nError: field labeled '" << listName << "' is not a JSON array \n ";
    errs = ss.str();
    return false;
  }
  llvm::json::Path::Root Root("Try-to-parse-integer");
  int theNumber;
  for (const auto &elt :
       *(bnds->getAsArray())) { // loop over a json::Array type
    if (!fromJSON(elt, theNumber, Root)) {
      std::stringstream ss;
      ss << llvm::toString(Root.getError()) << "\n";
      errs = ss.str();
      return false;
    }
    out.push_back(theNumber);
  }
  return true;
}

bool parseBool(llvm::json::Object *obj, std::string boolName, bool &out,
               std::string &errs) {
  llvm::json::Value *theBool = obj->get(StringRef(boolName));
  if (!theBool) {
    std::stringstream ss;
    ss << "\nError: field labeled '" << boolName << "' does not exist \n ";
    errs = ss.str();
    return false;
  }
  if (!theBool->getAsBoolean()) { // getAsBoolean returns an optional boolean
    std::stringstream ss;
    ss << "\nError: field labeled '" << boolName << "' is not a boolean \n ";
    errs = ss.str();
    return false;
  }
  out = *(theBool->getAsBoolean());
  return true;
}

bool TilingScheme::getTiles_flat(llvm::SmallVector<int64_t> &out) {
  if (out.size() != tiles.size()) {
    return false;
  } else {
    for (size_t i = 0; i < tiles.size(); i++) {
      out[i] = (int64_t)tiles[i][0];
    }
  }
  return true;
}

bool TilingScheme::getOrder_flat(llvm::SmallVector<int64_t> &out) {
  if (out.size() != order.size()) {
    return false;
  } else {
    for (size_t i = 0; i < order.size(); i++) {
      out[i] = (int64_t)order[i][0];
    }
  }

  return true;
}

std::string TilingScheme::str() {
  std::stringstream ts_ss;
  ts_ss << *this;
  return ts_ss.str();
}

std::stringstream &operator<<(std::stringstream &ss,
                              const struct TilingScheme &ts) {
  ss << "tiling scheme: {\nbounds: [ ";
  for (const auto &sublist : ts.tiles) {
    ss << "[ ";
    for (const auto &tile : sublist) {
      ss << " " << tile << " ";
    }
    ss << "] ";
  }
  ss << "]\n";
  ss << "order: [ ";
  for (const auto &sublist : ts.order) {
    ss << "[ ";
    for (const auto &pos : sublist) {
      ss << " " << pos << " ";
    }
    ss << "] ";
  }
  ss << "]\n";
  ss << "dual buffer: " << ts.dualBuffer << "\n";
  ss << "]\n}";
  return ss;
}

} // namespace quidditch
