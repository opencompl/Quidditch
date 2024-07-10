#include "QuidditchSnitchAttrs.h"

using namespace mlir;
using namespace quidditch::Snitch;

//===----------------------------------------------------------------------===//
// LoweringConfigAttr::LoweringConfigAttrInterface
//===----------------------------------------------------------------------===//

SmallVector<int64_t> LoweringConfigAttr::getWorkgroupTileSizes() const {
  return llvm::to_vector(getWorkgroupTiles());
}
