// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#pragma once

#include <string>

#include "compiler/src/iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Support/LogicalResult.h"

namespace Quidditch {

// Defines an `iree_hal_executable_library_v0_t` and builds the runtime metadata
// structures and query functions.
//
// See Quidditch/executable/executable_library.h for more information.
//
// Usage:
//  LibraryBuilder builder(&module);
//  builder.addExport(
//     "hello", "source.mlir", 123, "test tag", DispatchAttrs{}, &helloFunc);
//  ...
//  auto *queryFunc = builder.build("_query_library_foo");
//  // call queryFunc, export it, etc
class LibraryBuilder {
public:
  // Builder mode setting.
  enum class Mode : uint32_t {
    NONE = 0u,
    // Include entry point names and tags.
    // If not specified then the reflection strings will be excluded to reduce
    // binary size.
    INCLUDE_REFLECTION_ATTRS = 1 << 0u,
  };

  // iree_hal_executable_library_version_t
  enum class Version : uint32_t {
    // NOTE: until we hit v1 the versioning scheme here is not set in stone.
    // We may want to make this major release number, date codes (0x20220307),
    // or some semantic versioning we track in whatever spec we end up having.
    V_0_3 = 0x0000'0003u, // v0.3 - ~2022-08-08
    V_0_4 = 0x0000'0004u, // v0.4 - ~2024-03-12

    // Pinned to the latest version.
    // Requires that the runtime be compiled with the same version.
    LATEST = V_0_4,
  };

  // iree_hal_executable_library_features_t
  enum class Features : uint32_t {
    // IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE
    NONE = 0u,
  };

  // iree_hal_executable_library_sanitizer_kind_t
  enum class SanitizerKind : uint32_t {
    // IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE
    NONE = 0u,
    // IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_ADDRESS
    ADDRESS = 1u,
    // IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_MEMORY
    MEMORY = 2u,
    // IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_THREAD
    THREAD = 3u,
    // IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_UNDEFINED
    UNDEFINED = 4u,
  };

  // IREE_HAL_WORKGROUP_LOCAL_MEMORY_PAGE_SIZE
  static const int64_t kWorkgroupLocalMemoryPageSize = 4096;

  // iree_hal_executable_dispatch_attrs_v0_t
  struct DispatchAttrs {
    // Required workgroup local memory size, in bytes.
    int64_t localMemorySize = 0;

    // True if all values are default and the attributes may be omitted.
    constexpr bool isDefault() const { return localMemorySize == 0; }
  };

  // iree_hal_executable_source_location_v0_t
  struct SourceLocation {
    std::string stage;
    std::string path;
    uint32_t line;
  };

  LibraryBuilder(llvm::Module *module, Mode mode,
                 Version version = Version::LATEST)
      : module(module), mode(mode), version(version) {}

  // Adds a new required feature flag bit. The runtime must support the feature
  // for the library to be usable.
  void addRequiredFeature(Features feature) {
    features = static_cast<Features>(static_cast<uint32_t>(features) |
                                     static_cast<uint32_t>(feature));
  }

  // Sets the LLVM sanitizer the executable is built to be used with. The
  // runtime must also be compiled with the same sanitizer enabled.
  void setSanitizerKind(SanitizerKind sanitizerKind) {
    this->sanitizerKind = sanitizerKind;
  }

  // Defines a new runtime import function.
  // The declared ordinal of the import matches the order they are declared.
  void addImport(llvm::StringRef name, bool weak) {
    imports.push_back({name.str(), weak});
  }

  // Defines a new entry point on the library implemented by |func|.
  // |name| will be used as the library export
  // |sourceFile| and |sourceLoc| are optional source information
  // |tag| is an optional attachment
  void addExport(llvm::StringRef name, SourceLocation sourceLocation,
                 llvm::SmallVector<SourceLocation> stageLocations,
                 llvm::StringRef tag, DispatchAttrs attrs,
                 llvm::Function *compute_func, llvm::Function *dmaFunc) {
    exports.push_back({name.str(), std::move(sourceLocation),
                       std::move(stageLocations), tag.str(), attrs,
                       compute_func, dmaFunc});
  }

  // Defines a source file embedded in the library.
  void addSourceFile(llvm::StringRef path, llvm::SmallVector<char> contents) {
    sourceFiles.push_back({path.str(), std::move(contents)});
  }

  // Builds a `iree_hal_executable_library_query_fn_t` with the given
  // |queryFuncName| that will return the current library metadata.
  //
  // The returned function will be inserted into the module with internal
  // linkage. Callers may change the linkage based on their needs (such as
  // exporting if producing a dynamic library or making dso_local for static
  // libraries that reference the function via extern in another compilation
  // unit, etc).
  llvm::Function *build(llvm::StringRef queryFuncName);

private:
  // Builds and returns an iree_hal_executable_library_v0_t global constant.
  llvm::Constant *buildLibraryV0(std::string libraryName);
  llvm::Constant *buildLibraryV0ImportTable(std::string libraryName);
  llvm::Constant *buildLibraryV0ExportTable(std::string libraryName);
  llvm::Constant *buildLibraryV0ConstantTable(std::string libraryName);
  llvm::Constant *buildLibraryV0SourceTable(std::string libraryName);

  llvm::Module *module = nullptr;
  Mode mode = Mode::INCLUDE_REFLECTION_ATTRS;
  Version version = Version::LATEST;
  Features features = Features::NONE;
  SanitizerKind sanitizerKind = SanitizerKind::NONE;

  struct Import {
    std::string symbol_name;
    bool weak = false;
  };
  llvm::SmallVector<Import> imports;

  struct Dispatch {
    std::string name;
    SourceLocation sourceLocation;
    llvm::SmallVector<SourceLocation> stageLocations;
    std::string tag;
    DispatchAttrs attrs;
    llvm::Function *compute_func;
    llvm::Function *dma_func;
  };
  std::vector<Dispatch> exports;

  size_t constantCount = 0;

  struct SourceFile {
    std::string path;
    llvm::SmallVector<char> contents;
  };
  llvm::SmallVector<SourceFile> sourceFiles;
};

} // namespace Quidditch