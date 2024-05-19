#  Licensed under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os

import lit.formats

config.name = "Quidditch"
config.suffixes = [".mlir"]
config.test_format = lit.formats.ShTest()

config.excludes = ["Inputs", "CMakeLists.txt", "lit.cfg.py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.binary_dir, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%xdsl-opt", config.xdsl_opt))
config.substitutions.append(("%quidditch-toolchain-root", config.quidditch_toolchain_root))

# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment["FILECHECK_OPTS"] = "-enable-var-scope --allow-unused-prefixes=false"
