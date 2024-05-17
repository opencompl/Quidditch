# Quidditch

## Building

Only linux is currently supported as a build environment

Building Quidditch requires the following tools to be installed:
* CMake 3.21 or newer
* A C++17 compiler
* Cargo
* Python 3
* Ninja (ideally)
* the Quidditch toolchain
* the Pulp toolchain

`docker` is required to install the Quidditch toolchain using:
```shell
docker run --rm ghcr.io/opencompl/Quidditch/toolchain:main tar -cC /opt/quidditch-toolchain .\
 | tar -xC $INSTALL_DIR/quidditch-toolchain
```
See [the toolchain directory for more details](runtime/toolchain/README.md).

To get the pulp toolchain, you can run:
```shell
mkdir $INSTALL_DIR/pulp-toolchain
wget -qO- https://github.com/pulp-platform/llvm-project/releases/download/0.12.0/riscv32-pulp-llvm-ubuntu2004-0.12.0.tar.gz \
| tar --strip-components=1 -xzv -C $INSTALL_DIR/pulp-toolchain
```
or similar if your system is binary compatible with Ubuntu 22.04.

Afterward, you can perform a mega build using:
```shell
git clone --recursive https://github.com/opencompl/quidditch
cd quidditch

python -m venv venv
source ./venv/bin/activate

mkdir build && cd build
cmake .. -GNinja \
  # Optional but highly recommended 
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_COMPILER=clang++ \
  # Optional for improved caching, requires ccache to be installed.
  # -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  # -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DPULP_CLANG_PATH=$INSTALL_DIR/pulp-toolchain/bin/clang \
  -DQUIDDITCH_TOOLCHAIN_FILE=$INSTALL_DIR/quidditch-toolchain/ToolchainFile.cmake
  
# Build everything:
cmake --build .
# Test everything. Requires everything to be built.
cmake --build . --target test
```

This will execute both `codegen` and `runtime` builds as external builds of the top-level cmake.
This is useful for quickly building and testing but may be undesirable for development.
See the respective READMEs in [codegen](codegen/README.md) and [runtime](runtime/README.md) for how to configure them
independently.
