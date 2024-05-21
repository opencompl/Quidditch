# Quidditch

## Building

Only linux is currently supported as a build environment

Building Quidditch requires the following tools to be installed:
* CMake 3.21 or newer
* A C++17 compiler
* Python 3
* Ninja (ideally)
* Docker to install the Quidditch toolchain. See [the toolchain directory for more details](runtime/toolchain/README.md)

Afterward, you can perform a mega build using:
```shell
git clone --recursive https://github.com/opencompl/quidditch
cd quidditch

docker run --rm ghcr.io/opencompl/Quidditch/toolchain:main tar -cC /opt/quidditch-toolchain .\
 | tar -xC ./toolchain

python -m venv venv
source ./venv/bin/activate

mkdir build && cd build
cmake .. -GNinja \
  # Optional but highly recommended 
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  # Optional for improved caching, requires ccache to be installed.
  # -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  # -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake

# Build and Test everything.
cmake --build . --target test
```

This will execute both `codegen` and `runtime` builds as external builds of the top-level cmake.
This is useful for quickly building and testing but may be undesirable for development.
See the respective READMEs in [codegen](codegen/README.md) and [runtime](runtime/README.md) for how to configure them
independently.
