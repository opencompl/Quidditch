# Quidditch Toolchain

This repo contains the `Dockerfile`, cmake build files and transient patches used to build the toolchain for the
quidditch runtime.

The goals of the toolchain are:

* Minimal: The image is currently "only" 800MB.
* Easy to maintain: Only upstream components are used, no forks.
* Modern: The toolchain uses a full LLVM 18 toolchain with `picolibc` as C standard library.

Why not https://github.com/pulp-platform/llvm-project/?
The pulp toolchain is based on LLVM 12 and a fork of LLVM, therefore not easily maintainable.
Furthermore, the toolchain has generally not been used for larger applications requiring a more complete `libc` and
therefore does not have out-of-the-box support for things like `malloc`.
**Note that this toolchain does not support any of the intrinsics or mnemonics that the pulp toolchain does**

## Installation

### Installing locally

The toolchain is released as an alpine docker image and built as fully static binaries that are capable of running on
any linux distro.

To copy the toolchain run:

```shell
docker run --rm ghcr.io/opencompl/Quidditch/toolchain:main tar -cC /opt/quidditch-toolchain .\
 | tar -xC $INSTALL_DIR/quidditch-toolchain
```

### Using in Docker

Integrating into another docker image can be done
using [`COPY --from`](https://docs.docker.com/reference/dockerfile/#copy---from)

Example:

```dockerfile
COPY --from=ghcr.io/opencompl/Quidditch/toolchain:main /opt/quidditch-toolchain $INSTALL_DIR/quidditch-toolchain
```

## Using in CMake

The toolchain ships with a so-called toolchain file that is used to tell cmake about the cross compilation
environment.
The toolchain file is located at `<root-dir>/ToolchainFile.cmake`.
When building with cmake, add `-DCMAKE_TOOLCHAIN_FILE=<root-dir>/ToolchainFile.cmake` to your `cmake` command line to
start using the toolchain.
