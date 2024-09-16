# Quidditch

## Building

Only linux is currently supported as a build environment

Building Quidditch requires the following tools to be installed:

* CMake 3.21 or newer
* A C++17 compiler
* lld
* Python 3.11
* Ninja (ideally)
* Docker to install the Quidditch toolchain. See [the toolchain directory for more details](runtime/toolchain/README.md)

The repository is self-contained, contains the definition of the docker container, and contains no magic.
The compilation is driven by CMake, and uses Docker.
Python 3.11 is used for the runtime and the tracing script, as well as PyTorch etc., it's not strictly required for compilation except for the lit tests.
Python 3.12 is not yet supported by PyTorch.
The first time to compile can take up to one hour, to build MLIR, IREE, and the rest of the project.

The docker is mostly used as a convenient way to install the Snitch toolchain, runtime, verilator, etc.
It's not used for the compiler, rather to make the Snitch tools available.
The dockerfile can be found in `runtime/toolchain/Dockerfile`.
When updating the snitch runtime submodule, the docker image needs to be updated with the new commit hash to make sure the hardware and software are in sync.

The easiest way to iterate on the docker image is to develop on it locally, and then pushing the changes to GitHub, which will detect that the toolchain directory is changed, and will release a new image automatically.

There are two CMake projects in Quidditch, one is the IREE-based compiler (`codegen/`) which compiles to an executable to run your local machine, and the second is the runtime (`runtime/`), which uses the toolchain from the docker image, and compiles code for Snitch.

To build, run the following commands:

```shell
git clone --recursive https://github.com/opencompl/quidditch
cd quidditch

mkdir toolchain
docker run --rm ghcr.io/opencompl/quidditch/toolchain:main tar -cC /opt/quidditch-toolchain .\
 | tar -xC ./toolchain

python -m venv venv
source ./venv/bin/activate

mkdir build && cd build
cmake .. -GNinja \
  # Optional but highly recommended \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  # Optional for improved caching, requires ccache to be installed. \
  # -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  # -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DQUIDDITCH_TOOLCHAIN_FILE=../toolchain/ToolchainFile.cmake

# Build and Test everything.
cmake --build . --target test
```

> [!NOTE]
> The Snitch runtime is built as part of the Quidditch compilation process, which lets us navigate the sources, we have ported the Snitch runtime to CMake to integrate it with the rest of the list.

This will execute both `codegen` and `runtime` builds as external builds of the top-level cmake.
This is useful for quickly building and testing but may be undesirable for development.
See the respective READMEs in [codegen](codegen/README.md) and [runtime](runtime/README.md) for how to configure them
independently.

> [!NOTE]
> It should be possible to decouple the compilation infrastructure from the runtime, letting users compile the neural networks on their local machines, to then execute it on Linux.
> Everything to do with Verilator is only supported on Linux, so that part cannot be made platform-independent, but the `runtime/toolchain/Dockerfile` can be massaged to exclude everything Verilator-related, to build MLIR+IREE+Quidditch, and be run as a script to produce a compiler that emits everything necessary to just run it on a Linux machine.

## Compiling Neural Networks

Any information you can find on how to use `iree-compile` is applicable, with the only major difference bing the `--iree-hal-target-backends=llvm-cpu`, which also supports our `quidditch` value for the flag, which leverages our flow.
The inputs to `iree-compile` are all `.mlir` files, so you need to somehow get the file, [here's](https://iree.dev/guides/ml-frameworks/) the guide on how to get one.
The PyTorch and ONNX imports are the flakiest of the imports in our experience, although they can be made to work.

The PyTorch flow uses [iree-turbine](https://github.com/iree-org/iree-turbine), which is included in the virtualenv of the Docker container.
We use it to compile nsnet2 (`runtime/samples/nsnet2/NsNet2.py`).
The MLIR file extracted this way will contain functions that are named the same as the Python model functions.
If the model doesn't contain the weights, we recommend seeding the tensors with random data in the Python file for reproducible behavior.

### Adding a new Neural Network

Neural networks live in `samples/`

To add a new model, replicate the structure of the existing models, and add the subdirectory name to `samples/CMakeLists.txt`.

A convenient way to run it is to add a new `test_executable` line at the end of `tests/CMakeLists.txt`
Then run:

``` shell
# Build everything
ninja
# To run a specific test
ctest -R mynewmodel
```

The last command takes a regex so can run multiple tests, and can be parallelized with `-j`

### MLIR to Executable

The simplest example is `runtime/samples/vec_multiply/`.

The `quiddtch_module` macro in the `CMakeLists.txt` turns the MLIR file into an executable.
`ASSERT_XDSL` is a flag that will error out if xDSL micro-kernel compilation fails.
This macro creates a library called `mynewmodel` (removing the `.mlir` suffix).
It will also create two new header files, `mynewmodel.h`, and `mynewmodel_module.h` (the device and host code).
These need to be imported in the `main.c` file for your model.
The `add_executable` macro is used to specify the name of the executable that `main.c` will be compiled to, let's say `mynewmodelexecutable`.

`run_model.h` in the `utils/` folder specifies the information needed to run the model, which needs to be filled in the `main.c`.
NB: the module of the mlir module in the input file MUST have a name, which must be specified in the `main.c`.

In summary:

 0. Get your MLIR file from your model
 1. Copy the folder of one of the existing models, and adjust the files to your module
 2. Refer to the `utils/run_model.h` file for documentation, and other models for examples
 3. Go to the build directory, and run `ninja`, which recompiles everything, including your new changes
 4. The cmake build system mirrors your source directory, so if you added `runtime/samples/mynewmodel/`, the build location will be `build/runtime/samples/mynewmodel`.
 5. Add your new executable to `tests/CMakeLists.txt`

> NB. you can override the compilation with xDSL by adding `LLVM` to the `quidditch_module` macro, which overrides the XDSL pipeline entirely and just computes everything with stock IREE `llvm-cpu` target.

### Calling the Neural Network

Once the above steps are executed, you can run your neural network either by directly executing `build/runtime/samples/mynewmodel/mynewmodelexecutable`, or by running `ctest -R mynewmodelexecutable`.
The second way to do it will generate traces.

## Quidditch Compiler Structure

All source code is in `codegen/compiler/src/Quirritch`.
Within it there are three top level directories:

1. Conversion - containing dialect conversions, in our case Snitch to LLVM and the xDSL linalg to RISC-V lowering
2. Dialect  - where all the dialects (only one right now) are
3. Target  - defines all the passes that aren't a part of the dialect, but rather target optimisations

The biggest and most important file is `QuidditchTarget.cpp`, with the information necessary for adding a new backend to IREE.

### Target

`QuidditchTarget.cpp`

The QuidditchTargetBackend class subclasses IREE::HAL::TargetBackend.

`buildConfigurationPassPipeline`, `buildTranslationPassPipeline` and `buildLinkingPassPipeline` build the pass pipeline:

1. `buildConfigurationPassPipeline` adds the annotations used by IREE, such as tile sizes (the padding which happens in a later step is a multiple of the tile sizes)
2. `buildTranslationPassPipeline` takes us from `linalg` on `tensor` all the way down to `llvm`
3. `buildLinkingPassPipeline` does some small cleanup like deduplication

The tile sizes are currently manually specified in the C++ source code, and attached to the IR to be used by a downstream pass, but they could also specified as part of the source (like in `big_matmul.mlir`), or with a `transform` dialect script in the `.mlir` source file.
The tile sizes are the size that all cores will be working on, not individual cores, and must fit into half the scratch memory, due to double buffering.
Importantly, if you choose tile sizes where each tile does not fit into half the scratch memory, then compilation will fail.
_TODO proper guide on how to pick tile sizes._

`buildTranslationPassPipeline` is currently specialised for _MatVec_ and _MatMul_, some more work is needed to run other linalg operations.
The first big sequence of `.addPass` contains almost all the optimisations, and the remainder of the function is lowering.

The `serializeExecutable` function takes MLIR llvm input, and constructs an object file, which contains everything including the xDSL microkernels.

N.B. The heuristic to pick the tile sizes for individual cores, which happens after the L1 tiling mentioned above, lives in `TensorTile.cpp`, and just picks the largest parallel dimension, and divides it by 8, which will be iterated on in the future.

### Dialect

There is currently only one dialect, `quidditch_snitch`, which might be split up in the future.

The parameters fort the loweing config are in `QuidditchSnitchAttrs.td`, in `QuidditchSnitch_LoweringConfigAttr`.

### Conversion

Shouldn't be anything surprising in here.

The Snitch to LLVM covnersion converts operations to LLVM function calls, and the `linalg` to RISC-V just calls to the `xdsl-opt` executable in the virtual environment and stores the result in a `StringAttr`.

The scratch memory is modeled as one big `memref`, the code for which is in `ConvertSnitchToLLVM.cpp`
