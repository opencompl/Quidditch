
# Global args.
ARG LLVM_VERSION=18
ARG PICOLIBC_VERSION=1.8.6
ARG SNITCH_CLUSTER_SHA=1f984781957520b3784b9ab17e5eeb34eec763f1
ARG VERILATOR_RELEASE=v4.228
ARG BENDER_RELEASE=0.27.1
ARG INSTALL_DIR=/opt
ARG PULP_TOOLCHAIN_RELEASE=0.12.0
ARG SNITCH_SPIKE_SHA=6c5c9cde499e46c68d59fc12c55a9bdf59e088b2

FROM alpine:3.18 AS toolchain-build

# Args needed by a build stage need to redeclared.
ARG LLVM_VERSION
ARG PICOLIBC_VERSION
ARG INSTALL_DIR

RUN apk update && apk add --no-cache cmake git musl-dev clang clang-dev ccache  \
    python3 ninja py3-pip meson

ENV CCACHE_DIR=/ccache

WORKDIR $INSTALL_DIR

RUN git clone --depth 1  https://github.com/llvm/llvm-project/ -b release/$LLVM_VERSION.x

COPY toolchain/phase1.cmake /root/

RUN mkdir llvm-project/llvm/build && cd llvm-project/llvm/build && \
    cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/quidditch-toolchain \
     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
     -DCMAKE_C_COMPILER_LAUNCHER=ccache  \
     -DLLVM_BUILD_STATIC=ON  \
     -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
     -C /root/phase1.cmake && ninja install

RUN git clone --depth 1 https://github.com/picolibc/picolibc -b $PICOLIBC_VERSION

RUN echo -e "[binaries]\n\
c = '$INSTALL_DIR/quidditch-toolchain/bin/clang'\n\
ar = '$INSTALL_DIR/quidditch-toolchain/bin/llvm-ar'\n\
as = '$INSTALL_DIR/quidditch-toolchain/bin/llvm-as'\n\
ld = '$INSTALL_DIR/quidditch-toolchain/bin/ld.lld'\n\
strip = '$INSTALL_DIR/quidditch-toolchain/bin/llvm-strip'\n\
\n\
[host_machine]\n\
system = 'unknown'\n\
cpu_family = 'riscv'\n\
cpu = 'riscv'\n\
endian = 'little'\n\
" > /cross.txt

RUN mkdir picolibc/build && cd picolibc/build &&  \
    meson ..  \
    -Dincludedir=include \
    -Dlibdir=lib  \
    --cross-file /cross.txt  \
    --buildtype=release  \
    -Dnewlib-nano-malloc=false \
    -Dpicocrt=false  \
    -Dpicolib=false  \
    -Dposix-console=true  \
    -Dprefix=$INSTALL_DIR/quidditch-toolchain  \
    -Dspecsdir=none && ninja install

COPY toolchain/phase2.cmake /root/

RUN cd llvm-project/llvm/build && rm -rf * && \
    cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/quidditch-toolchain \
         -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
         -DCMAKE_C_COMPILER_LAUNCHER=ccache  \
         -DLLVM_BUILD_STATIC=ON  \
         -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
         -C  /root/phase2.cmake && \
        ninja install/strip

# Ubuntu as Rust has issues with alpine.
FROM ubuntu:24.04 AS bender-build

ARG BENDER_RELEASE
ARG INSTALL_DIR

ENV RUSTFLAGS="-C target-feature=+crt-static"

RUN apt-get update && apt-get install -y cargo
RUN cargo install bender --locked --version $BENDER_RELEASE --target x86_64-unknown-linux-gnu
RUN mkdir -p $INSTALL_DIR/quidditch-toolchain/bin/
RUN cp /root/.cargo/bin/bender $INSTALL_DIR/quidditch-toolchain/bin/

FROM alpine:3.18 AS pulp-build

ARG INSTALL_DIR
ARG PULP_TOOLCHAIN_RELEASE

RUN apk update && apk add --no-cache cmake git musl-dev clang clang-dev \
    python3 ninja

RUN git clone --depth 1  https://github.com/pulp-platform/llvm-project -b $PULP_TOOLCHAIN_RELEASE
RUN cd llvm-project/llvm && mkdir build && cd build && \
    cmake .. -GNinja -DLLVM_BUILD_STATIC=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DLLVM_TARGETS_TO_BUILD="RISCV" -DCMAKE_BUILD_TYPE="Release" \
    -DLLVM_DEFAULT_TARGET_TRIPLE="riscv32-unknown-elf" \
    && ninja llvm-mc
RUN mkdir -p $INSTALL_DIR/quidditch-toolchain/bin
RUN cp llvm-project/llvm/build/bin/llvm-mc $INSTALL_DIR/quidditch-toolchain/bin/pulp-as
RUN strip $INSTALL_DIR/quidditch-toolchain/bin/pulp-as

FROM alpine:3.18 AS verilator-build

ARG VERILATOR_RELEASE
ARG INSTALL_DIR

WORKDIR $INSTALL_DIR

RUN apk update && apk add --no-cache py3-pip git autoconf flex-dev bison make musl-dev clang coreutils dtc

ENV PATH=/root:$PATH
COPY --from=bender-build $INSTALL_DIR/quidditch-toolchain/bin/bender /root

# snitch_cluster repo does not yet support Verilator 5, see: https://github.com/pulp-platform/snitch_cluster/pull/76.
ENV VERILATOR_ROOT=$INSTALL_DIR/verilator
RUN git clone --depth 1 https://github.com/verilator/verilator -b $VERILATOR_RELEASE
RUN cd verilator && autoconf && CC=clang CXX=clang++ ./configure && make -j$(nproc)

FROM alpine:3.18 AS snitch-verilator-build-base

ARG SNITCH_CLUSTER_SHA
ARG INSTALL_DIR

COPY --from=bender-build $INSTALL_DIR/quidditch-toolchain/bin/bender /root
COPY --from=verilator-build $INSTALL_DIR/verilator $INSTALL_DIR/verilator

WORKDIR $INSTALL_DIR

ENV PATH=/root:$PATH
ENV VERILATOR_ROOT=$INSTALL_DIR/verilator
ENV VLT=$VERILATOR_ROOT/bin/verilator

RUN apk update && apk add --no-cache py3-pip git make musl-dev clang compiler-rt coreutils patch perl dtc llvm

COPY requirements.txt /root/
RUN pip install --break-system-packages -r /root/requirements.txt
RUN git clone https://github.com/pulp-platform/snitch_cluster && \
    cd snitch_cluster && git checkout $SNITCH_CLUSTER_SHA && git submodule update --init --recursive
COPY toolchain/multi-thread-verilator.diff /root/multi-thread-verilator.diff
WORKDIR $INSTALL_DIR/snitch_cluster
RUN git apply /root/multi-thread-verilator.diff
RUN bender vendor init

FROM snitch-verilator-build-base AS snitch-verilator-build

RUN make -C target/snitch_cluster \
    LDFLAGS="-static" \
    VLT_USE_LLVM=ON \
    bin/snitch_cluster.vlt -j$(nproc)
RUN mkdir -p $INSTALL_DIR/quidditch-toolchain/bin/
RUN cp ./target/snitch_cluster/bin/snitch_cluster.vlt $INSTALL_DIR/quidditch-toolchain/bin/snitch_cluster_nofdiv.vlt
RUN strip $INSTALL_DIR/quidditch-toolchain/bin/snitch_cluster_nofdiv.vlt

FROM snitch-verilator-build-base AS snitch-fdiv-verilator-build

RUN make -C target/snitch_cluster \
    CLANG_CXXFLAGS="-fprofile-instr-generate" \
    CLANG_LDFLAGS="-static -fprofile-instr-generate" \
    CFG_OVERRIDE=cfg/fdiv.hjson VLT_USE_LLVM=ON bin/snitch_cluster.vlt -j$(nproc)
RUN mkdir -p $INSTALL_DIR/quidditch-toolchain/bin/
RUN cp ./target/snitch_cluster/bin/snitch_cluster.vlt $INSTALL_DIR/quidditch-toolchain/bin/

FROM ghcr.io/pulp-platform/snitch_cluster:main AS pgo-test-build

ARG SNITCH_CLUSTER_SHA
ARG INSTALL_DIR

COPY --from=bender-build $INSTALL_DIR/quidditch-toolchain/bin/bender /root
ENV PATH=/root:$PATH

WORKDIR /
RUN git clone https://github.com/pulp-platform/snitch_cluster
WORKDIR /snitch_cluster
RUN git checkout $SNITCH_CLUSTER_SHA && git submodule update --init --recursive
RUN bender vendor init && make -C target/snitch_cluster sw -j$(nproc)

WORKDIR /snitch_cluster/target/snitch_cluster
RUN mkdir bin
COPY --from=snitch-fdiv-verilator-build $INSTALL_DIR/quidditch-toolchain/bin/snitch_cluster.vlt ./bin/
RUN LLVM_PROFILE_FILE="/root/pgo/code_%m.profraw" ./util/run.py sw/run.yaml --simulator verilator -j

FROM snitch-fdiv-verilator-build AS snitch-fdiv-verilator-build-pgo

COPY --from=pgo-test-build /root/pgo /root/

RUN llvm-profdata merge -output=/root/code.profdata /root/code_*.profraw

RUN make -C target/snitch_cluster clean-vlt && \
    make -C target/snitch_cluster \
    CLANG_CXXFLAGS="-fprofile-instr-use=/root/code.profdata" \
    CLANG_LDFLAGS="-static" \
    CFG_OVERRIDE=cfg/fdiv.hjson VLT_USE_LLVM=ON  bin/snitch_cluster.vlt -j$(nproc)

RUN mkdir -p $INSTALL_DIR/quidditch-toolchain/bin/
RUN cp ./target/snitch_cluster/bin/snitch_cluster.vlt $INSTALL_DIR/quidditch-toolchain/bin/
RUN strip $INSTALL_DIR/quidditch-toolchain/bin/snitch_cluster.vlt

FROM alpine:3.18 AS spike-build

ARG INSTALL_DIR
ARG SNITCH_SPIKE_SHA

ENV CFLAGS=-static
ENV CXXFLAGS=-static
ENV LDFLAGS=-static

RUN apk update && apk add --no-cache git autoconf make musl-dev clang dtc

RUN git clone https://github.com/opencompl/riscv-isa-sim && cd riscv-isa-sim && git checkout $SNITCH_SPIKE_SHA
RUN cd riscv-isa-sim && ./configure --with-target=riscv32-unknown-elf --with-isa=RV32IMAFD --prefix=$INSTALL_DIR && \
    make -j$(nproc) && make install
RUN strip $INSTALL_DIR/bin/spike
RUN strip $INSTALL_DIR/bin/spike-dasm

FROM alpine:3.18

ARG INSTALL_DIR

COPY --from=toolchain-build $INSTALL_DIR/quidditch-toolchain $INSTALL_DIR/quidditch-toolchain
COPY --from=bender-build $INSTALL_DIR/quidditch-toolchain/bin/bender $INSTALL_DIR/quidditch-toolchain/bin
COPY --from=pulp-build $INSTALL_DIR/quidditch-toolchain/bin/pulp-as $INSTALL_DIR/quidditch-toolchain/bin
COPY --from=snitch-verilator-build $INSTALL_DIR/quidditch-toolchain/bin/snitch_cluster_nofdiv.vlt $INSTALL_DIR/quidditch-toolchain/bin
COPY --from=snitch-fdiv-verilator-build-pgo $INSTALL_DIR/quidditch-toolchain/bin/snitch_cluster.vlt $INSTALL_DIR/quidditch-toolchain/bin
COPY --from=spike-build $INSTALL_DIR/bin/spike $INSTALL_DIR/quidditch-toolchain/bin
COPY --from=spike-build $INSTALL_DIR/bin/spike-dasm $INSTALL_DIR/quidditch-toolchain/bin
