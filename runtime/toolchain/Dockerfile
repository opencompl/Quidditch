
# Global args.
ARG LLVM_VERSION=18
ARG PICOLIBC_VERSION=1.8.6
ARG SNITCH_CLUSTER_SHA=1f984781957520b3784b9ab17e5eeb34eec763f1
ARG VERILATOR_RELEASE=v4.228
ARG BENDER_RELEASE=0.27.1
ARG INSTALL_DIR=/opt
ARG PULP_TOOLCHAIN_RELEASE=0.12.0
ARG SNITCH_SPIKE_SHA=fd3f3da0d0ad9a3dd2f8283c4c3d874900aa8a47

FROM alpine:3.18 as toolchain-build

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
FROM ubuntu:24.04 as bender-build

ARG BENDER_RELEASE
ARG INSTALL_DIR

ENV RUSTFLAGS="-C target-feature=+crt-static"

RUN apt-get update && apt-get install -y cargo
RUN cargo install bender --locked --version $BENDER_RELEASE --target x86_64-unknown-linux-gnu
RUN mkdir -p $INSTALL_DIR/quidditch-toolchain/bin/
RUN cp /root/.cargo/bin/bender $INSTALL_DIR/quidditch-toolchain/bin/

FROM alpine:3.18 as pulp-build

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

FROM alpine:3.18 as verilator-build

ARG SNITCH_CLUSTER_SHA
ARG VERILATOR_RELEASE
ARG INSTALL_DIR

WORKDIR $INSTALL_DIR

RUN apk update && apk add --no-cache py3-pip git autoconf flex-dev bison make musl-dev clang coreutils dtc patch

ENV PATH=/root:$PATH
COPY --from=bender-build $INSTALL_DIR/quidditch-toolchain/bin/bender /root

COPY requirements.txt /root/
RUN pip install --break-system-packages -r /root/requirements.txt
RUN git clone https://github.com/pulp-platform/snitch_cluster && \
    cd snitch_cluster && git checkout $SNITCH_CLUSTER_SHA && git submodule update --init --recursive

# snitch_cluster repo does not yet support Verilator 5, see: https://github.com/pulp-platform/snitch_cluster/pull/76.
ENV VERILATOR_ROOT=$INSTALL_DIR/verilator
RUN git clone --depth 1 https://github.com/verilator/verilator -b $VERILATOR_RELEASE
RUN cd verilator && autoconf && CC=clang CXX=clang++ ./configure && make -j$(nproc)
ENV VLT=$VERILATOR_ROOT/bin/verilator
RUN cd ./snitch_cluster/target/snitch_cluster && \
    make VLT_USE_LLVM=ON LDFLAGS=-static bin/snitch_cluster.vlt -j$(nproc)

RUN mkdir -p $INSTALL_DIR/quidditch-toolchain/bin/
RUN cp ./snitch_cluster/target/snitch_cluster/bin/snitch_cluster.vlt $INSTALL_DIR/quidditch-toolchain/bin/
RUN strip $INSTALL_DIR/quidditch-toolchain/bin/snitch_cluster.vlt

FROM alpine:3.18 as spike-build

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
COPY --from=verilator-build $INSTALL_DIR/quidditch-toolchain/bin/snitch_cluster.vlt $INSTALL_DIR/quidditch-toolchain/bin
COPY --from=spike-build $INSTALL_DIR/bin/spike $INSTALL_DIR/quidditch-toolchain/bin
COPY --from=spike-build $INSTALL_DIR/bin/spike-dasm $INSTALL_DIR/quidditch-toolchain/bin
