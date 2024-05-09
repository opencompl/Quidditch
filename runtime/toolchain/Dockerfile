
# Global args.
ARG LLVM_VERSION=18
ARG PICOLIBC_VERSION=1.8.6
ARG SNITCH_CLUSTER_SHA=e02cc9e3f24b92d4607455d5345caba3eb6273b2
ARG VERILATOR_RELEASE=v4.228
ARG BENDER_RELEASE=0.27.1
ARG PRESERVE_SOURCE=false
ARG INSTALL_DIR=/opt

FROM alpine:3.18 as toolchain-build

ARG LLVM_VERSION
ARG PICOLIBC_VERSION
ARG INSTALL_DIR

RUN apk update && apk add --no-cache cmake git musl-dev clang clang-dev ccache  \
    python3 ninja py3-pip meson

ENV CCACHE_DIR=/ccache

WORKDIR $INSTALL_DIR

RUN git clone --depth 1  https://github.com/llvm/llvm-project/ -b release/$LLVM_VERSION.x

COPY phase1.cmake /root/

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
    -Dpicocrt=false  \
    -Dpicolib=false  \
    -Dposix-console=true  \
    -Dprefix=$INSTALL_DIR/quidditch-toolchain  \
    -Dspecsdir=none && ninja install

COPY phase2.cmake /root/

RUN cd llvm-project/llvm/build && rm -rf * && \
    cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/quidditch-toolchain \
         -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
         -DCMAKE_C_COMPILER_LAUNCHER=ccache  \
         -DLLVM_BUILD_STATIC=ON  \
         -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
         -C  /root/phase2.cmake && \
        ninja install/strip

FROM alpine:3.18 as verilator-build

ARG SNITCH_CLUSTER_SHA
ARG VERILATOR_RELEASE
ARG BENDER_RELEASE
ARG INSTALL_DIR

ENV CCACHE_DIR=/ccache

WORKDIR $INSTALL_DIR

RUN apk update && apk add --no-cache py3-pip git ccache cargo autoconf flex-dev bison make gcc g++ coreutils dtc patch

RUN cargo install bender --locked --version $BENDER_RELEASE
ENV PATH=$PATH:/root/.cargo/bin
RUN pip install --break-system-packages hjson==3.1.* jsonref==1.1.* mako==1.3.* jsonschema==4.21.* tabulate==0.9.*  \
    PyYAML==6.0.*
RUN git clone https://github.com/pulp-platform/snitch_cluster && \
    cd snitch_cluster && git checkout $SNITCH_CLUSTER_SHA && git submodule update --init --recursive

# snitch_cluster repo does not yet support Verilator 5, see: https://github.com/pulp-platform/snitch_cluster/pull/76.
ENV VERILATOR_ROOT=$INSTALL_DIR/verilator
RUN git clone --depth 1 https://github.com/verilator/verilator -b $VERILATOR_RELEASE
RUN cd verilator && autoconf && ./configure && make -j$(nproc)
ENV VLT=$VERILATOR_ROOT/bin/verilator
COPY snRuntime-01.patch /root/
COPY static-verilator-02.patch /root/
RUN cd ./snitch_cluster && git apply /root/snRuntime-01.patch && git apply /root/static-verilator-02.patch
RUN cd ./snitch_cluster/target/snitch_cluster && \
    make bin/snitch_cluster.vlt -j$(nproc)

RUN mkdir -p $INSTALL_DIR/quidditch-toolchain/bin/
RUN cp ./snitch_cluster/target/snitch_cluster/bin/snitch_cluster.vlt $INSTALL_DIR/quidditch-toolchain/bin/
RUN strip $INSTALL_DIR/quidditch-toolchain/bin/snitch_cluster.vlt
RUN rm -rf ./snitch_cluster/target/snitch_cluster/bin  \
    ./snitch_cluster/target/snitch_cluster/generated  \
    ./snitch_cluster/target/snitch_cluster/work-vlt \
    /ccache

FROM alpine:3.18

ARG INSTALL_DIR

COPY --from=toolchain-build $INSTALL_DIR/quidditch-toolchain $INSTALL_DIR/quidditch-toolchain
COPY --from=verilator-build $INSTALL_DIR/quidditch-toolchain/bin/snitch_cluster.vlt $INSTALL_DIR/quidditch-toolchain
