
# "${CONDA_PREFIX}"/snax-utils/snax-mac/target/snitch_cluster
set(SNAX_ROOT_DIR "/home/vasich/wb/snax-mlir/.pixi/envs/default/snax-utils/snax-mac/")

find_package(SNAX REQUIRED)

add_executable(snax-helloworld main.c)

target_link_libraries(
    snax-helloworld
    PRIVATE
    samples_util
    SNAX::SNAX
    Quidditch::dispatch::dispatch
)

