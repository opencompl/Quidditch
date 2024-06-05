
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/NsNet2.py nsnet2.mlirbc --dtype=f32
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS NsNet2.py
    COMMENT "Translating NsNet2.py using iree-turbine"
)

quidditch_module(SRC ${CMAKE_CURRENT_BINARY_DIR}/nsnet2.mlirbc LLVM)

add_executable(NsNet2LLVM main.c)
target_link_libraries(
    NsNet2LLVM
    PRIVATE
    samples_util
    nsnet2_module
    snRuntime
)
