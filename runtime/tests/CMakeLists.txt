include(CTest)

# Sanity check that our toolchain, emulator etc. work
add_executable(HelloWorld main.c)
target_link_libraries(HelloWorld snRuntime)


add_test(NAME HelloWorld COMMAND HelloWorld)
