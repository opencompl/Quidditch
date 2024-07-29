include(CTest)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test.o
    COMMAND ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/pulp-as
    ${CMAKE_CURRENT_LIST_DIR}/test.s --filetype=obj --target-abi=ilp32d
    --mcpu=snitch -g -o ${CMAKE_CURRENT_BINARY_DIR}/test.o
    DEPENDS test.s ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/pulp-as)

# Sanity check that our toolchain, emulator etc. work
add_executable(HelloWorld main.c ${CMAKE_CURRENT_BINARY_DIR}/test.o)
target_link_libraries(HelloWorld snRuntime)

macro(test_executable target_name)
  cmake_parse_arguments(_RULE "PRECOMMIT;NIGHTLY" "" "" ${ARGN})

  set(label)
  if (_RULE_PRECOMMIT)
    set(label "precommit")
  elseif (_RULE_NIGHTLY)
    set(label "nightly")
  else ()
    message(FATAL_ERROR "Test must either be in precommit or nightly")
  endif ()

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.test)
  add_test(NAME ${target_name}
      COMMAND ${target_name}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.test)
  set_tests_properties(${target_name} PROPERTIES
      FIXTURES_REQUIRED ${target_name}-fixture
      LABELS ${label}
  )
  if (_RULE_NIGHTLY)
    set_tests_properties(${target_name} PROPERTIES
        TIMEOUT 0
    )
  endif ()

  add_test(NAME ${target_name}-setup
      COMMAND rm -rf ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.test/*
  )
  set_tests_properties(${target_name}-setup PROPERTIES
      FIXTURES_SETUP ${target_name}-fixture
      LABELS ${label}
  )

  set(gen_traces_targets)
  set(gen_traces_jsons)
  set(gen_traces_raw)
  foreach (i RANGE 0 8)
    add_test(NAME ${target_name}-analysis-hart-${i}
        COMMAND
        sh -c "\
        cat ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.test/logs/trace_hart_0000000${i}.dasm \
        | ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/spike-dasm > \
        ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.test/logs/trace_hart_0000000${i}.asm \
        && ${Python3_EXECUTABLE} ${SNITCH_CLUSTER_SOURCE_DIR}/util/trace/gen_trace.py \
        ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.test/logs/trace_hart_0000000${i}.asm \
        -d hart_${i}_perf.json > metrics_hart_${i}.txt"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.test
    )
    set_tests_properties(${target_name}-analysis-hart-${i} PROPERTIES
        FIXTURES_CLEANUP ${target_name}-fixture
        LABELS ${label}
    )
    list(APPEND gen_traces_targets "${target_name}-analysis-hart-${i}")
    list(APPEND gen_traces_jsons "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.test/hart_${i}_perf.json")
    list(APPEND gen_traces_raw "${CMAKE_CURRENT_BINARY_DIR}/${target_name}.test/logs/trace_hart_0000000${i}.dasm")
  endforeach ()

  add_test(NAME ${target_name}-events.json
      COMMAND
      ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/util/snitch_trace_to_perfetto.py
      -i ${gen_traces_jsons} --traces ${gen_traces_raw} --elf $<TARGET_FILE:${target_name}>
      --addr2line ${QUIDDITCH_TOOLCHAIN_ROOT}/bin/llvm-addr2line
      -o events.json
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${target_name}.test
      COMMAND_EXPAND_LISTS
  )
  set_tests_properties(${target_name}-events.json PROPERTIES
      FIXTURES_CLEANUP ${target_name}-fixture
      DEPENDS "${gen_traces_targets}"
      LABELS ${label}
  )
endmacro()

test_executable(HelloWorld PRECOMMIT)
test_executable(vec_multiply PRECOMMIT)
test_executable(NsNet2 NIGHTLY)
test_executable(NsNet2LLVM NIGHTLY)
