execute_process(
  COMMAND bison -rall ${PROJECT_SOURCE_DIR}/euler/parser/gremlin.y -d -b ${PROJECT_SOURCE_DIR}/euler/parser/gremlin)
execute_process(
  COMMAND flex -o ${PROJECT_SOURCE_DIR}/euler/parser/lex.yy.c ${PROJECT_SOURCE_DIR}/euler/parser/gremlin.l)

SET_SOURCE_FILES_PROPERTIES(lex.yy.c gremlin.tab.c PROPERTIES LANGUAGE CXX)

add_library(gen_lex lex.yy.c gremlin.tab.c)
target_compile_options(gen_lex PRIVATE -Wno-all)

add_library(compiler attribute_calculator.cc gen_node_def_input_output.cc translator.cc optimizer.cc compiler.cc optimize_type.cc)
target_link_libraries(compiler gen_lex common dag_def)

add_executable(tree_test tree_test.cc)
target_link_libraries(tree_test gtest gtest_main)
add_test(NAME tree_test COMMAND tree_test)

add_executable(translator_test translator_test.cc)
target_link_libraries(translator_test compiler gtest gtest_main)
add_test(NAME translator_test COMMAND translator_test)

add_executable(compiler_test compiler_test.cc)
target_link_libraries(compiler_test compiler framework gtest gtest_main)
add_test(NAME compiler_test COMMAND compiler_test)
