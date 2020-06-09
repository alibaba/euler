add_library(index index_meta.cc index_manager.cc common_index_result.cc)
target_link_libraries(index framework core)

add_executable(hash_sample_index_test hash_sample_index_test.cc)
target_link_libraries(hash_sample_index_test index
                      ${CMAKE_THREAD_LIBS_INIT}
                      gtest gtest_main common)

add_executable(range_sample_index_test range_sample_index_test.cc)
target_link_libraries(range_sample_index_test index
                      ${CMAKE_THREAD_LIBS_INIT}
                      gtest gtest_main common)

add_executable(hash_range_sample_index_test hash_range_sample_index_test.cc)
target_link_libraries(hash_range_sample_index_test index
                      ${CMAKE_THREAD_LIBS_INIT}
                      gtest gtest_main common)


add_executable(hash_index_result_test hash_index_result_test.cc)
target_link_libraries(hash_index_result_test index
                      ${CMAKE_THREAD_LIBS_INIT}
                      gtest gtest_main common)

add_executable(range_index_result_test range_index_result_test.cc)
target_link_libraries(range_index_result_test index
                      ${CMAKE_THREAD_LIBS_INIT}
                      gtest gtest_main common)

add_executable(common_index_result_test common_index_result_test.cc)
target_link_libraries(common_index_result_test index
                      ${CMAKE_THREAD_LIBS_INIT}
                      gtest gtest_main common)

add_executable(index_meta_test index_meta_test.cc)
target_link_libraries(index_meta_test index
                      ${CMAKE_THREAD_LIBS_INIT}
                      gtest gtest_main common)

add_executable(index_manager_test index_manager_test.cc)
target_link_libraries(index_manager_test index
                      ${CMAKE_THREAD_LIBS_INIT}
                      gtest gtest_main common)

add_test(NAME hash_sample_index_test COMMAND hash_sample_index_test)
add_test(NAME range_sample_index_test COMMAND range_sample_index_test)
add_test(NAME hash_range_sample_index_test COMMAND hash_range_sample_index_test)
add_test(NAME hash_index_result_test COMMAND hash_index_result_test)
add_test(NAME range_index_result_test COMMAND range_index_result_test)
add_test(NAME common_index_result_test COMMAND common_index_result_test)
add_test(NAME index_meta_test COMMAND index_meta_test)
add_test(NAME index_manager_test COMMAND index_manager_test)
