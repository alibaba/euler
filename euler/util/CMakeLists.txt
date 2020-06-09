add_library(euler_util SHARED python_api.cc)
target_link_libraries(euler_util common)

add_custom_command(
  TARGET euler_util POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:common> ${PROJECT_SOURCE_DIR}/euler/tools
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:euler_util> ${PROJECT_SOURCE_DIR}/euler/tools
  COMMENT "Building test data")
