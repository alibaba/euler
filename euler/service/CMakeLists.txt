add_definitions(-Wno-unused-parameter)

add_library(service STATIC graph_service.cc call_data.cc)
target_link_libraries(service core common proto)

add_library(euler_service SHARED graph_service.cc call_data.cc python_api.cc)
target_link_libraries(euler_service core common proto jemalloc_STATIC_PIC)

set(LIB_DST ${CMAKE_CURRENT_SOURCE_DIR}/../python)
add_custom_command(TARGET euler_service POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:euler_service> ${LIB_DST}
)

