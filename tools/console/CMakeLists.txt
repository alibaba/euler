add_executable(console console.cc)
target_link_libraries(console client ${linenoise_STATIC})
