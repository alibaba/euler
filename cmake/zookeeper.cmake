set(WANT_CPPUNIT OFF CACHE BOOL "disable test" FORCE)
set(ZOOKEEPER_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/zookeeper/zookeeper-client/zookeeper-client-c)
add_subdirectory(${ZOOKEEPER_ROOT_DIR} third_party/zookeeper)
set(ZOOKEEPER_INCLUDE_DIRS ${ZOOKEEPER_ROOT_DIR}/include ${ZOOKEEPER_ROOT_DIR}/generated)
