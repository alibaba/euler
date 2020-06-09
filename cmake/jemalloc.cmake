include (ExternalProject)

set(jemalloc_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/src/jemalloc/include)
set(jemalloc_URL https://github.com/jemalloc/jemalloc/archive/5.1.0.tar.gz)
set(jemalloc_HASH SHA256=ff28aef89df724bd7b6bd6fde8597695514e0e3404d1afad2f1eb8b55ef378d3)

set(jemalloc_BUILD ${CMAKE_CURRENT_BINARY_DIR}/jemalloc/)

ExternalProject_Add(jemalloc
    PREFIX jemalloc
    URL ${jemalloc_URL}
    URL_HASH ${jemalloc_HASH}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    LOG_UPDATE 0
    LOG_CONFIGURE 0
    LOG_BUILD 0
    BUILD_BYPRODUCTS ${jemalloc_STATIC} ${jemalloc_STATIC_PIC}
    CONFIGURE_COMMAND ./autogen.sh && ./configure --disable-initial-exec-tls --prefix=${CMAKE_CURRENT_BINARY_DIR}/jemalloc
    BUILD_COMMAND ${MAKE}
    INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Skipping install step."
)

ExternalProject_Get_Property(jemalloc INSTALL_DIR)

add_library(jemalloc_STATIC STATIC IMPORTED)
set_property(TARGET jemalloc_STATIC PROPERTY IMPORTED_LOCATION ${INSTALL_DIR}/src/jemalloc/lib/libjemalloc.a)
add_dependencies(jemalloc_STATIC jemalloc)

add_library(jemalloc_STATIC_PIC STATIC IMPORTED)
set_property(TARGET jemalloc_STATIC_PIC PROPERTY IMPORTED_LOCATION ${INSTALL_DIR}/src/jemalloc/lib/libjemalloc_pic.a)
add_dependencies(jemalloc_STATIC_PIC jemalloc)

add_library(jemalloc_SHARED SHARED IMPORTED)
set_property(TARGET jemalloc_SHARED PROPERTY IMPORTED_LOCATION ${INSTALL_DIR}/src/jemalloc/lib/libjemalloc.so)
add_dependencies(jemalloc_SHARED jemalloc)

if (!APPLE)
  link_libraries(-Wl,--no-as-needed)
endif(!APPLE)

link_libraries(dl ${jemalloc_STATIC_PIC})
