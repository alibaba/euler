include (ExternalProject)

set(linenoise_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/linenoise/src)

set(linenoise_URL https://github.com/antirez/linenoise/archive/1.0.tar.gz)
set(linenoise_HASH SHA256=f5054a4fe120d43d85427cf58af93e56b9bb80389d507a9bec9b75531a340014)

set(linenoise_BUILD ${CMAKE_CURRENT_BINARY_DIR}/linenoise/src/linenoise)

ExternalProject_Add(
    linenoise
    PREFIX linenoise
    URL ${linenoise_URL}
    URL_HASH ${linenoise_HASH}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS linenoise_STATIC
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E echo "Skipping configure"
    BUILD_COMMAND  ${CMAKE_C_COMPILER} -c -fPIC linenoise.c && ${CMAKE_AR} rcs liblinenoise.a linenoise.o
    INSTALL_COMMAND ${CMAKE_COMMAND} -E echo "Skipping install"
)

ExternalProject_Get_Property(linenoise INSTALL_DIR)

set(linenoise_STATIC ${linenoise_BUILD}/liblinenoise.a)
