add_executable(turbo64-16-server server-main.cpp private-logs.cpp)
target_link_libraries(turbo64-16-server PRIVATE llama-server-impl)
target_compile_features(turbo64-16-server PRIVATE cxx_std_17)
if (LLAMA_BUILD_TESTS)
    add_executable(test-turbo64-16-private-logs test-private-logs.cpp private-logs.cpp)
    target_link_libraries(test-turbo64-16-private-logs PRIVATE llama-common)
    target_compile_features(test-turbo64-16-private-logs PRIVATE cxx_std_17)
endif()
if (WIN32)
    target_link_libraries(turbo64-16-server PRIVATE psapi shell32)
endif()
set(TURBO64_16_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bin" CACHE PATH "Turbo64-16 server package directory")
add_custom_target(turbo64-16-package ALL
    COMMAND ${CMAKE_COMMAND}
        "-DSERVER=$<TARGET_FILE:turbo64-16-server>"
        "-DBINARY_DIR=$<TARGET_FILE_DIR:turbo64-16-server>"
        "-DDESTINATION=${TURBO64_16_PACKAGE_DIR}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/package-server.cmake
    DEPENDS turbo64-16-server
    VERBATIM)
