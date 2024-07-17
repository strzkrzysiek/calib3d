cmake_minimum_required(VERSION 3.16)

project(bolt6-ba LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -pedantic -Werror)

add_subdirectory(calib3d)

if(NOT CLANG_FORMAT_BIN)
    find_program(CLANG_FORMAT_BIN NAMES clang-format)
    if(NOT CLANG_FORMAT_BIN)
        message(FATAL_ERROR "clang-format not found")
    endif()
endif()

file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/*.cpp
    ${CMAKE_SOURCE_DIR}/*.h
    ${CMAKE_SOURCE_DIR}/*.hpp
)

add_custom_target(
    format
    COMMAND ${CLANG_FORMAT_BIN}
    -i
    ${ALL_CXX_SOURCE_FILES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-format"
)
