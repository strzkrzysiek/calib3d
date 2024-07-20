cmake_minimum_required(VERSION 3.16)

project(bolt6-ba LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS math_c99)
find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)
find_package(glog REQUIRED)
find_package(GTest REQUIRED)
find_package(nlohmann_json REQUIRED)

add_subdirectory(third-party)
add_subdirectory(calib3d)

if(NOT CLANG_FORMAT_BIN)
    find_program(CLANG_FORMAT_BIN NAMES clang-format)
    if(NOT CLANG_FORMAT_BIN)
        message(FATAL_ERROR "clang-format not found")
    endif()
endif()

file(GLOB_RECURSE ALL_CXX_SOURCE_FILES
    ${CMAKE_SOURCE_DIR}/calib3d/*.cpp
    ${CMAKE_SOURCE_DIR}/calib3d/*.h
    ${CMAKE_SOURCE_DIR}/calib3d/*.hpp
)

add_custom_target(
    format
    COMMAND ${CLANG_FORMAT_BIN}
    -i
    ${ALL_CXX_SOURCE_FILES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-format"
)
