add_compile_options(-Wall -Wextra -pedantic -Werror)

add_library(calib3d
    include/calib3d/BAProblem.h
    include/calib3d/calib_utils.h
    include/calib3d/calib_utils.hpp
    include/calib3d/calib3d.h
    include/calib3d/dataset.h
    include/calib3d/ransac.h
    include/calib3d/ransac.hpp
    include/calib3d/ThreeViewReconstruction.h
    include/calib3d/ThreeViewReconstructionWithBA.h
    include/calib3d/types.h
    src/BAProblem.cpp
    src/calib3d.cpp
    src/dataset.cpp
    src/ThreeViewReconstruction.cpp
    src/ThreeViewReconstructionWithBA.cpp
)

set_source_files_properties(src/BAProblem.cpp PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)

target_include_directories(calib3d
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(calib3d
    PUBLIC Eigen3::Eigen
    PUBLIC sophus
    PRIVATE Ceres::ceres
    PRIVATE glog::glog
    PRIVATE nlohmann_json::nlohmann_json
)

enable_testing()

add_executable(test_calib3d
    tests/test_main.cpp
    tests/test_calib_utils.cpp
    tests/test_three_view_reconstruction.cpp
)

set(DATASET_FILE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../data/accurate-calibrations-and-accurate-world-points.json")
target_compile_definitions(test_calib3d PRIVATE DATASET_FILE_PATH="${DATASET_FILE_PATH}")

target_link_libraries(test_calib3d
    PRIVATE calib3d
    PRIVATE GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_calib3d)
