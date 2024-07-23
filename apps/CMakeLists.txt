# Copyright 2024 Krzysztof Wrobel

add_executable(calib3d-cli
    calib3d_cli.cpp
)

target_link_libraries(calib3d-cli
    Boost::program_options
    calib3d
    glog::glog
)
