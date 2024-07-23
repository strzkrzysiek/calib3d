# Calib3D

TThis project provides a library for the autocalibration of cameras' extrinsic and intrinsic parameters, along with 3D scene reconstruction. The purpose of the project is to showcase the computer vision skills of the author.

## Building

The project has been developed on Ubuntu and compiled with `g++ 11.4.0`. . All necessary libraries have been installed with apt-get. However, with some small modifications, it should be easy to compile and run this project with other UNIX-based toolchains. Windows is not supported.



Below is the list of required libraries:

```shell
sudo apt-get install cmake 
sudo apt-get install libgoogle-glog-dev libgflags-dev libgtest-dev
sudo apt-get install libeigen3-dev nlohmann-json3-dev libboost-all-dev
```

Remember to clone the repo together with the submodules (`ceres-solver` and `Sophus`):
```shell
git clone --recurse-submodules https://github.com/strzkrzysiek/calib3d.git
```

CMake is used as the build system. Make sure to build in the release configuration, otherwise the performance will be poor.

```shell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```


## Algorithm

The input of the algorithm is a list of image sizes and matched correspondences in each image. The algorithm requires no other prior information.

It assumes a pinhole camera model with no distortion, square pixels (zero skew and focal lengths equal on X and Y axes), and the principal point relatively in the center of the image.

To some extent, it is robust to noise in the provided observations and some outliers.

Below is the outline of the algorithm:

1. For the first three cameras, perform **metric reconstruction**:
   * Perform **perspective reconstruction**
   * Perform **metric rectification**
   * Optimize solution with **bundle adjustment**
2. Iteratively, for each of the remaining cameras:
   * Recover camera calibration using 2D-3D correspondences
   * Triangulate new points & retriangulate outliers
   * Optimize solution with **bundle adjustment**

The library doesn't use any OpenCV utilities. That's why it implements a bunch of helper functions that could be found in OpenCV's `calib3d` module:
* Homography estimation
* Projection matrix estimation
* RANSAC engine,
* Point triangulation,
* Etc.

## App

There is a simple app provided to generate calibrated scenes.
```
Calib3D - usage:
  --help                  Produce help message
  -i [ --input ] arg      Input file path
  -o [ --output ] arg     Output file path
  -n [ --noise ] arg (=2) Standard deviation of observation error
  -p [ --optprincipal ]   Optimize also the principal point
```

## Tests

There are several unit tests implemented that show how to use particular parts of the library code. In particular, the most complicated case of autocalibration with noisy observations and outliers is covered.
