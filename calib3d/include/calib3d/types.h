// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <optional>
#include <sophus/se3.hpp>

namespace calib3d {

// Type aliases for common Eigen vector and matrix types
template <class T>
using Vec2T = Eigen::Vector2<T>;
using Vec2 = Vec2T<double>;

template <class T>
using Vec3T = Eigen::Vector3<T>;
using Vec3 = Vec3T<double>;

template <class T>
using Vec4T = Eigen::Vector4<T>;
using Vec4 = Vec4T<double>;

template <class T>
using VecXT = Eigen::VectorX<T>;
using VecX = VecXT<double>;

template <class T>
using Mat3T = Eigen::Matrix3<T>;
using Mat3 = Mat3T<double>;

template <class T>
using Mat4T = Eigen::Matrix4<T>;
using Mat4 = Mat4T<double>;

template <class T>
using Mat3x4T = Eigen::Matrix<T, 3, 4>;
using Mat3x4 = Mat3x4T<double>;

template <class T>
using Mat2XT = Eigen::Matrix2X<T>;
using Mat2X = Mat2XT<double>;

template <class T>
using Mat3XT = Eigen::Matrix3X<T>;
using Mat3X = Mat3XT<double>;

template <class T>
using Mat4XT = Eigen::Matrix4X<T>;
using Mat4X = Mat4XT<double>;

template <class T>
using SE3T = Sophus::SE3<T>;
using SE3 = SE3T<double>;

using CamId = size_t;
using PointId = size_t;

using Observations = std::map<CamId, Vec2>; // Map of camera ID to 2D observations
using CameraSize = Eigen::Vector2i;         // Size of the camera

template <class T>
using CameraExtrinsicsT = SE3T<T>;

using CameraExtrinsics = CameraExtrinsicsT<double>; // Camera extrinsics type

// Structure to hold camera intrinsics parameters
struct CameraIntrinsics {
  Vec2 principal_point; // Principal point of the camera
  double focal_length;  // Focal length of the camera

  [[nodiscard]] Mat3 K() const {
    // Return the camera intrinsic matrix
    // clang-format off
    return (Mat3() <<
        focal_length, 0.,           principal_point[0],
        0.,           focal_length, principal_point[1],
        0.,           0.,           1.).finished();
    // clang-format on
  }
};

// Structure to hold complete camera calibration data
struct CameraCalib {
  CameraIntrinsics intrinsics; // Intrinsic parameters
  CameraExtrinsics world2cam;  // Extrinsic parameters (world to camera transformation)
  CameraSize size;             // Size of the camera

  [[nodiscard]] Mat3x4 P() const { return intrinsics.K() * world2cam.matrix3x4(); } // Return the projection matrix
};

template <class T>
using ThreeOf = std::array<T, 3>; // Array of three elements

// Structure to hold point data in the world and image coordinates
struct PointData {
  std::optional<Vec3> world_pt;    // 3D point in the world coordinate system
  std::map<CamId, Vec2> image_pts; // Map of camera ID to 2D image points
};

} // namespace calib3d
