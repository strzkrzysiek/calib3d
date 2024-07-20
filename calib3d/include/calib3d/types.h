#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <optional>
#include <sophus/se3.hpp>

namespace calib3d {

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

using Observations = std::map<CamId, Vec2>;
using CameraSize = Eigen::Vector2i;

template <class T>
using CameraExtrinsicsT = SE3T<T>;

using CameraExtrinsics = CameraExtrinsicsT<double>;

struct CameraIntrinsics {
  Vec2 principal_point;
  double focal_length;

  [[nodiscard]] Mat3 K() const {
    // clang-format off
    return (Mat3() <<
        focal_length, 0.,           principal_point[0],
        0.,           focal_length, principal_point[1],
        0.,           0.,           1.).finished();
    // clang-format on
  }
};

// template <class T>
// struct CameraIntrinsicsT : public Eigen::Vector3<T> {
//  [[nodiscard]] auto principal_point() { return this->template head<2>(); }
//  [[nodiscard]] auto principal_point() const { return this->template head<2>(); }
//
//  [[nodiscard]] auto& f() { return (*this)[2]; }
//  [[nodiscard]] const auto& f() const { return (*this)[2]; }
//
//  [[nodiscard]] Mat3T<T> K() const {
//    // clang-format off
//    return (Eigen::Matrix3<T>() <<
//        f(), 0.,  principal_point()[0],
//        0.,  f(), principal_point()[1],
//        0.,  0.,  1.).finished();
//    // clang-format on
//  }
//};
//
// using CameraIntrinsics = CameraIntrinsicsT<double>;

struct CameraCalib {
  CameraIntrinsics intrinsics;
  CameraExtrinsics world2cam;
  CameraSize size;

  [[nodiscard]] Mat3x4 P() const { return intrinsics.K() * world2cam.matrix3x4(); }
};

template <class T>
using ThreeOf = std::array<T, 3>;

struct PointData {
  std::optional<Vec3> world_pt;
  std::map<CamId, Vec2> image_pts;
};

} // namespace calib3d
