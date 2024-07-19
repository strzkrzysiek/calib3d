#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <map>
#include <optional>

namespace calib3d {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using VecX = Eigen::VectorXd;

using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using Mat3x4 = Eigen::Matrix<double, 3, 4>;
using Mat2X = Eigen::Matrix2Xd;
using Mat3X = Eigen::Matrix3Xd;
using Mat4X = Eigen::Matrix4Xd;

using Quat = Eigen::Quaterniond;

struct CameraIntrinsics {
  Vec2 principal_point;
  double f;

  [[nodiscard]] Mat3 K() const {
    return (Mat3() << f, 0., principal_point[0], 0., f, principal_point[1], 0., 0., 1.).finished();
  }
};

struct CameraExtrinsics {
  Quat world2cam_rot;
  Vec3 world_in_cam_pos;

  [[nodiscard]] Mat4 matrix() const {
    Mat4 world2cam = Mat4::Identity();
    world2cam.topLeftCorner<3, 3>() = world2cam_rot.matrix();
    world2cam.topRightCorner<3, 1>() = world_in_cam_pos;

    return world2cam;
  }

  [[nodiscard]] Vec3 cam_in_world_pos() const { return -(world2cam_rot.conjugate() * world_in_cam_pos); }
};

using CamId = size_t;
using PointId = size_t;
using CameraSize = Eigen::Vector2i;

using Observations = std::map<CamId, Vec2>;

struct CameraCalib {
  CameraIntrinsics intrinsics;
  CameraExtrinsics extrinsics;
  CameraSize size;

  [[nodiscard]] Mat3x4 P() const { return intrinsics.K() * extrinsics.matrix().topRows<3>(); }
};

template <class T>
using ThreeOf = std::array<T, 3>;

struct PointData {
  std::optional<Vec3> world_pt;
  std::map<CamId, Vec2> image_pts;
};

} // namespace calib3d
