#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace calib3d {

struct CameraIntrinsics {
  Eigen::Vector2d principal_point;
  double f;

  [[nodiscard]] Eigen::Matrix3d K() const {
    return (Eigen::Matrix3d() << f, 0., principal_point[0], 0., f, principal_point[1], 0., 0., 1.)
        .finished();
  }
};

struct CameraExtrinsics {
  Eigen::Quaterniond world2cam_rot;
  Eigen::Vector3d world_in_cam_pos;

  [[nodiscard]] Eigen::Matrix4d matrix() const {
    Eigen::Matrix4d world2cam = Eigen::Matrix4d::Identity();
    world2cam.topLeftCorner<3, 3>() = world2cam_rot.matrix();
    world2cam.topRightCorner<3, 1>() = world_in_cam_pos;

    return world2cam;
  }
};

using CameraSize = Eigen::Vector2i;

struct CameraCalib {
  CameraIntrinsics intrinsics;
  CameraExtrinsics extrinsics;
};

} // namespace calib3d
