#pragma once

#include <Eigen/Core>
#include <map>
#include <string>

#include <calib3d/types.h>

namespace calib3d {

struct CameraBundle {
  CameraIntrinsics intrinsics;
  CameraExtrinsics extrinsics;
  CameraSize size;
  Observations observations;
};

struct Dataset {
  std::map<size_t, CameraBundle> cameras;
  std::map<size_t, Eigen::Vector3d> world_points;
};

bool loadJsonDataset(const std::string& filename, Dataset& dataset);

} // namespace calib3d
