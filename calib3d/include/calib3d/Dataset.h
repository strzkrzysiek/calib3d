#pragma once

#include <Eigen/Core>
#include <map>
#include <string>

#include <calib3d/types.h>

namespace calib3d {

struct CameraBundle {
  CameraCalib calib;
  Observations observations;
};

struct Dataset {
  std::map<CamId, CameraBundle> cameras;
  std::map<PointId, Eigen::Vector3d> world_points;

  bool loadFromJson(const std::string& filename);
};

} // namespace calib3d
