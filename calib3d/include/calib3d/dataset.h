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
};

bool loadJsonDataset(const std::string& filename, Dataset& dataset);

} // namespace calib3d
