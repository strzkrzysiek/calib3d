#pragma once

#include <Eigen/Core>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <calib3d/types.h>

namespace calib3d {

struct CameraBundle {
  CameraCalib calib;
  Observations observations;
};

struct Dataset {
  std::map<CamId, CameraBundle> cameras;
  std::map<PointId, Vec3> world_points;

  bool loadFromJson(const std::string& filename);

  bool verifyDataset() const;

  void addObservationNoise(double noise, size_t seed = std::random_device()());
  void addObservationOutliers(double inlier_prob, size_t seed = std::random_device()());

  [[nodiscard]] std::vector<PointId> getCommonPointIds(const std::vector<CamId>& cam_ids) const;
  [[nodiscard]] Mat2X getImagePointArray(CamId cam_id, const std::vector<PointId>& pt_ids) const;
  [[nodiscard]] Mat3X getWorldPointArray(const std::vector<PointId>& pt_ids) const;
};

} // namespace calib3d
