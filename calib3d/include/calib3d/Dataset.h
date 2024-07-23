// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <Eigen/Core>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <calib3d/types.h>

namespace calib3d {


// Structure to hold camera calibration data and observations
struct CameraBundle {
  CameraCalib calib;              // Camera calibration parameters
  Observations observations;      // Observed image points
};

// Dataset structure to manage multiple cameras and 3D world points
struct Dataset {
  std::map<CamId, CameraBundle> cameras;  // Map of camera IDs to camera bundles
  std::map<PointId, Vec3> world_points;   // Map of point IDs to 3D world points

  // Loads dataset from a JSON file
  bool loadFromJson(const std::string& filename, bool verify = false);

  // Dumps dataset to a JSON file
  bool dumpToJson(const std::string& filename);

  // Verifies the geometric integrity of the dataset by verifying the reprojection error
  [[nodiscard]] bool verifyReprojectionError() const;

  // Adds noise to the observations
  void addObservationNoise(double noise, size_t seed = std::random_device()());

  // Adds outliers to the observations
  void addObservationOutliers(double inlier_prob, size_t seed = std::random_device()());

  // Gets common point IDs observed by a set of cameras
  [[nodiscard]] std::vector<PointId> getCommonPointIds(const std::vector<CamId>& cam_ids) const;

  // Gets image points for a specific camera and set of point IDs
  [[nodiscard]] Mat2X getImagePointArray(CamId cam_id, const std::vector<PointId>& pt_ids) const;

  // Gets world points for a set of point IDs
  [[nodiscard]] Mat3X getWorldPointArray(const std::vector<PointId>& pt_ids) const;
};

} // namespace calib3d
