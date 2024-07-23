// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <memory>

#include <calib3d/types.h>

namespace calib3d {

// Class for defining and solving a Bundle Adjustment problem
class BAProblem {
public:
  // Enumeration for different camera types
  enum class CameraType { CAM_0, CAM_1, CAM_N };

  // Constructor to initialize with observation noise
  explicit BAProblem(double observation_noise);

  // Destructor
  ~BAProblem();

  // Adds a camera calibration to the problem
  void addCamera(CameraCalib& calib, CameraType type = CameraType::CAM_N);

  // Adds an observation to the problem
  void addObservation(CameraCalib& calib, Vec3& world_pt, const Vec2& image_pt);

  // Optimizes the bundle adjustment problem
  void optimize();

  // Sets the principal point as a variable to be optimized
  void setPrincipalPointVariable();

  // Sets the principal point as a constant
  void setPrincipalPointConstant();

private:
  struct Impl;  // Forward declaration for the implementation
  std::unique_ptr<Impl> impl_;  // Unique pointer to the implementation
};

} // namespace calib3d
