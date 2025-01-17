// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <memory>

#include <calib3d/types.h>

namespace calib3d {

// Class for refining camera calibration parameters (extrinsics and focal length)
// with an iterative non-linear least squares method
class CameraCalibRefinementProblem {
public:
  // Constructor to initialize with a camera calibration, initial projection matrix, and outlier threshold
  CameraCalibRefinementProblem(CameraCalib& calib, const Mat3x4& initial_P, double outlier_thr);

  // Destructor
  ~CameraCalibRefinementProblem();

  // Adds correspondences between world points and image points
  // Only inliers are added to the optimization procedure
  void addCorrespondences(const Mat3X& world_pts, const Mat2X& image_pts);

  // Optimizes the camera calibration
  void optimize();

private:
  // The implementation is hidden so that the ceres headers are not included in other parts of the project
  struct Impl;                 // Forward declaration for the implementation
  std::unique_ptr<Impl> impl_; // Unique pointer to the implementation
};

} // namespace calib3d
