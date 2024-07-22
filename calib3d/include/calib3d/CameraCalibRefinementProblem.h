#pragma once

#include <memory>

#include <calib3d/types.h>

namespace calib3d {

class CameraCalibRefinementProblem {
public:
  CameraCalibRefinementProblem(CameraCalib& calib, const Mat3x4& initial_P, double outlier_thr);
  ~CameraCalibRefinementProblem();

  void addCorrespondences(const Mat3X& world_pts, const Mat2X& image_pts);
  void optimize();

private:
  struct Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace calib3d
