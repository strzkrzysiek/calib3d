#pragma once

#include <memory>

#include <calib3d/types.h>

namespace calib3d {

class BAProblem {
public:
  enum class CameraType { CAM_0, CAM_1, CAM_N };

  explicit BAProblem(double observation_noise);
  ~BAProblem();

  void addCamera(CameraCalib& calib, CameraType type = CameraType::CAM_N);
  void addObservation(CameraCalib& calib, Vec3& world_pt, const Vec2& image_pt);

  void optimize();

  void setPrincipalPointVariable();
  void setPrincipalPointConstant();

private:
  struct Impl;

  std::unique_ptr<Impl> impl_;
};

} // namespace calib3d
