#pragma once

#include <Eigen/Core>
#include <array>
#include <map>
#include <vector>

#include <calib3d/types.h>

namespace calib3d {

class ThreeViewReconstruction {
public:
  ThreeViewReconstruction(const CameraSize& cam0_size,
                          const CameraSize& cam1_size,
                          const CameraSize& cam2_size,
                          double observation_noise);

  void reconstruct(const Observations& cam0_obs,
                   const Observations& cam1_obs,
                   const Observations& cam2_obs);

private:
  void prepareCommonObservations(const Observations& cam0_obs,
                                 const Observations& cam1_obs,
                                 const Observations& cam2_obs);

  void performProjectiveReconstruction();
  static Eigen::Matrix<double, 3, 4> getProjectionMatrixFromFundamentalMatrix(
      const Eigen::Matrix3d& F);

  void performMetricRectification();
  [[nodiscard]] Eigen::Matrix4d findAbsoluteDualQuadratic() const;
  void findCameraMatrices(const Eigen::Matrix4d& ADQ);
  [[nodiscard]] Eigen::Matrix4d findRectificationHomography(const Eigen::Matrix4d& ADQ) const;
  void transformReconstruction(const Eigen::Matrix4d& H);

  void recoverCameraCalibrations();

  void prepareFinalReconstruction(const Observations& cam0_obs,
                                  const Observations& cam1_obs,
                                  const Observations& cam2_obs);

  double observation_noise_;
  std::array<CameraSize, 3> cam_sizes_;
  std::vector<size_t> common_pt_ids_;
  std::array<Eigen::Matrix2Xd, 3> image_pts_;
  Eigen::Matrix3Xd world_pts_;

  std::array<Eigen::Matrix<double, 3, 4>, 3> P_;
  std::array<Eigen::Matrix3d, 3> K_;
  std::array<Eigen::Matrix3d, 3> R_;
  std::array<Eigen::Vector3d, 3> t_;

  std::array<CameraCalib, 3> camera_calib_;
  std::map<size_t, Eigen::Vector3d> reconstruction_;
};

} // namespace calib3d
