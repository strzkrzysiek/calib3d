#pragma once

#include <Eigen/Core>
#include <map>
#include <utility>
#include <vector>

#include <calib3d/types.h>

namespace calib3d {

class ThreeViewReconstruction {
public:
  explicit ThreeViewReconstruction(double observation_noise);

  [[nodiscard]] const std::map<CamId, CameraCalib>& getCameras() const;
  [[nodiscard]] const std::map<PointId, PointData>& getPoints() const;

  virtual void reconstruct(CamId cam0_id,
                           const CameraSize& cam0_size,
                           const Observations& cam0_obs,
                           CamId cam1_id,
                           const CameraSize& cam1_size,
                           const Observations& cam1_obs,
                           CamId cam2_id,
                           const CameraSize& cam2_size,
                           const Observations& cam2_obs);

protected:
  void insertCameraData(CamId cam_id, const CameraSize& cam_size, const Observations& cam_obs);

  void performThreeViewReconstruction(const ThreeOf<CamId>& cam_ids);

  [[nodiscard]] std::pair<std::vector<PointId>, ThreeOf<Mat2X>> prepareCommonObservations(
      const ThreeOf<CamId>& cam_ids) const;

  [[nodiscard]] std::pair<ThreeOf<Mat3x4>, Mat3X> performProjectiveReconstruction(
      const ThreeOf<Mat2X>& image_pts) const;
  [[nodiscard]] static Mat3x4 getProjectionMatrixFromFundamentalMatrix(const Mat3& F);

  static ThreeOf<Mat3> performMetricRectification(ThreeOf<Mat3x4>& P, Mat3X& world_pts);
  [[nodiscard]] static Mat4 findAbsoluteDualQuadratic(const ThreeOf<Mat3x4>& P);
  [[nodiscard]] static ThreeOf<Mat3> findCameraMatrices(const Mat4& ADQ, const ThreeOf<Mat3x4>& P);
  [[nodiscard]] static Mat3 findCameraMatrix(const Mat4& ADQ, const Mat3x4& P);
  [[nodiscard]] static Mat4 findRectifyingHomography(const Mat4& ADQ, const Mat3& K0, const Mat3X& world_pts);
  static void transformReconstruction(const Mat4& H, ThreeOf<Mat3x4>& P, Mat3X& world_pts);

  void recoverCameraCalibrations(const ThreeOf<CamId>& cam_ids, const ThreeOf<Mat3x4>& P, const ThreeOf<Mat3>& K);
  static void recoverCameraCalibration(const Mat3x4& P, const Mat3& K, CameraCalib& calib);

  void refineCameraCalibrations(const ThreeOf<CamId>& cam_ids,
                                const ThreeOf<Mat3x4>& P,
                                const Mat3X& world_pts,
                                const ThreeOf<Mat2X>& image_pts);
  void refineCameraCalibration(const Mat3x4& P,
                               const Mat3X& world_pts,
                               const Mat2X& image_pts,
                               CameraCalib& calib) const;

  void fixRotationAndScale(const ThreeOf<CamId>& cam_ids, Mat3X& world_pts);

  void writeBackWorldPoints(const std::vector<PointId>& common_pt_ids, const Mat3X& world_pts);

  void triangulateRemainingPoints();

protected:
  std::map<CamId, CameraCalib> cameras_;
  std::map<PointId, PointData> points_;
  const double observation_noise_;
  const double percentile_95_;
  const double ransac_thr_;
  const double ransac_confidence_;
  const size_t ransac_max_iters_;
};

} // namespace calib3d
