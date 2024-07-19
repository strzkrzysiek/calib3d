#pragma once

#include <Eigen/Core>
#include <map>
#include <utility>
#include <vector>

#include <calib3d/types.h>

namespace calib3d {

struct ThreeViewReconstructionParams {
  double observation_noise = 3.0;
};

class ThreeViewReconstruction {
public:
  explicit ThreeViewReconstruction(const ThreeViewReconstructionParams& params);

  [[nodiscard]] const std::map<CamId, CameraCalib>& getCameras() const;
  [[nodiscard]] const std::map<PointId, PointData>& getPoints() const;

  void reconstruct(CamId cam0_id,
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

private:
  void performThreeViewReconstruction();

  [[nodiscard]] std::pair<std::vector<PointId>, ThreeOf<Mat2X>> prepareCommonObservations() const;

  [[nodiscard]] std::pair<ThreeOf<Mat3x4>, Mat3X> performProjectiveReconstruction(
      const ThreeOf<Mat2X>& image_pts) const;
  [[nodiscard]] static Mat3x4 getProjectionMatrixFromFundamentalMatrix(const Mat3& F);

  static ThreeOf<Mat3> performMetricRectification(ThreeOf<Mat3x4>& P, Mat3X& world_pts);
  [[nodiscard]] static Mat4 findAbsoluteDualQuadratic(const ThreeOf<Mat3x4>& P);
  [[nodiscard]] static ThreeOf<Mat3> findCameraMatrices(const Mat4& ADQ, const ThreeOf<Mat3x4>& P);
  [[nodiscard]] static Mat4 findRectifyingHomography(const Mat4& ADQ,
                                                     const Mat3& K0,
                                                     const ThreeOf<Mat3x4>& P,
                                                     const Mat3X& world_pts);
  static void transformReconstruction(const Mat4& H, ThreeOf<Mat3x4>& P, Mat3X& world_pts);

  void recoverCameraCalibrations(const ThreeOf<Mat3x4>& P, const ThreeOf<Mat3>& K);

  void writeBackWorldPoints(const std::vector<PointId>& common_pt_ids, const Mat3X& world_pts);

  void triangulateRemainingPoints();

protected:
  std::map<CamId, CameraCalib> cameras_;
  std::map<PointId, PointData> points_;

private:
  const ThreeViewReconstructionParams& params_;
};

} // namespace calib3d
