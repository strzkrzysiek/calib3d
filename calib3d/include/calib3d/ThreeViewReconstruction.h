// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <Eigen/Core>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include <calib3d/types.h>

namespace calib3d {

// Class for reconstructing a 3D scene from three views
class ThreeViewReconstruction {
public:
  // Constructor to initialize with observation noise
  explicit ThreeViewReconstruction(double observation_noise);

  // Gets the calibrated cameras
  [[nodiscard]] const std::map<CamId, CameraCalib>& getCameras() const;

  // Gets the reconstructed 3D points' info
  [[nodiscard]] const std::map<PointId, PointData>& getPoints() const;

  // Main reconstruction function from three views
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
  // Inserts camera data into the reconstruction system data structures
  void insertCameraData(CamId cam_id, const CameraSize& cam_size, const Observations& cam_obs);

  // Performs the reconstruction process for three views
  void performThreeViewReconstruction(const ThreeOf<CamId>& cam_ids);

  // Prepares common observations from three cameras
  [[nodiscard]] std::pair<std::vector<PointId>, ThreeOf<Mat2X>> prepareCommonObservations(
      const ThreeOf<CamId>& cam_ids) const;

  // Performs projective reconstruction
  [[nodiscard]] std::pair<ThreeOf<Mat3x4>, Mat3X> performProjectiveReconstruction(
      const ThreeOf<Mat2X>& image_pts) const;

  // Gets projection matrix from fundamental matrix
  [[nodiscard]] static Mat3x4 getProjectionMatrixFromFundamentalMatrix(const Mat3& F);

  // Performs metric rectification on the reconstructed points and cameras
  static ThreeOf<Mat3> performMetricRectification(ThreeOf<Mat3x4>& P, Mat3X& world_pts);
  [[nodiscard]] static Mat4 findAbsoluteDualQuadratic(const ThreeOf<Mat3x4>& P);
  [[nodiscard]] static ThreeOf<Mat3> findCameraMatrices(const Mat4& ADQ, const ThreeOf<Mat3x4>& P);
  [[nodiscard]] static Mat3 findCameraMatrix(const Mat4& ADQ, const Mat3x4& P);
  [[nodiscard]] static Mat4 findRectifyingHomography(const Mat4& ADQ, const Mat3& K0, const Mat3X& world_pts);
  static void transformReconstruction(const Mat4& H, ThreeOf<Mat3x4>& P, Mat3X& world_pts);

  // Recovers camera calibrations from projection matrices and intrinsic parameters
  void recoverCameraCalibrations(const ThreeOf<CamId>& cam_ids, const ThreeOf<Mat3x4>& P, const ThreeOf<Mat3>& K);
  static void recoverCameraCalibration(const Mat3x4& P, const Mat3& K, CameraCalib& calib);

  // Refines camera calibrations using an iterative optimization method
  void refineCameraCalibrations(const ThreeOf<CamId>& cam_ids,
                                const ThreeOf<Mat3x4>& P,
                                const Mat3X& world_pts,
                                const ThreeOf<Mat2X>& image_pts);
  void refineCameraCalibration(const Mat3x4& P,
                               const Mat3X& world_pts,
                               const Mat2X& image_pts,
                               CameraCalib& calib) const;

  // Fixes the rotation and scale of the reconstructed points
  void fixRotationAndScale(const ThreeOf<CamId>& cam_ids, Mat3X& world_pts);

  // Writes back the world points to the internal data structures
  void writeBackWorldPoints(const std::vector<PointId>& common_pt_ids, const Mat3X& world_pts);

  // Identifies outliers in the reconstructed points
  [[nodiscard]] std::set<PointId> identifyOutliers(const ThreeOf<CamId>& cam_ids,
                                                   const Mat3X& world_pts,
                                                   const ThreeOf<Mat2X>& image_pts,
                                                   const std::vector<PointId>& pt_ids) const;
  [[nodiscard]] std::set<PointId> identifyOutliers(CamId cam_id,
                                                   const Mat3X& world_pts,
                                                   const Mat2X& image_pts,
                                                   const std::vector<PointId>& pt_ids) const;

  // Retriangulates outliers to improve reconstruction accuracy
  void retriangulateOutliers(const std::set<PointId>& outlier_ids);

  // Triangulates remaining points that were not initially reconstructed
  void triangulateRemainingPoints();

protected:
  std::map<CamId, CameraCalib> cameras_;  // Map of camera calibrations
  std::map<PointId, PointData> points_;   // Map of reconstructed 3D points
  const double observation_noise_;        // Observation noise level
  const double percentile_95_;            // 95th percentile threshold for identifying outliers
  const double ransac_thr_;               // RANSAC threshold for inlier selection
  const double ransac_confidence_;        // RANSAC confidence level
  const size_t ransac_max_iters_;         // Maximum RANSAC iterations
};

} // namespace calib3d
