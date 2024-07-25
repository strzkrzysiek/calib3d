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
//
// It calculates a coarse estimate of world point coordinates and camera intrinsic and extrinsic parameters.
// The algorithm relies heavily on the assumptions that the cameras:
// * can be defined with a pinhole camera model
// * have no distortion
// * have square pixels which implies zero skew and that fx = fy.
// * have their principal points relatively in the center of the image
// This means that the camera matrix K (for image size [sx, sy]) is defined as follows:
//     |   f     0   ~sx/2 |
// K = |   0     f   ~sy/2 |
//     |   0     0     1   |
// There is no requirement that the cameras have the same intrinsic parameters.
//
// The algorithm implements one of the autocalibration methods using the guidelines defined
// in Section 19.3: Calibration using the absolute dual quadric in Hartley and Zisserman 2003.
// and specifically in Example 19.5. Linear solution for variable focal length
//
// This class is a base class for finer reconstruction methods defined within this library
class ThreeViewReconstruction {
public:
  // Constructor to initialize with an estimated observation noise
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

  // Prepares common observations from three cameras and puts them into a contiguous matrix
  // The coordinates of the points are shifted by half of the image size
  // so that the assumed principal point is in [0, 0]
  [[nodiscard]] std::pair<std::vector<PointId>, ThreeOf<Mat2X>> prepareCommonObservations(
      const ThreeOf<CamId>& cam_ids) const;

  // Performs projective reconstruction
  // It returns the projection matrices for each camera together with triangulated set of 3D points
  // Projective reconstruction means that the ambiguity of reconstruction
  // is expressed by an arbitrary projective transformation as explained in
  // Section 10.2 Reconstruction ambiguity in Hartley and Zisserman (2003)
  [[nodiscard]] std::pair<ThreeOf<Mat3x4>, Mat3X> performProjectiveReconstruction(
      const ThreeOf<Mat2X>& image_pts) const;

  // Gets projection matrix from fundamental matrix
  // It returns the projection matrix of the second camera assuming that
  // the projection matrix of the first one is P = [I | 0]
  // See: Result 9.14. in Hartley and Zisserman (2003)
  [[nodiscard]] static Mat3x4 getProjectionMatrixFromFundamentalMatrix(const Mat3& F);

  // Performs metric rectification on the reconstructed points and cameras
  // This means that given a projective reconstruction it finds and applies a transformation
  // that rectifies the reconstruction. Then, the ambiguity of the resulting reconstruction is expressed
  // by arbitrary similarity transform. This means that all angles and distance ratios are recovered.
  // As a result of metric rectification, the camera matrices are calculated.
  // The rectification method finds an absolute dual quadric which is said to be invariant
  // under any similarity transformation and in metric space is equal to Vec4(1., 1., 1., 0).asDiagonal()
  // See: Section 19.3: Calibration using the absolute dual quadric in Hartley and Zisserman (2003)
  static ThreeOf<Mat3> performMetricRectification(ThreeOf<Mat3x4>& P, Mat3X& world_pts);

  // Finds the ADQ (absolute dual quadric) in the space of the projective reconstruction.
  // It uses the assumptions on the camera matrix as constraints to solve for ADQ.
  // See: Section 19.3.1 Linear solutions for ADQ from a set of images in Hartley and Zisserman (2003)
  [[nodiscard]] static Mat4 findAbsoluteDualQuadric(const ThreeOf<Mat3x4>& P);

  // Given the absolute dual quadric and projection matrix of each camera
  [[nodiscard]] static ThreeOf<Mat3> findCameraMatrices(const Mat4& ADQ, const ThreeOf<Mat3x4>& P);
  [[nodiscard]] static Mat3 findCameraMatrix(const Mat4& ADQ, const Mat3x4& P);

  // Given the absolute dual quadric, camera matrix of the first camera and world points,
  // find the rectifying transformation as described above
  // See: Result 19.1. in Hartley and Zisserman (2003)
  [[nodiscard]] static Mat4 findRectifyingHomography(const Mat4& ADQ, const Mat3& K0, const Mat3X& world_pts);

  // Transforms the projective reconstruction [world_pts, P_i] into metric reconstruction
  // with the calculated rectifying homography
  static void transformReconstruction(const Mat4& H, ThreeOf<Mat3x4>& P, Mat3X& world_pts);

  // Recovers camera calibrations from projection matrices and intrinsic parameters
  // See: Section 6.2.4 Decomposition of the camera matrix in Hartley and Zisserman (2003)
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

  // Fixes the orientation and scale of the reconstructed points
  // Without any additional information, there is no possibility to recover the scale,
  // origin and orientation of the metric reconstruction.
  // We fix the origin and orientation by placing the first camera at the origin of the coordinate system
  // and the scale is fixed by positioning the second camera at the unit sphere from the origin
  // so that the distance between the first and the second camera is one.
  void fixRotationAndScale(const ThreeOf<CamId>& cam_ids, Mat3X& world_pts);

  // Writes back the world points to the internal data structures
  void writeBackWorldPoints(const std::vector<PointId>& common_pt_ids, const Mat3X& world_pts);

  // Identifies outliers in the reconstructed points
  // Constructs a set of point IDs that are considered as outliers if in any of the camera
  // the reprojection error exceeds the threshold
  [[nodiscard]] std::set<PointId> identifyOutliers(const ThreeOf<CamId>& cam_ids,
                                                   const Mat3X& world_pts,
                                                   const ThreeOf<Mat2X>& image_pts,
                                                   const std::vector<PointId>& pt_ids) const;
  [[nodiscard]] std::set<PointId> identifyOutliers(CamId cam_id,
                                                   const Mat3X& world_pts,
                                                   const Mat2X& image_pts,
                                                   const std::vector<PointId>& pt_ids) const;

  // Retriangulates outliers with a RANSAC method to improve reconstruction accuracy
  void retriangulateOutliers(const std::set<PointId>& outlier_ids);

  // Triangulates remaining points that were not initially reconstructed
  // In case of the 3-view reconstruction, these are the points visible only by two cameras
  void triangulateRemainingPoints();

protected:
  std::map<CamId, CameraCalib> cameras_; // Map of camera calibrations
  std::map<PointId, PointData> points_;  // Map of reconstructed 3D points together with their observations
  const double observation_noise_;       // Observation noise level
  const double percentile_95_;           // 95th percentile threshold for identifying outliers
  const double ransac_thr_;              // RANSAC threshold for inlier selection
  const double ransac_confidence_;       // RANSAC confidence level
  const size_t ransac_max_iters_;        // Maximum RANSAC iterations
};

} // namespace calib3d
