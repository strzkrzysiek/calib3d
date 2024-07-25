// Copyright 2024 Krzysztof Wrobel

#include <calib3d/NViewReconstruction.h>

#include <calib3d/calib_utils.h>

namespace calib3d {

NViewReconstruction::NViewReconstruction(double observation_noise)
    : ThreeViewReconstructionWithBA(observation_noise), metric_ADQ_(Vec4(1., 1., 1., 0.).asDiagonal()) {}

void NViewReconstruction::initReconstruction(CamId cam0_id,
                                             const CameraSize& cam0_size,
                                             const Observations& cam0_obs,
                                             CamId cam1_id,
                                             const CameraSize& cam1_size,
                                             const Observations& cam1_obs,
                                             CamId cam2_id,
                                             const CameraSize& cam2_size,
                                             const Observations& cam2_obs) {
  ThreeViewReconstructionWithBA::reconstruct(
      cam0_id, cam0_size, cam0_obs, cam1_id, cam1_size, cam1_obs, cam2_id, cam2_size, cam2_obs);
}

void NViewReconstruction::addNewCamera(CamId cam_id, const CameraSize& cam_size, const Observations& cam_obs) {
  // Insert new camera data into the internal data structures
  insertCameraData(cam_id, cam_size, cam_obs);

  // Calculate the coarse calibration for the new camera
  initializeNewCamera(cam_id, cam_size, cam_obs);

  // Triangulate new points
  triangulateRemainingPoints();

  // Run bundle adjustment to refine the solution
  optimizeNewCamera(cam_id, cam_obs);
}

void NViewReconstruction::initializeNewCamera(CamId cam_id, const CameraSize& cam_size, const Observations& cam_obs) {
  VLOG(1) << "Adding camera (ID: " << cam_id << ")";

  // Prepare world and image points as contiguous arrays and move the image points by half of the camera size
  // to perform the calculations as if the principal point was at (0, 0)
  const auto [world_pts, image_pts, pt_ids] = prepare3D2DCorrespondences(cam_size, cam_obs);

  // Find the projection matrix using 3D-2D correspondences
  Mat3x4 P = findProjectionMatrixRansac(world_pts, image_pts, ransac_thr_, ransac_confidence_, ransac_max_iters_);
  VLOG(2) << "Projection matrix:\n" << P;

  // Recover the camera matrix from the projection matrix using the metric ADQ
  Mat3 K = findCameraMatrix(metric_ADQ_, P);

  auto& calib = cameras_[cam_id];
  // Recover coarse camera extrinsic params
  VLOG(1) << "Recovering camera params";
  recoverCameraCalibration(P, K, calib);
  // Refine camera extrinsic params and the focal length with iterative non-linear least squares method
  VLOG(1) << "Refining camera params";
  refineCameraCalibration(P, world_pts, image_pts, calib);

  // Identify outliers in the set of current camera observations
  // Simply, if there is a big reprojection error for a particular 3D-2D pair, consider it as an outlier
  auto outlier_ids = identifyOutliers(cam_id, world_pts, image_pts, pt_ids);
  // and retriangulate it
  retriangulateOutliers(outlier_ids);
}

std::tuple<Mat3X, Mat2X, std::vector<PointId>> NViewReconstruction::prepare3D2DCorrespondences(
    const CameraSize& cam_size, const Observations& cam_obs) {
  Mat3X world_pts(3, cam_obs.size());
  Mat2X image_pts(2, cam_obs.size());
  std::vector<PointId> pt_ids;
  pt_ids.reserve(cam_obs.size());

  Vec2 principal_pt_offset = cam_size.cast<double>() / 2.;

  for (const auto& [pt_id, image_pt] : cam_obs) {
    const auto& pt_data = points_.at(pt_id);
    if (!pt_data.world_pt) {
      continue;
    }

    world_pts.col(pt_ids.size()) = pt_data.world_pt.value();
    image_pts.col(pt_ids.size()) = image_pt - principal_pt_offset;
    pt_ids.push_back(pt_id);
  }

  world_pts.conservativeResize(Eigen::NoChange, pt_ids.size());
  image_pts.conservativeResize(Eigen::NoChange, pt_ids.size());

  return {world_pts, image_pts, pt_ids};
}

void NViewReconstruction::optimizeNewCamera(CamId cam_id, const Observations& cam_obs) {
  auto& calib = cameras_.at(cam_id);
  ba_problem_.addCamera(calib, BAProblem::CameraType::CAM_N);

  // Add new observations to the optimization problem
  for (const auto& [pt_id, image_pt] : cam_obs) {
    auto& pt_data = points_.at(pt_id);
    if (!pt_data.world_pt) {
      continue;
    }

    auto& world_pt = pt_data.world_pt.value();
    const size_t n_obs = pt_data.image_pts.size();
    CHECK_GE(n_obs, 2);

    if (n_obs >= 3) {
      // If there are at least three observations, this means that the point have been already triangulated before
      // and already the other observations have been already added to the problem

      ba_problem_.addObservation(calib, world_pt, image_pt);
    } else {
      // If there are only two observations, add them both as this point has been just triangulated
      for (auto& [obs_cam_id, obs_image_pt] : pt_data.image_pts) {
        ba_problem_.addObservation(cameras_.at(obs_cam_id), world_pt, obs_image_pt);
      }
    }
  }

  VLOG(1) << "Performing BA";
  ba_problem_.optimize();
}

} // namespace calib3d
