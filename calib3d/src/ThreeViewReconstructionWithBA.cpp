// Copyright 2024 Krzysztof Wrobel

#include <calib3d/ThreeViewReconstructionWithBA.h>

#include <glog/logging.h>

namespace calib3d {

ThreeViewReconstructionWithBA::ThreeViewReconstructionWithBA(double observation_noise)
    : ThreeViewReconstruction(observation_noise), ba_problem_(observation_noise) {}

void ThreeViewReconstructionWithBA::reconstruct(CamId cam0_id,
                                                const CameraSize& cam0_size,
                                                const Observations& cam0_obs,
                                                CamId cam1_id,
                                                const CameraSize& cam1_size,
                                                const Observations& cam1_obs,
                                                CamId cam2_id,
                                                const CameraSize& cam2_size,
                                                const Observations& cam2_obs) {
  // Invoke the initial coarse three-view reconstruction
  ThreeViewReconstruction::reconstruct(
      cam0_id, cam0_size, cam0_obs, cam1_id, cam1_size, cam1_obs, cam2_id, cam2_size, cam2_obs);

  // And optimize the scene with bundle adjustment

  // Add the cameras specifying their types to set proper parameterization
  ba_problem_.addCamera(cameras_[cam0_id], BAProblem::CameraType::CAM_0);
  ba_problem_.addCamera(cameras_[cam1_id], BAProblem::CameraType::CAM_1);
  ba_problem_.addCamera(cameras_[cam2_id], BAProblem::CameraType::CAM_N);

  for (auto& [pt_id, pt_data] : points_) {
    if (!pt_data.world_pt) {
      continue;
    }

    // For points with triangulated world point, add its observations to the BA problem
    auto& world_pt = pt_data.world_pt.value();
    for (auto& [cam_id, image_pt] : pt_data.image_pts) {
      ba_problem_.addObservation(cameras_.at(cam_id), world_pt, image_pt);
    }
  }

  VLOG(1) << "Performing BA";
  ba_problem_.optimize();
}

void ThreeViewReconstructionWithBA::optimizeWithPrincipalPoint() {
  VLOG(1) << "Performing BA with variable principal point";
  ba_problem_.setPrincipalPointVariable();
  ba_problem_.optimize();
  ba_problem_.setPrincipalPointConstant();
}

} // namespace calib3d
