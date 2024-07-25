// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <calib3d/BAProblem.h>
#include <calib3d/ThreeViewReconstruction.h>

namespace calib3d {

// Derived class for three-view reconstruction with bundle adjustment
class ThreeViewReconstructionWithBA : protected ThreeViewReconstruction {
public:
  // Inherit base class functions
  using ThreeViewReconstruction::getCameras;
  using ThreeViewReconstruction::getPoints;

  // Constructor to initialize with observation noise
  explicit ThreeViewReconstructionWithBA(double observation_noise);

  // Main reconstruction function with bundle adjustment
  void reconstruct(CamId cam0_id,
                   const CameraSize& cam0_size,
                   const Observations& cam0_obs,
                   CamId cam1_id,
                   const CameraSize& cam1_size,
                   const Observations& cam1_obs,
                   CamId cam2_id,
                   const CameraSize& cam2_size,
                   const Observations& cam2_obs) override;

  // Optimizes the reconstruction with principal point adjustment
  // This is an optional method to call if all the observations are accurate and an exact principal point
  // should be calculated. It should be called as a final step of whole optimization procedure.
  // Otherwise, it is going to spoil the reconstruction.
  void optimizeWithPrincipalPoint();

protected:
  BAProblem ba_problem_; // Bundle adjustment problem instance
};

} // namespace calib3d
