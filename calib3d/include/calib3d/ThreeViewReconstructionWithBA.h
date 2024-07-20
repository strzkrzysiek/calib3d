#pragma once

#include <calib3d/BAProblem.h>
#include <calib3d/ThreeViewReconstruction.h>

namespace calib3d {

class ThreeViewReconstructionWithBA : protected ThreeViewReconstruction {
public:
  using ThreeViewReconstruction::getCameras;
  using ThreeViewReconstruction::getPoints;

  explicit ThreeViewReconstructionWithBA(double observation_noise);

  void reconstruct(CamId cam0_id,
                   const CameraSize& cam0_size,
                   const Observations& cam0_obs,
                   CamId cam1_id,
                   const CameraSize& cam1_size,
                   const Observations& cam1_obs,
                   CamId cam2_id,
                   const CameraSize& cam2_size,
                   const Observations& cam2_obs) override;

protected:
  BAProblem ba_problem_;
};

} // namespace calib3d
