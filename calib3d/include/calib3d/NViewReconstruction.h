// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <calib3d/ThreeViewReconstructionWithBA.h>

namespace calib3d {

class NViewReconstruction : protected ThreeViewReconstructionWithBA {
public:
  using ThreeViewReconstructionWithBA::getCameras;
  using ThreeViewReconstructionWithBA::getPoints;
  using ThreeViewReconstructionWithBA::optimizeWithPrincipalPoint;

  explicit NViewReconstruction(double observation_noise);

  void initReconstruction(CamId cam0_id,
                          const CameraSize& cam0_size,
                          const Observations& cam0_obs,
                          CamId cam1_id,
                          const CameraSize& cam1_size,
                          const Observations& cam1_obs,
                          CamId cam2_id,
                          const CameraSize& cam2_size,
                          const Observations& cam2_obs);

  void addNewCamera(CamId cam_id, const CameraSize& cam_size, const Observations& cam_obs);

protected:
  void initializeNewCamera(CamId cam_id, const CameraSize& cam_size, const Observations& cam_obs);
  std::tuple<Mat3X, Mat2X, std::vector<PointId>> prepare3D2DCorrespondences(const CameraSize& cam_size,
                                                                            const Observations& cam_obs);

  void optimizeNewCamera(CamId cam_id, const Observations& cam_obs);

  const Mat4 metric_ADQ_;
};

} // namespace calib3d
