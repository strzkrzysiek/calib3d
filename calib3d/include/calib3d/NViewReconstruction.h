// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <calib3d/ThreeViewReconstructionWithBA.h>

namespace calib3d {

// Class for reconstructing a 3D scene from multiple views
class NViewReconstruction : protected ThreeViewReconstructionWithBA {
public:
  // Inherit base class functions
  using ThreeViewReconstructionWithBA::getCameras;
  using ThreeViewReconstructionWithBA::getPoints;
  using ThreeViewReconstructionWithBA::optimizeWithPrincipalPoint;

  // Constructor to initialize with observation noise
  explicit NViewReconstruction(double observation_noise);

  // Initializes the reconstruction with three initial cameras
  void initReconstruction(CamId cam0_id,
                          const CameraSize& cam0_size,
                          const Observations& cam0_obs,
                          CamId cam1_id,
                          const CameraSize& cam1_size,
                          const Observations& cam1_obs,
                          CamId cam2_id,
                          const CameraSize& cam2_size,
                          const Observations& cam2_obs);

  // Adds a new camera to the reconstruction
  void addNewCamera(CamId cam_id, const CameraSize& cam_size, const Observations& cam_obs);

protected:
  // Initializes data for a new camera
  void initializeNewCamera(CamId cam_id, const CameraSize& cam_size, const Observations& cam_obs);

  // Prepares 3D to 2D correspondences for a new camera
  std::tuple<Mat3X, Mat2X, std::vector<PointId>> prepare3D2DCorrespondences(const CameraSize& cam_size,
                                                                            const Observations& cam_obs);

  // Optimizes the parameters for a new camera
  void optimizeNewCamera(CamId cam_id, const Observations& cam_obs);

  const Mat4 metric_ADQ_;  // Absolute dual quadratic matrix for metric rectification
};

} // namespace calib3d
