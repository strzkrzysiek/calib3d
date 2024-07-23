// Copyright 2024 Krzysztof Wrobel

#include <calib3d/NViewReconstruction.h>
#include <calib3d/ThreeViewReconstruction.h>
#include <calib3d/ThreeViewReconstructionWithBA.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <calib3d/Dataset.h>

using namespace calib3d;

class ReconstructionTestFixture : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(dataset.loadFromJson(DATASET_FILE_PATH, true));
    ASSERT_FALSE(dataset.cameras.empty());
    ASSERT_FALSE(dataset.world_points.empty());
  }

  template <class RECONSTRUCTION>
  void verifyCameras(const RECONSTRUCTION& reconstruction) {
    const auto& reconstructed_cameras = reconstruction.getCameras();
    ASSERT_FALSE(reconstructed_cameras.empty());

    double max_f_err = -1.0;
    double max_principal_point_err = -1.0;
    double max_cam_t_err = -1.0;
    double max_cam_R_err_in_degrees = -1.0;

    for (const auto& [cam_id, reconstructed_calib] : reconstructed_cameras) {
      const auto& true_calib = dataset.cameras.at(cam_id).calib;

      const auto& recon_intr = reconstructed_calib.intrinsics;
      const auto& true_intr = true_calib.intrinsics;

      double f_err = std::abs(recon_intr.focal_length - true_intr.focal_length);
      EXPECT_NEAR(recon_intr.focal_length, true_intr.focal_length, f_abs_tolerance_);
      max_f_err = std::max(f_err, max_f_err);

      Vec2 principal_point_err = (recon_intr.principal_point - true_intr.principal_point).cwiseAbs();
      EXPECT_NEAR(recon_intr.principal_point[0], true_intr.principal_point[0], principal_point_abs_tolerance_);
      EXPECT_NEAR(recon_intr.principal_point[1], true_intr.principal_point[1], principal_point_abs_tolerance_);
      max_principal_point_err = std::max(principal_point_err.maxCoeff(), max_principal_point_err);

      const auto& recon_w2c = reconstructed_calib.world2cam;
      const auto& true_w2c = true_calib.world2cam;

      Vec3 recon_cam_in_world = recon_w2c.inverse().translation();
      Vec3 true_cam_in_world = true_w2c.inverse().translation();
      Vec3 cam_t_err = (recon_cam_in_world - true_cam_in_world).cwiseAbs();
      EXPECT_NEAR(recon_cam_in_world[0], true_cam_in_world[0], cam_t_abs_tolerance_) << "Cam: " << cam_id;
      EXPECT_NEAR(recon_cam_in_world[1], true_cam_in_world[1], cam_t_abs_tolerance_) << "Cam: " << cam_id;
      EXPECT_NEAR(recon_cam_in_world[2], true_cam_in_world[2], cam_t_abs_tolerance_) << "Cam: " << cam_id;
      max_cam_t_err = std::max(cam_t_err.maxCoeff(), max_cam_t_err);

      const auto recon_cam2true_cam = true_w2c.so3() * recon_w2c.so3().inverse();
      const auto cam_R_err_in_degrees = std::abs(recon_cam2true_cam.logAndTheta().theta / M_PI * 180.0);
      EXPECT_LT(cam_R_err_in_degrees, cam_R_abs_tolerance_in_degrees_);
      max_cam_R_err_in_degrees = std::max(cam_R_err_in_degrees, max_cam_R_err_in_degrees);
    }

    LOG(INFO) << "Max focal length err: " << max_f_err;
    LOG(INFO) << "Max principal point error: " << max_principal_point_err;
    LOG(INFO) << "Max camera translation error: " << max_cam_t_err;
    LOG(INFO) << "Max camera rotation error in degrees: " << max_cam_R_err_in_degrees;
  }

  template <class RECONSTRUCTION>
  void verifyWorldPoints(const RECONSTRUCTION& reconstruction) {
    const auto& reconstructed_points = reconstruction.getPoints();
    ASSERT_FALSE(reconstructed_points.empty());

    double max_err = -1.0;

    for (const auto& [pt_id, reconstructed_pt_data] : reconstructed_points) {
      if (!reconstructed_pt_data.world_pt) {
        continue;
      }

      const auto& true_world_pt = dataset.world_points.at(pt_id);
      const auto& recon_world_pt = reconstructed_pt_data.world_pt.value();

      Vec3 world_pt_err = (recon_world_pt - true_world_pt).cwiseAbs();
      EXPECT_NEAR(recon_world_pt[0], true_world_pt[0], world_pt_abs_tolerance_) << "Pt: " << pt_id;
      EXPECT_NEAR(recon_world_pt[1], true_world_pt[1], world_pt_abs_tolerance_) << "Pt: " << pt_id;
      EXPECT_NEAR(recon_world_pt[2], true_world_pt[2], world_pt_abs_tolerance_) << "Pt: " << pt_id;
      max_err = std::max(world_pt_err.maxCoeff(), max_err);
    }

    LOG(INFO) << "Max world point position error: " << max_err;
  }

  Dataset dataset;

  double f_abs_tolerance_ = 1.0;
  double principal_point_abs_tolerance_ = 1.0;
  double cam_t_abs_tolerance_ = 0.001;
  double cam_R_abs_tolerance_in_degrees_ = 0.1;
  double world_pt_abs_tolerance_ = 0.001;
};

TEST_F(ReconstructionTestFixture, ThreeView) {
  const double observation_noise = 2.0;

  f_abs_tolerance_ = 1.0;
  principal_point_abs_tolerance_ = 1.0;
  cam_t_abs_tolerance_ = 0.0005;
  cam_R_abs_tolerance_in_degrees_ = 0.05;
  world_pt_abs_tolerance_ = 0.001;

  ThreeViewReconstruction reconstruction(observation_noise);

  reconstruction.reconstruct(0,
                             dataset.cameras[0].calib.size,
                             dataset.cameras[0].observations,
                             1,
                             dataset.cameras[1].calib.size,
                             dataset.cameras[1].observations,
                             2,
                             dataset.cameras[2].calib.size,
                             dataset.cameras[2].observations);

  verifyCameras(reconstruction);
  verifyWorldPoints(reconstruction);
}

TEST_F(ReconstructionTestFixture, DISABLED_ThreeViewNoisyObservations) {
  const double observation_noise = 2.0;
  const size_t seed = 7;
  dataset.addObservationNoise(observation_noise, seed);

  f_abs_tolerance_ = 100.;
  principal_point_abs_tolerance_ = 10.0;
  cam_t_abs_tolerance_ = 0.25;
  cam_R_abs_tolerance_in_degrees_ = 5.0;
  world_pt_abs_tolerance_ = 0.25;

  ThreeViewReconstruction reconstruction(observation_noise);

  reconstruction.reconstruct(0,
                             dataset.cameras[0].calib.size,
                             dataset.cameras[0].observations,
                             1,
                             dataset.cameras[1].calib.size,
                             dataset.cameras[1].observations,
                             2,
                             dataset.cameras[2].calib.size,
                             dataset.cameras[2].observations);

  verifyCameras(reconstruction);
  verifyWorldPoints(reconstruction);
}

TEST_F(ReconstructionTestFixture, ThreeViewWithBA) {
  const double observation_noise = 2.0;

  f_abs_tolerance_ = 1.0;
  principal_point_abs_tolerance_ = 1.0;
  cam_t_abs_tolerance_ = 0.0005;
  cam_R_abs_tolerance_in_degrees_ = 0.05;
  world_pt_abs_tolerance_ = 0.001;

  ThreeViewReconstructionWithBA reconstruction(observation_noise);

  reconstruction.reconstruct(0,
                             dataset.cameras[0].calib.size,
                             dataset.cameras[0].observations,
                             1,
                             dataset.cameras[1].calib.size,
                             dataset.cameras[1].observations,
                             2,
                             dataset.cameras[2].calib.size,
                             dataset.cameras[2].observations);

  reconstruction.optimizeWithPrincipalPoint();

  verifyCameras(reconstruction);
  verifyWorldPoints(reconstruction);
}

TEST_F(ReconstructionTestFixture, ThreeViewWithBANoisyObservations) {
  const double observation_noise = 2.0;
  const size_t seed = 7;
  dataset.addObservationNoise(observation_noise, seed);

  f_abs_tolerance_ = 100.;
  principal_point_abs_tolerance_ = 1.0;
  cam_t_abs_tolerance_ = 0.05;
  cam_R_abs_tolerance_in_degrees_ = 1.0;
  world_pt_abs_tolerance_ = 0.05;

  ThreeViewReconstructionWithBA reconstruction(observation_noise);

  reconstruction.reconstruct(0,
                             dataset.cameras[0].calib.size,
                             dataset.cameras[0].observations,
                             1,
                             dataset.cameras[1].calib.size,
                             dataset.cameras[1].observations,
                             2,
                             dataset.cameras[2].calib.size,
                             dataset.cameras[2].observations);

  verifyCameras(reconstruction);
  verifyWorldPoints(reconstruction);
}

TEST_F(ReconstructionTestFixture, NView) {
  const double observation_noise = 2.0;

  f_abs_tolerance_ = 1e-3;
  principal_point_abs_tolerance_ = 1e-4;
  cam_t_abs_tolerance_ = 1e-6;
  cam_R_abs_tolerance_in_degrees_ = 1e-5;
  world_pt_abs_tolerance_ = 1e-7;

  NViewReconstruction reconstruction(observation_noise);

  reconstruction.initReconstruction(0,
                                    dataset.cameras[0].calib.size,
                                    dataset.cameras[0].observations,
                                    1,
                                    dataset.cameras[1].calib.size,
                                    dataset.cameras[1].observations,
                                    2,
                                    dataset.cameras[2].calib.size,
                                    dataset.cameras[2].observations);

  for (const auto& [cam_id, cam_data] : dataset.cameras) {
    if (cam_id < 3) {
      continue;
    }

    reconstruction.addNewCamera(cam_id, cam_data.calib.size, cam_data.observations);
  }

  reconstruction.optimizeWithPrincipalPoint();

  verifyCameras(reconstruction);
  verifyWorldPoints(reconstruction);
}

TEST_F(ReconstructionTestFixture, NViewNoisyObservations) {
  const double observation_noise = 2.0;
  const size_t seed = 7;
  dataset.addObservationNoise(observation_noise, seed);

  f_abs_tolerance_ = 50.;
  principal_point_abs_tolerance_ = 1.0;
  cam_t_abs_tolerance_ = 0.1;
  cam_R_abs_tolerance_in_degrees_ = 2.0;
  world_pt_abs_tolerance_ = 0.01;

  NViewReconstruction reconstruction(observation_noise);

  reconstruction.initReconstruction(0,
                                    dataset.cameras[0].calib.size,
                                    dataset.cameras[0].observations,
                                    1,
                                    dataset.cameras[1].calib.size,
                                    dataset.cameras[1].observations,
                                    2,
                                    dataset.cameras[2].calib.size,
                                    dataset.cameras[2].observations);

  for (const auto& [cam_id, cam_data] : dataset.cameras) {
    if (cam_id < 3) {
      continue;
    }

    reconstruction.addNewCamera(cam_id, cam_data.calib.size, cam_data.observations);
  }

  verifyCameras(reconstruction);
  verifyWorldPoints(reconstruction);
}

TEST_F(ReconstructionTestFixture, NViewNoisyObservationsWithOutliers) {
  const double observation_noise = 2.0;
  const size_t seed = 7;
  const double inlier_prob = 0.9;
  dataset.addObservationNoise(observation_noise, seed);
  dataset.addObservationOutliers(inlier_prob, seed);

  f_abs_tolerance_ = 50.;
  principal_point_abs_tolerance_ = 1.0;
  cam_t_abs_tolerance_ = 0.1;
  cam_R_abs_tolerance_in_degrees_ = 2.0;
  world_pt_abs_tolerance_ = 0.05;

  NViewReconstruction reconstruction(observation_noise);

  reconstruction.initReconstruction(0,
                                    dataset.cameras[0].calib.size,
                                    dataset.cameras[0].observations,
                                    1,
                                    dataset.cameras[1].calib.size,
                                    dataset.cameras[1].observations,
                                    2,
                                    dataset.cameras[2].calib.size,
                                    dataset.cameras[2].observations);

  for (const auto& [cam_id, cam_data] : dataset.cameras) {
    if (cam_id < 3) {
      continue;
    }

    reconstruction.addNewCamera(cam_id, cam_data.calib.size, cam_data.observations);
  }

  verifyCameras(reconstruction);
  verifyWorldPoints(reconstruction);
}
