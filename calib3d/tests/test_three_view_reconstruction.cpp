#include <calib3d/ThreeViewReconstruction.h>
#include <calib3d/ThreeViewReconstructionWithBA.h>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <calib3d/dataset.h>

using namespace calib3d;

class ThreeViewReconstructionTestFixture : public ::testing::Test {
protected:
  Dataset dataset;

  void SetUp() override {
    ASSERT_TRUE(loadJsonDataset(DATASET_FILE_PATH, dataset));
    ASSERT_FALSE(dataset.cameras.empty());
    ASSERT_FALSE(dataset.world_points.empty());
  }
};

TEST_F(ThreeViewReconstructionTestFixture, Reconstruction) {
  const double observation_noise = 3.0;

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

  const auto& reconstructed_points = reconstruction.getPoints();
  for (const auto& [pt_id, pt_data] : reconstructed_points) {
    if (!pt_data.world_pt) {
      continue;
    }
    EXPECT_TRUE(pt_data.world_pt.value().isApprox(dataset.world_points[pt_id], 1e-3));
  }

  for (CamId cam_id : {0, 1, 2}) {
    const auto& true_calib = dataset.cameras.at(cam_id).calib;
    const auto& reconstructed_calib = reconstruction.getCameras().at(cam_id);

    EXPECT_NEAR(true_calib.intrinsics.focal_length, reconstructed_calib.intrinsics.focal_length, 1.0);

    Vec3 reconstructed_cam_in_world = reconstructed_calib.world2cam.inverse().translation();
    Vec3 true_cam_in_world = true_calib.world2cam.inverse().translation();
    EXPECT_TRUE(true_cam_in_world.isApprox(reconstructed_cam_in_world, 1e-3));

    auto reconstructed_cam2true_cam = true_calib.world2cam.so3() * reconstructed_calib.world2cam.so3().inverse();
    double error_angle = reconstructed_cam2true_cam.logAndTheta().theta;
    EXPECT_NEAR(error_angle / M_PI * 180, 0., 1.0);
  }
}

TEST_F(ThreeViewReconstructionTestFixture, ReconstructionWithBA) {
  const double observation_noise = 3.0;

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

  const auto& reconstructed_points = reconstruction.getPoints();
  for (const auto& [pt_id, pt_data] : reconstructed_points) {
    if (!pt_data.world_pt) {
      continue;
    }
    EXPECT_TRUE(pt_data.world_pt.value().isApprox(dataset.world_points[pt_id], 1e-3));
  }

  for (CamId cam_id : {0, 1, 2}) {
    const auto& true_calib = dataset.cameras.at(cam_id).calib;
    const auto& reconstructed_calib = reconstruction.getCameras().at(cam_id);

    EXPECT_NEAR(true_calib.intrinsics.focal_length, reconstructed_calib.intrinsics.focal_length, 1.0);

    Vec3 reconstructed_cam_in_world = reconstructed_calib.world2cam.inverse().translation();
    Vec3 true_cam_in_world = true_calib.world2cam.inverse().translation();
    EXPECT_TRUE(true_cam_in_world.isApprox(reconstructed_cam_in_world, 1e-3));

    auto reconstructed_cam2true_cam = true_calib.world2cam.so3() * reconstructed_calib.world2cam.so3().inverse();
    double error_angle = reconstructed_cam2true_cam.logAndTheta().theta;
    EXPECT_NEAR(error_angle / M_PI * 180, 0., 1.0);
  }
}
