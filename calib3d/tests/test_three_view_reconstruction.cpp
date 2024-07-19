#include <calib3d/ThreeViewReconstruction.h>

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
  ThreeViewReconstructionParams params;
  params.observation_noise = 3.0;

  ThreeViewReconstruction reconstruction(params);

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

    EXPECT_NEAR(true_calib.intrinsics.f, reconstructed_calib.intrinsics.f, 1.0);
    Vec3 reconstructed_cam_in_world = reconstructed_calib.extrinsics.cam_in_world_pos();
    Vec3 true_cam_in_world = true_calib.extrinsics.cam_in_world_pos();
    EXPECT_TRUE(true_cam_in_world.isApprox(reconstructed_cam_in_world, 1e-3));
    Quat reconstructed_world2cam_rot = reconstructed_calib.extrinsics.world2cam_rot;
    Quat true_world2cam_rot = true_calib.extrinsics.world2cam_rot;
    double angle_between_orientations = true_world2cam_rot.angularDistance(reconstructed_world2cam_rot);
    EXPECT_NEAR(angle_between_orientations / M_PI * 180, 0., 1.0);
  }
}
