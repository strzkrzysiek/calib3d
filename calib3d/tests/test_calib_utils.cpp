// Copyright 2024 Krzysztof Wrobel

#include <calib3d/calib_utils.h>

#include <Eigen/QR>
#include <Eigen/SVD>
#include <gtest/gtest.h>

#include <calib3d/Dataset.h>

using namespace calib3d;

class CalibUtilsTestFixture : public ::testing::Test {
protected:
  Dataset dataset;

  Mat3 K0, K1;
  Mat4 world2cam0, world2cam1;
  Mat3x4 P0, P1;
  Mat3 F01;

  std::vector<size_t> common_pt_ids;
  Mat2X image_pts0, image_pts1;
  Mat3X world_pts;

  void SetUp() override {
    ASSERT_TRUE(dataset.loadFromJson(DATASET_FILE_PATH, true));
    ASSERT_FALSE(dataset.cameras.empty());
    ASSERT_FALSE(dataset.world_points.empty());

    K0 = dataset.cameras[0].calib.intrinsics.K();
    K1 = dataset.cameras[1].calib.intrinsics.K();

    world2cam0 = dataset.cameras[0].calib.world2cam.matrix();
    world2cam1 = dataset.cameras[1].calib.world2cam.matrix();

    P0 = K0 * world2cam0.topRows<3>();
    P1 = K1 * world2cam1.topRows<3>();

    Eigen::Matrix<double, 4, 3> P0_pinv = P0.completeOrthogonalDecomposition().pseudoInverse();
    Vec4 cam0_center = Eigen::JacobiSVD(P0, Eigen::ComputeFullV).matrixV().rightCols<1>();
    Vec3 left_epipole = P1 * cam0_center;

    F01 = skewSymmetric(left_epipole) * P1 * P0_pinv;
    F01 /= F01(2, 2);

    common_pt_ids = dataset.getCommonPointIds({0, 1});
    image_pts0 = dataset.getImagePointArray(0, common_pt_ids);
    image_pts1 = dataset.getImagePointArray(1, common_pt_ids);
    world_pts = dataset.getWorldPointArray(common_pt_ids);
  }
};

TEST_F(CalibUtilsTestFixture, NormalizePoints2D) {
  constexpr double epsilon = 1e-10;

  {
    // Dynamic size input

    Mat3X pts_normed;
    Mat3 T;
    normalizePoints(image_pts0, pts_normed, T);

    EXPECT_TRUE(pts_normed.row(2).isApproxToConstant(1., epsilon));

    Vec2 mean = pts_normed.topRows<2>().rowwise().mean();
    EXPECT_TRUE(mean.isZero(epsilon));

    double rms = std::sqrt(pts_normed.topRows<2>().array().square().colwise().sum().mean());
    EXPECT_NEAR(rms, std::sqrt(2), epsilon);

    Mat3 Tinv = T.inverse();
    EXPECT_TRUE((Tinv * pts_normed).topRows<2>().isApprox(image_pts0, epsilon));
  }

  {
    // Fixed size input

    Eigen::Matrix<double, 2, 4> fixed_matrix = image_pts0.leftCols<4>();

    Mat3x4 pts_normed;
    Mat3 T;
    normalizePoints(fixed_matrix, pts_normed, T);

    EXPECT_TRUE(pts_normed.row(2).isApproxToConstant(1., epsilon));

    Vec2 mean = pts_normed.topRows<2>().rowwise().mean();
    EXPECT_TRUE(mean.isZero(epsilon));

    double rms = std::sqrt(pts_normed.topRows<2>().array().square().colwise().sum().mean());
    EXPECT_NEAR(rms, std::sqrt(2), epsilon);

    Mat3 Tinv = T.inverse();
    EXPECT_TRUE((Tinv * pts_normed).topRows<2>().isApprox(fixed_matrix, epsilon));
  }

  {
    // Matrix-expression input

    Mat3x4 pts_normed;
    Mat3 T;
    normalizePoints(image_pts0(Eigen::all, {2, 4, 7, 11}), pts_normed, T);

    EXPECT_TRUE(pts_normed.row(2).isApproxToConstant(1., epsilon));

    Vec2 mean = pts_normed.topRows<2>().rowwise().mean();
    EXPECT_TRUE(mean.isZero(epsilon));

    double rms = std::sqrt(pts_normed.topRows<2>().array().square().colwise().sum().mean());
    EXPECT_NEAR(rms, std::sqrt(2), epsilon);

    Mat3 Tinv = T.inverse();
    EXPECT_TRUE((Tinv * pts_normed).topRows<2>().isApprox(image_pts0(Eigen::all, {2, 4, 7, 11}), epsilon));
  }
}

TEST_F(CalibUtilsTestFixture, NormalizePoints3D) {
  constexpr double epsilon = 1e-10;

  {
    // Dynamic size input

    Mat4X pts_normed;
    Mat4 T;
    normalizePoints(world_pts, pts_normed, T);

    EXPECT_TRUE(pts_normed.row(3).isApproxToConstant(1., epsilon));

    Vec3 mean = pts_normed.topRows<3>().rowwise().mean();
    EXPECT_TRUE(mean.isZero(epsilon));

    double rms = std::sqrt(pts_normed.topRows<3>().array().square().colwise().sum().mean());
    EXPECT_NEAR(rms, std::sqrt(3), epsilon);

    Mat4 Tinv = T.inverse();
    EXPECT_TRUE((Tinv * pts_normed).topRows<3>().isApprox(world_pts, epsilon));
  }

  {
    // Fixed size input

    Mat3x4 fixed_matrix = world_pts.leftCols<4>();

    Mat4 pts_normed;
    Mat4 T;
    normalizePoints(fixed_matrix, pts_normed, T);

    EXPECT_TRUE(pts_normed.row(3).isApproxToConstant(1., epsilon));

    Vec3 mean = pts_normed.topRows<3>().rowwise().mean();
    EXPECT_TRUE(mean.isZero(epsilon));

    double rms = std::sqrt(pts_normed.topRows<3>().array().square().colwise().sum().mean());
    EXPECT_NEAR(rms, std::sqrt(3), epsilon);

    Mat4 Tinv = T.inverse();
    EXPECT_TRUE((Tinv * pts_normed).topRows<3>().isApprox(fixed_matrix, epsilon));
  }

  {
    // Matrix-expression input

    Mat4 pts_normed;
    Mat4 T;
    normalizePoints(world_pts(Eigen::all, {2, 4, 7, 11}), pts_normed, T);

    EXPECT_TRUE(pts_normed.row(3).isApproxToConstant(1., epsilon));

    Vec3 mean = pts_normed.topRows<3>().rowwise().mean();
    EXPECT_TRUE(mean.isZero(epsilon));

    double rms = std::sqrt(pts_normed.topRows<3>().array().square().colwise().sum().mean());
    EXPECT_NEAR(rms, std::sqrt(3), epsilon);

    Mat4 Tinv = T.inverse();
    EXPECT_TRUE((Tinv * pts_normed).topRows<3>().isApprox(world_pts(Eigen::all, {2, 4, 7, 11}), epsilon));
  }
}

TEST_F(CalibUtilsTestFixture, FindFundamentalMatrix) {
  Mat3 F = findFundamentalMatrix(image_pts0, image_pts1);
  EXPECT_TRUE(F.isApprox(F01, 1e-8));
}

TEST_F(CalibUtilsTestFixture, FindFundamentalMatrixNoisyObservations) {
  const double observation_noise = 2.0;
  const size_t seed = 7;

  dataset.addObservationNoise(observation_noise, seed);
  Mat2X noisy_image_pts0 = dataset.getImagePointArray(0, common_pt_ids);
  Mat2X noisy_image_pts1 = dataset.getImagePointArray(1, common_pt_ids);

  const double inlier_thr = observation_noise * 5.99;
  Mat3 F = findFundamentalMatrix(noisy_image_pts0, noisy_image_pts1);

  auto distance = symmetricEpipolarDistance(image_pts0, image_pts1, F);

  const size_t n_inliers = (distance.array() < inlier_thr * inlier_thr).count();
  const double inlier_ratio = static_cast<double>(n_inliers) / image_pts0.cols();

  EXPECT_GT(inlier_ratio, 0.95);
}

TEST_F(CalibUtilsTestFixture, FindFundamentalMatrixRansacAccurateObservations) {
  const size_t ransac_seed = 42;
  Mat3 F = findFundamentalMatrixRansac(image_pts0, image_pts1, 5.0, 0.99, 1000, ransac_seed);
  EXPECT_TRUE(F.isApprox(F01, 1e-8));
}

TEST_F(CalibUtilsTestFixture, FindFundamentalMatrixRansacNoisyObservations) {
  const double observation_noise = 2.0;
  const size_t seed = 7;
  const size_t ransac_seed = 42;

  dataset.addObservationNoise(observation_noise, seed);
  Mat2X noisy_image_pts0 = dataset.getImagePointArray(0, common_pt_ids);
  Mat2X noisy_image_pts1 = dataset.getImagePointArray(1, common_pt_ids);

  const double inlier_thr = observation_noise * 5.99;
  Mat3 F = findFundamentalMatrixRansac(noisy_image_pts0, noisy_image_pts1, inlier_thr, 0.99, 1000, ransac_seed);

  auto distance = symmetricEpipolarDistance(image_pts0, image_pts1, F);

  const size_t n_inliers = (distance.array() < inlier_thr * inlier_thr).count();
  const double inlier_ratio = static_cast<double>(n_inliers) / image_pts0.cols();

  EXPECT_GT(inlier_ratio, 0.95);
}

TEST_F(CalibUtilsTestFixture, FindFundamentalMatrixRansacNoisyObservationsWithOutliers) {
  const double observation_noise = 2.0;
  const size_t seed = 7;
  const size_t ransac_seed = 42;
  const double inlier_prob = 0.75;

  dataset.addObservationNoise(observation_noise, seed);
  dataset.addObservationOutliers(inlier_prob, seed);

  Mat2X noisy_image_pts0 = dataset.getImagePointArray(0, common_pt_ids);
  Mat2X noisy_image_pts1 = dataset.getImagePointArray(1, common_pt_ids);

  const double inlier_thr = observation_noise * 5.99;
  Mat3 F = findFundamentalMatrixRansac(noisy_image_pts0, noisy_image_pts1, inlier_thr, 0.99, 1000, ransac_seed);

  auto distance = symmetricEpipolarDistance(image_pts0, image_pts1, F);

  const size_t n_inliers = (distance.array() < inlier_thr * inlier_thr).count();
  const double inlier_ratio = static_cast<double>(n_inliers) / image_pts0.cols();

  EXPECT_GT(inlier_ratio, 0.95);
}

TEST_F(CalibUtilsTestFixture, FindProjectionMatrix) {
  Mat3x4 P = findProjectionMatrix(world_pts, image_pts1);
  P /= P(0, 0);
  P1 /= P1(0, 0);

  EXPECT_TRUE(P.isApprox(P1, 1e-5));
}

TEST_F(CalibUtilsTestFixture, FindProjectionMatrixRansacAccurateCorrespondences) {
  const size_t ransac_seed = 42;
  Mat3x4 P = findProjectionMatrixRansac(world_pts, image_pts1, 5.0, 0.99, 1000, ransac_seed);
  P /= P(0, 0);
  P1 /= P1(0, 0);

  EXPECT_TRUE(P.isApprox(P1, 1e-5));
}

TEST_F(CalibUtilsTestFixture, FindProjectionMatrixRansacNoisyObservations) {
  const double observation_noise = 2.0;
  const size_t seed = 7;
  const size_t ransac_seed = 42;

  dataset.addObservationNoise(observation_noise, seed);
  Mat2X noisy_image_pts1 = dataset.getImagePointArray(1, common_pt_ids);

  const double inlier_thr = observation_noise * 5.99;
  Mat3x4 P = findProjectionMatrixRansac(world_pts, noisy_image_pts1, inlier_thr, 0.99, 1000, ransac_seed);

  auto distance = ProjectionEstimatorRansacSpec::distance(world_pts, image_pts1, P);

  const size_t n_inliers = (distance.array() < inlier_thr * inlier_thr).count();
  const double inlier_ratio = static_cast<double>(n_inliers) / image_pts0.cols();

  EXPECT_GT(inlier_ratio, 0.95);
}

TEST_F(CalibUtilsTestFixture, FindProjectionMatrixRansacNoisyObservationsWithOutliers) {
  const double observation_noise = 2.0;
  const size_t seed = 7;
  const size_t ransac_seed = 42;
  const double inlier_prob = 0.75;

  dataset.addObservationNoise(observation_noise, seed);
  dataset.addObservationOutliers(inlier_prob, seed);

  Mat2X noisy_image_pts0 = dataset.getImagePointArray(0, common_pt_ids);
  Mat2X noisy_image_pts1 = dataset.getImagePointArray(1, common_pt_ids);

  const double inlier_thr = observation_noise * 5.99;
  Mat3x4 P = findProjectionMatrixRansac(world_pts, noisy_image_pts1, inlier_thr, 0.99, 1000, ransac_seed);

  auto distance = ProjectionEstimatorRansacSpec::distance(world_pts, image_pts1, P);

  const size_t n_inliers = (distance.array() < inlier_thr * inlier_thr).count();
  const double inlier_ratio = static_cast<double>(n_inliers) / image_pts0.cols();

  EXPECT_GT(inlier_ratio, 0.95);
}

TEST_F(CalibUtilsTestFixture, TriangulatePoints) {
  auto triangulated_pts = triangulatePoints(image_pts0, image_pts1, P0, P1);

  EXPECT_TRUE((triangulated_pts - world_pts).colwise().squaredNorm().isZero());
}

TEST_F(CalibUtilsTestFixture, TriangulatePointsRansac) {
  const size_t seed = 7;
  const size_t ransac_seed = 45;
  const double inlier_prob = 0.9;

  const double observation_noise = 2.0;
  const double inlier_thr = observation_noise * 5.99;

  dataset.addObservationOutliers(inlier_prob, seed);

  for (const auto& [pt_id, true_world_pt] : dataset.world_points) {
    Mat2X image_pts(2, dataset.cameras.size());
    Eigen::Matrix<double, 12, Eigen::Dynamic> Ps_flattened(12, dataset.cameras.size());

    int n_points = 0;
    for (const auto& [cam_id, cam_data] : dataset.cameras) {
      auto obs_it = cam_data.observations.find(pt_id);
      if (obs_it == cam_data.observations.end()) {
        continue;
      }

      image_pts.col(n_points) = obs_it->second;
      Ps_flattened.col(n_points) = cam_data.calib.P().reshaped();
      n_points++;
    }

    if (n_points < 2) {
      continue;
    }

    image_pts.conservativeResize(Eigen::NoChange, n_points);
    Ps_flattened.conservativeResize(Eigen::NoChange, n_points);

    Vec3 recon_world_pt = triangulatePointRansac(image_pts, Ps_flattened, inlier_thr, 0.99, 100, ransac_seed);
    LOG(INFO) << "Pt " << pt_id << ": " << recon_world_pt.transpose() << " vs. " << true_world_pt.transpose();
    EXPECT_NEAR(recon_world_pt.x(), true_world_pt.x(), 1e-2) << "PT: " << pt_id;
    EXPECT_NEAR(recon_world_pt.y(), true_world_pt.y(), 1e-2) << "PT: " << pt_id;
    EXPECT_NEAR(recon_world_pt.z(), true_world_pt.z(), 1e-2) << "PT: " << pt_id;
  }
}
