#include <calib3d/calib_utils.h>

#include <Eigen/QR>
#include <Eigen/SVD>
#include <gtest/gtest.h>

#include <calib3d/dataset.h>

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
    ASSERT_TRUE(loadJsonDataset(DATASET_FILE_PATH, dataset));
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

    const auto& cam0_observations = dataset.cameras[0].observations;
    const auto& cam1_observations = dataset.cameras[1].observations;
    std::set<size_t> cam0_pt_ids, cam1_pt_ids;
    for (const auto& kv : cam0_observations) {
      cam0_pt_ids.insert(kv.first);
    }
    for (const auto& kv : cam1_observations) {
      cam1_pt_ids.insert(kv.first);
    }
    std::set_intersection(cam0_pt_ids.begin(),
                          cam0_pt_ids.end(),
                          cam1_pt_ids.begin(),
                          cam1_pt_ids.end(),
                          std::back_inserter(common_pt_ids));

    image_pts0.resize(2, common_pt_ids.size());
    image_pts1.resize(2, common_pt_ids.size());
    world_pts.resize(3, common_pt_ids.size());
    for (size_t i = 0; i < common_pt_ids.size(); i++) {
      size_t pt_id = common_pt_ids[i];
      image_pts0.col(i) = cam0_observations.at(pt_id);
      image_pts1.col(i) = cam1_observations.at(pt_id);
      world_pts.col(i) = dataset.world_points.at(pt_id);
    }
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

TEST_F(CalibUtilsTestFixture, DISABLED_SampsonDistance) {
  const size_t data_seed = 7;
  std::mt19937 rng(data_seed);
  std::normal_distribution<double> gaussian_noise(0.0, 3.0); // very small noise

  Mat2X noisy_image_pts0 = image_pts0;
  Mat2X noisy_image_pts1 = image_pts1;

  for (int i = 0; i < image_pts0.cols(); i++) {
    noisy_image_pts0(0, i) += gaussian_noise(rng);
    noisy_image_pts0(1, i) += gaussian_noise(rng);
    noisy_image_pts1(0, i) += gaussian_noise(rng);
    noisy_image_pts1(1, i) += gaussian_noise(rng);
  }

  Mat2X noise0 = noisy_image_pts0 - image_pts0;
  Mat2X noise1 = noisy_image_pts1 - image_pts1;

  auto sampson_dist = sampsonDistanceFromFundamentalMatrix(noisy_image_pts0, noisy_image_pts1, F01);
  auto epipolar_dist = symmetricEpipolarDistance(noisy_image_pts0, noisy_image_pts1, F01);

  LOG(INFO) << "Epipolar distance vs. Sampson distance";
  for (int i = 0; i < image_pts0.cols(); i++) {
    LOG(INFO) << "Point " << i << ": " << epipolar_dist(i) << " vs. " << sampson_dist(i) << " / "
              << noise0.col(i).transpose() << " / " << noise1.col(i).transpose();
  }
  for (int i = 0; i < image_pts0.cols(); i++) {
    LOG(INFO) << epipolar_dist(i);
  }
}

TEST_F(CalibUtilsTestFixture, FindFundamentalMatrix) {
  Mat3 F = findFundamentalMatrix(image_pts0, image_pts1);
  EXPECT_TRUE(F.isApprox(F01, 1e-8));
}

TEST_F(CalibUtilsTestFixture, FindFundamentalMatrixNoisyObservations) {
  const size_t data_seed = 7;
  std::mt19937 rng(data_seed);
  const double pixel_noise = 2.0;
  std::normal_distribution<double> gaussian_noise(0.0, pixel_noise);

  Mat2X noisy_image_pts0 = image_pts0;
  Mat2X noisy_image_pts1 = image_pts1;

  for (int i = 0; i < image_pts0.cols(); i++) {
    noisy_image_pts0(0, i) += gaussian_noise(rng);
    noisy_image_pts0(1, i) += gaussian_noise(rng);
    noisy_image_pts1(0, i) += gaussian_noise(rng);
    noisy_image_pts1(1, i) += gaussian_noise(rng);
  }

  const double inlier_thr = pixel_noise * 5.99;
  Mat3 F = findFundamentalMatrix(noisy_image_pts0, noisy_image_pts1);

  auto distance = symmetricEpipolarDistance(image_pts0, image_pts1, F);

  const size_t n_inliers = (distance.array() < inlier_thr * inlier_thr).count();
  const double inlier_ratio = static_cast<double>(n_inliers) / image_pts0.cols();
  LOG(INFO) << "Inlier ratio: " << inlier_ratio;

  EXPECT_GT(inlier_ratio, 0.95);
}

TEST_F(CalibUtilsTestFixture, FindFundamentalMatrixRansacAccurateObservations) {
  const size_t ransac_seed = 42;
  Mat3 F = findFundamentalMatrixRansac(image_pts0, image_pts1, 5.0, 0.99, 1000, ransac_seed);
  EXPECT_TRUE(F.isApprox(F01, 1e-8));
}

TEST_F(CalibUtilsTestFixture, FindFundamentalMatrixRansacNoisyObservations) {
  const size_t data_seed = 7;
  std::mt19937 rng(data_seed);
  const double pixel_noise = 2.0;
  std::normal_distribution<double> gaussian_noise(0.0, pixel_noise);

  Mat2X noisy_image_pts0 = image_pts0;
  Mat2X noisy_image_pts1 = image_pts1;

  for (int i = 0; i < image_pts0.cols(); i++) {
    noisy_image_pts0(0, i) += gaussian_noise(rng);
    noisy_image_pts0(1, i) += gaussian_noise(rng);
    noisy_image_pts1(0, i) += gaussian_noise(rng);
    noisy_image_pts1(1, i) += gaussian_noise(rng);
  }

  const double inlier_thr = pixel_noise * 5.99;
  Mat3 F = findFundamentalMatrixRansac(noisy_image_pts0, noisy_image_pts1, inlier_thr, 0.99, 1000, data_seed);

  auto distance = symmetricEpipolarDistance(image_pts0, image_pts1, F);

  const size_t n_inliers = (distance.array() < inlier_thr * inlier_thr).count();
  const double inlier_ratio = static_cast<double>(n_inliers) / image_pts0.cols();
  LOG(INFO) << "Inlier ratio: " << inlier_ratio;

  EXPECT_GT(inlier_ratio, 0.95);
}

TEST_F(CalibUtilsTestFixture, FindFundamentalMatrixRansacNoisyObservationsWithOutliers) {
  const size_t data_seed = 7;
  std::mt19937 rng(data_seed);
  const double pixel_noise = 2.0;
  std::normal_distribution<double> gaussian_noise(0.0, pixel_noise);

  Mat2X noisy_image_pts0 = image_pts0;
  Mat2X noisy_image_pts1 = image_pts1;

  for (int i = 0; i < image_pts0.cols(); i++) {
    noisy_image_pts0(0, i) += gaussian_noise(rng);
    noisy_image_pts0(1, i) += gaussian_noise(rng);
    noisy_image_pts1(0, i) += gaussian_noise(rng);
    noisy_image_pts1(1, i) += gaussian_noise(rng);
  }

  const double inlier_prob = 0.75;
  std::uniform_real_distribution<double> x_dist(0., dataset.cameras[0].calib.size[0]);
  std::uniform_real_distribution<double> y_dist(0., dataset.cameras[0].calib.size[1]);
  std::uniform_real_distribution<double> inlier_dist(0., 1.0);

  size_t n_outliers = 0;

  for (int i = 0; i < image_pts0.cols(); i++) {
    if (inlier_dist(rng) < inlier_prob) {
      continue;
    }

    n_outliers++;

    noisy_image_pts0(0, i) = x_dist(rng);
    noisy_image_pts0(1, i) = y_dist(rng);
    noisy_image_pts1(0, i) = x_dist(rng);
    noisy_image_pts1(1, i) = y_dist(rng);
  }

  LOG(INFO) << "N outliers: " << n_outliers;
  const double inlier_ratio_in_sample =
      static_cast<double>(noisy_image_pts0.cols() - n_outliers) / noisy_image_pts0.cols();

  const double inlier_thr = pixel_noise * 5.99;
  Mat3 F = findFundamentalMatrixRansac(noisy_image_pts0, noisy_image_pts1, inlier_thr, 0.99, 1000, data_seed);

  auto distance = symmetricEpipolarDistance(image_pts0, image_pts1, F);

  const size_t n_inliers = (distance.array() < inlier_thr * inlier_thr).count();
  const double inlier_ratio = static_cast<double>(n_inliers) / image_pts0.cols();
  LOG(INFO) << "Inlier ratio: " << inlier_ratio;

  EXPECT_GT(inlier_ratio, 0.95);

  auto noisy_distance = symmetricEpipolarDistance(noisy_image_pts0, noisy_image_pts1, F);

  const size_t noisy_n_inliers = (noisy_distance.array() < inlier_thr * inlier_thr).count();
  const double noisy_inlier_ratio = static_cast<double>(noisy_n_inliers) / noisy_image_pts0.cols();
  LOG(INFO) << "Noisy_inlier ratio: " << noisy_inlier_ratio;

  EXPECT_LE(noisy_inlier_ratio, inlier_ratio_in_sample);
  EXPECT_GT(inlier_ratio, 0.95 * inlier_ratio_in_sample);
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
  const size_t data_seed = 7;
  std::mt19937 rng(data_seed);
  const double pixel_noise = 2.0;
  std::normal_distribution<double> gaussian_noise(0.0, pixel_noise);

  Mat2X noisy_image_pts1 = image_pts1;

  for (int i = 0; i < image_pts1.cols(); i++) {
    noisy_image_pts1(0, i) += gaussian_noise(rng);
    noisy_image_pts1(1, i) += gaussian_noise(rng);
  }

  const double inlier_thr = pixel_noise * 5.99;
  Mat3x4 P = findProjectionMatrixRansac(world_pts, noisy_image_pts1, inlier_thr, 0.99, 1000, data_seed);

  auto distance = ProjectionEstimatorRansacSpec::distance(world_pts, image_pts1, P);

  const size_t n_inliers = (distance.array() < inlier_thr * inlier_thr).count();
  const double inlier_ratio = static_cast<double>(n_inliers) / image_pts0.cols();
  LOG(INFO) << "Inlier ratio: " << inlier_ratio;

  EXPECT_GT(inlier_ratio, 0.95);
}

TEST_F(CalibUtilsTestFixture, FindProjectionMatrixRansacNoisyObservationsWithOutliers) {
  const size_t data_seed = 7;
  std::mt19937 rng(data_seed);
  const double pixel_noise = 2.0;
  std::normal_distribution<double> gaussian_noise(0.0, pixel_noise);

  Mat2X noisy_image_pts1 = image_pts1;

  for (int i = 0; i < image_pts1.cols(); i++) {
    noisy_image_pts1(0, i) += gaussian_noise(rng);
    noisy_image_pts1(1, i) += gaussian_noise(rng);
  }

  const double inlier_prob = 0.75;
  std::uniform_real_distribution<double> x_dist(0., dataset.cameras[0].calib.size[0]);
  std::uniform_real_distribution<double> y_dist(0., dataset.cameras[0].calib.size[1]);
  std::uniform_real_distribution<double> inlier_dist(0., 1.0);

  size_t n_outliers = 0;

  for (int i = 0; i < image_pts0.cols(); i++) {
    if (inlier_dist(rng) < inlier_prob) {
      continue;
    }

    n_outliers++;

    noisy_image_pts1(0, i) = x_dist(rng);
    noisy_image_pts1(1, i) = y_dist(rng);
  }

  LOG(INFO) << "N outliers: " << n_outliers;
  const double inlier_ratio_in_sample =
      static_cast<double>(noisy_image_pts1.cols() - n_outliers) / noisy_image_pts1.cols();

  const double inlier_thr = pixel_noise * 5.99;
  Mat3x4 P = findProjectionMatrixRansac(world_pts, noisy_image_pts1, inlier_thr, 0.99, 1000, data_seed);

  auto distance = ProjectionEstimatorRansacSpec::distance(world_pts, image_pts1, P);

  const size_t n_inliers = (distance.array() < inlier_thr * inlier_thr).count();
  const double inlier_ratio = static_cast<double>(n_inliers) / image_pts0.cols();
  LOG(INFO) << "Inlier ratio: " << inlier_ratio;

  EXPECT_GT(inlier_ratio, 0.95);

  auto noisy_distance = ProjectionEstimatorRansacSpec::distance(world_pts, noisy_image_pts1, P);

  const size_t noisy_n_inliers = (noisy_distance.array() < inlier_thr * inlier_thr).count();
  const double noisy_inlier_ratio = static_cast<double>(noisy_n_inliers) / noisy_image_pts1.cols();
  LOG(INFO) << "Noisy_inlier ratio: " << noisy_inlier_ratio;

  EXPECT_LE(noisy_inlier_ratio, inlier_ratio_in_sample);
  EXPECT_GT(inlier_ratio, 0.95 * inlier_ratio_in_sample);
}

TEST_F(CalibUtilsTestFixture, TriangulatePoints) {
  auto triangulated_pts = triangulatePoints(image_pts0, image_pts1, P0, P1);

  EXPECT_TRUE((triangulated_pts - world_pts).colwise().squaredNorm().isZero());
}
