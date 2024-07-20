#pragma once

#include <Eigen/SVD>
#include <cmath>
#include <glog/logging.h>

#include <calib3d/RansacEngine.h>

namespace calib3d {

template <class Derived>
inline Eigen::Matrix3d skewSymmetric(const Eigen::DenseBase<Derived>& v) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);

  Eigen::Matrix3d skew;
  // clang-format off
  skew <<     0., -v.z(),  v.y(),
           v.z(),     0., -v.x(),
          -v.y(),  v.x(),     0.;
  // clang-format on
  return skew;
}

template <class DerivedSrc, class DerivedDst>
Eigen::Vector<double, DerivedSrc::ColsAtCompileTime> sampsonDistanceFromFundamentalMatrix(
    const Eigen::DenseBase<DerivedSrc>& src_points,
    const Eigen::DenseBase<DerivedDst>& dst_points,
    const Eigen::Matrix3d& F) {
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(DerivedSrc, DerivedDst);
  static_assert(DerivedSrc::RowsAtCompileTime == 2);

  const int n_points = src_points.cols();
  Eigen::Vector<double, DerivedSrc::ColsAtCompileTime> distances(n_points);

  for (int i = 0; i < n_points; i++) {
    Eigen::Vector3d src = src_points.col(i).homogeneous();
    Eigen::Vector3d dst = dst_points.col(i).homogeneous();

    double numerator = (dst.transpose() * F * src).squaredNorm();
    double denominator = (F.topRows<2>() * src).squaredNorm() + (F.transpose().topRows<2>() * dst).squaredNorm();
    distances(i) = numerator / denominator;
  }

  return distances;
}

template <class DerivedSrc, class DerivedDst>
Eigen::Vector<double, DerivedSrc::ColsAtCompileTime> symmetricEpipolarDistance(
    const Eigen::DenseBase<DerivedSrc>& src_points,
    const Eigen::DenseBase<DerivedDst>& dst_points,
    const Eigen::Matrix3d& F) {
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(DerivedSrc, DerivedDst);
  static_assert(DerivedSrc::RowsAtCompileTime == 2);

  const int n_points = src_points.cols();
  Eigen::Vector<double, DerivedSrc::ColsAtCompileTime> distances(n_points);

  for (int i = 0; i < n_points; i++) {
    Eigen::Vector3d src = src_points.col(i).homogeneous();
    Eigen::Vector3d dst = dst_points.col(i).homogeneous();

    double numerator = (dst.transpose() * F * src).squaredNorm();
    double denominator1 = (F.topRows<2>() * src).squaredNorm();
    double denominator2 = (F.transpose().topRows<2>() * dst).squaredNorm();
    distances(i) = std::max(numerator / denominator1, numerator / denominator2);
  }

  return distances;
}

template <class Derived>
void normalizePoints(const Eigen::DenseBase<Derived>& points,
                     Eigen::Matrix<double, Derived::RowsAtCompileTime + 1, Derived::ColsAtCompileTime>& X_normed,
                     Eigen::Matrix<double, Derived::RowsAtCompileTime + 1, Derived::RowsAtCompileTime + 1>& T) {
  constexpr size_t Dim = Derived::RowsAtCompileTime;

  Eigen::Vector<double, Dim> centroid = points.rowwise().mean();
  double rms_dist = std::sqrt((points.colwise() - centroid).colwise().squaredNorm().mean());
  double scaling_param = rms_dist / std::sqrt(Dim);

  Eigen::Matrix<double, Dim + 1, Dim + 1> T_inv = Eigen::Matrix<double, Dim + 1, Dim + 1>::Identity();
  T_inv.template topLeftCorner<Dim, Dim>().diagonal().setConstant(scaling_param);
  T_inv.template topRightCorner<Dim, 1>() = centroid;
  T = T_inv.inverse();

  X_normed = T * points.colwise().homogeneous();
}

template <class Derived>
Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> normalizeMatrix(
    const Eigen::DenseBase<Derived>& matrix) {
  const double norm = matrix.reshaped().norm();
  const double desired_norm = std::sqrt(static_cast<double>(matrix.cols() * matrix.rows()));

  const double scale = desired_norm / norm;

  return matrix.derived() * scale;
}

template <class DerivedSrc, class DerivedDst>
Eigen::Matrix3d findFundamentalMatrix(const Eigen::DenseBase<DerivedSrc>& src_points,
                                      const Eigen::DenseBase<DerivedDst>& dst_points) {
  EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(DerivedSrc, DerivedDst);
  static_assert(DerivedSrc::RowsAtCompileTime == 2);
  CHECK_EQ(src_points.cols(), dst_points.cols());

  const size_t n_points = src_points.cols();
  CHECK_GE(n_points, 8);

  Eigen::Matrix<double, 3, DerivedSrc::ColsAtCompileTime> src_normed;
  Eigen::Matrix<double, 3, DerivedDst::ColsAtCompileTime> dst_normed;
  Eigen::Matrix3d T_src, T_dst;

  normalizePoints(src_points, src_normed, T_src);
  normalizePoints(dst_points, dst_normed, T_dst);

  Eigen::Matrix<double, DerivedSrc::ColsAtCompileTime, 9> A(n_points, 9);

  for (size_t i = 0; i < n_points; i++) {
    A.row(i) = (dst_normed.col(i) * src_normed.col(i).transpose()).reshaped();
  }

  Eigen::JacobiSVD svd_A(A, Eigen::ComputeFullV);
  Eigen::Matrix3d F_candidate = svd_A.matrixV().col(8).reshaped(3, 3);

  Eigen::JacobiSVD svd_F(F_candidate, Eigen::ComputeFullV | Eigen::ComputeFullU);
  Eigen::Vector3d F_singular_vals = svd_F.singularValues();
  F_singular_vals[2] = 0.;

  Eigen::Matrix3d F = svd_F.matrixU() * Eigen::DiagonalMatrix<double, 3>(F_singular_vals) * svd_F.matrixV().transpose();
  F = T_dst.transpose() * F * T_src;
  F /= F(2, 2);

  return F;
}

struct FMEstimatorRansacSpec {
  using ModelMatrix = Eigen::Matrix3d;
  static constexpr size_t MinSampleSize = 8;
  static constexpr size_t Dim1 = 2;
  static constexpr size_t Dim2 = 2;

  template <class Derived1, class Derived2>
  static ModelMatrix fitModel(const Eigen::DenseBase<Derived1>& points1, const Eigen::DenseBase<Derived2>& points2) {
    return findFundamentalMatrix(points1, points2);
  }

  template <class Derived1, class Derived2>
  static Eigen::Vector<double, Derived1::ColsAtCompileTime> distance(const Eigen::DenseBase<Derived1>& points1,
                                                                     const Eigen::DenseBase<Derived2>& points2,
                                                                     const ModelMatrix& model) {
    return symmetricEpipolarDistance(points1, points2, model);
  }
};

template <class DerivedSrc, class DerivedDst>
Eigen::Matrix3d findFundamentalMatrixRansac(const Eigen::DenseBase<DerivedSrc>& src_points,
                                            const Eigen::DenseBase<DerivedDst>& dst_points,
                                            double ransac_thr,
                                            double confidence,
                                            size_t max_iters,
                                            size_t seed) {
  return RansacEngine<FMEstimatorRansacSpec>::fit(src_points, dst_points, ransac_thr, confidence, max_iters, seed);
}

template <class Derived3D, class Derived2D>
Eigen::Matrix<double, 3, 4> findProjectionMatrix(const Eigen::DenseBase<Derived3D>& world_points,
                                                 const Eigen::DenseBase<Derived2D>& image_points) {
  static_assert(Derived3D::RowsAtCompileTime == 3);
  static_assert(Derived2D::RowsAtCompileTime == 2);
  CHECK_EQ(world_points.cols(), image_points.cols());

  const size_t n_points = world_points.cols();
  CHECK_GE(n_points, 6);

  Eigen::Matrix<double, 4, Derived3D::ColsAtCompileTime> world_points_normed;
  Eigen::Matrix<double, 3, Derived2D::ColsAtCompileTime> image_points_normed;
  Eigen::Matrix4d T_world;
  Eigen::Matrix3d T_image;

  normalizePoints(world_points, world_points_normed, T_world);
  normalizePoints(image_points, image_points_normed, T_image);

  constexpr auto ARows =
      (Derived3D::ColsAtCompileTime == Eigen::Dynamic) ? Eigen::Dynamic : 2 * Derived3D::ColsAtCompileTime;
  Eigen::Matrix<double, ARows, 12> A(n_points * 2, 12);
  A.setZero();

  for (size_t i = 0; i < n_points; i++) {
    A.template block<1, 4>(2 * i, 0) = image_points_normed(2, i) * world_points_normed.col(i);
    A.template block<1, 4>(2 * i, 8) = -image_points_normed(0, i) * world_points_normed.col(i);
    A.template block<1, 4>(2 * i + 1, 4) = image_points_normed(2, i) * world_points_normed.col(i);
    A.template block<1, 4>(2 * i + 1, 8) = -image_points_normed(1, i) * world_points_normed.col(i);
  }

  Eigen::JacobiSVD svd_A(A, Eigen::ComputeFullV);
  Eigen::Matrix<double, 3, 4> P = svd_A.matrixV().col(11).template reshaped<Eigen::RowMajor>(3, 4);
  P = T_image.inverse() * P * T_world;

  return normalizeMatrix(P);
}

struct ProjectionEstimatorRansacSpec {
  using ModelMatrix = Eigen::Matrix<double, 3, 4>;
  static constexpr size_t MinSampleSize = 6;
  static constexpr size_t Dim1 = 3;
  static constexpr size_t Dim2 = 2;

  template <class Derived1, class Derived2>
  static ModelMatrix fitModel(const Eigen::DenseBase<Derived1>& points1, const Eigen::DenseBase<Derived2>& points2) {
    return findProjectionMatrix(points1, points2);
  }

  template <class Derived1, class Derived2>
  static Eigen::Vector<double, Derived1::ColsAtCompileTime> distance(const Eigen::DenseBase<Derived1>& points1,
                                                                     const Eigen::DenseBase<Derived2>& points2,
                                                                     const ModelMatrix& model) {
    auto reprojected_points = (model * points1.colwise().homogeneous()).colwise().hnormalized().eval();
    return (reprojected_points - points2.derived()).colwise().squaredNorm();
  }
};

template <class Derived3D, class Derived2D>
Eigen::Matrix<double, 3, 4> findProjectionMatrixRansac(const Eigen::DenseBase<Derived3D>& world_points,
                                                       const Eigen::DenseBase<Derived2D>& image_points,
                                                       double ransac_thr,
                                                       double confidence,
                                                       size_t max_iters,
                                                       size_t seed) {
  return RansacEngine<ProjectionEstimatorRansacSpec>::fit(
      world_points, image_points, ransac_thr, confidence, max_iters, seed);
}

template <class Derived1, class Derived2>
Eigen::Matrix<double, 3, Derived1::ColsAtCompileTime> triangulatePoints(const Eigen::DenseBase<Derived1>& image_points1,
                                                                        const Eigen::DenseBase<Derived2>& image_points2,
                                                                        const Eigen::Matrix<double, 3, 4>& P1,
                                                                        const Eigen::Matrix<double, 3, 4>& P2) {
  static_assert(Derived1::RowsAtCompileTime == 2);
  static_assert(Derived2::RowsAtCompileTime == 2);
  CHECK_EQ(image_points1.cols(), image_points2.cols());
  const int n_points = image_points1.cols();

  Eigen::Matrix<double, 3, Derived1::ColsAtCompileTime> world_points(3, n_points);

  for (int i = 0; i < n_points; i++) {
    Eigen::Matrix4d A;
    Eigen::Vector2d pt1 = image_points1.col(i);
    Eigen::Vector2d pt2 = image_points2.col(i);

    A.row(0) = pt1.x() * P1.row(2) - P1.row(0);
    A.row(1) = pt1.y() * P1.row(2) - P1.row(1);
    A.row(2) = pt2.x() * P2.row(2) - P2.row(0);
    A.row(3) = pt2.y() * P2.row(2) - P2.row(1);

    Eigen::JacobiSVD svd_A(A, Eigen::ComputeFullV);
    world_points.col(i) = svd_A.matrixV().col(3).hnormalized();
  }

  return world_points;
}

} // namespace calib3d
