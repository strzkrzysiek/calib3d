#pragma once

#include <Eigen/SVD>
#include <cmath>
#include <glog/logging.h>

#include <calib3d/ransac.h>

namespace calib3d {

template <class Derived>
inline Eigen::Matrix3d skewSymmetric(const Eigen::DenseBase<Derived>& v) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);

  Eigen::Matrix3d skew;
  skew << 0., -v.z(), v.y(), v.z(), 0., -v.x(), -v.y(), v.x(), 0.;
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
    double denominator =
        (F.topRows<2>() * src).squaredNorm() + (F.transpose().topRows<2>() * dst).squaredNorm();
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
void normalizePoints(
    const Eigen::DenseBase<Derived>& points,
    Eigen::Matrix<double, Derived::RowsAtCompileTime + 1, Derived::ColsAtCompileTime>& X_normed,
    Eigen::Matrix<double, Derived::RowsAtCompileTime + 1, Derived::RowsAtCompileTime + 1>& T) {
  constexpr size_t Dim = Derived::RowsAtCompileTime;

  Eigen::Vector<double, Dim> centroid = points.rowwise().mean();
  double rms_dist = std::sqrt((points.colwise() - centroid).colwise().squaredNorm().mean());
  double scaling_param = rms_dist / std::sqrt(Dim);

  Eigen::Matrix<double, Dim + 1, Dim + 1> T_inv =
      Eigen::Matrix<double, Dim + 1, Dim + 1>::Identity();
  T_inv.template topLeftCorner<Dim, Dim>().diagonal().setConstant(scaling_param);
  T_inv.template topRightCorner<Dim, 1>() = centroid;
  T = T_inv.inverse();

  X_normed = T * points.colwise().homogeneous();
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

  Eigen::Matrix3d F = svd_F.matrixU() * Eigen::DiagonalMatrix<double, 3>(F_singular_vals) *
                      svd_F.matrixV().transpose();
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
  static ModelMatrix fitModel(const Eigen::DenseBase<Derived1>& points1,
                              const Eigen::DenseBase<Derived2>& points2) {
    return findFundamentalMatrix(points1, points2);
  }

  template <class Derived1, class Derived2>
  static Eigen::Vector<double, Derived1::ColsAtCompileTime> distance(
      const Eigen::DenseBase<Derived1>& points1,
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
  return RansacEngine<FMEstimatorRansacSpec>::fit(
      src_points, dst_points, ransac_thr, confidence, max_iters, seed);
}

} // namespace calib3d
