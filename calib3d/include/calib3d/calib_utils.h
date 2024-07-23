// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <Eigen/Core>
#include <random>

namespace calib3d {

// Computes the skew-symmetric matrix of a vector
template <class Derived>
inline Eigen::Matrix3d skewSymmetric(const Eigen::DenseBase<Derived>& v);

// Computes the Sampson distance for a set of points given a fundamental matrix
template <class DerivedSrc, class DerivedDst>
Eigen::Vector<double, DerivedSrc::ColsAtCompileTime> sampsonDistanceFromFundamentalMatrix(
    const Eigen::DenseBase<DerivedSrc>& src_points,
    const Eigen::DenseBase<DerivedDst>& dst_points,
    const Eigen::Matrix3d& F);

// Computes the symmetric epipolar distance for a set of points given a fundamental matrix
template <class DerivedSrc, class DerivedDst>
Eigen::Vector<double, DerivedSrc::ColsAtCompileTime> symmetricEpipolarDistance(
    const Eigen::DenseBase<DerivedSrc>& src_points,
    const Eigen::DenseBase<DerivedDst>& dst_points,
    const Eigen::Matrix3d& F);

// Normalizes points for numerical stability
template <class Derived>
void normalizePoints(const Eigen::DenseBase<Derived>& points,
                     Eigen::Matrix<double, Derived::RowsAtCompileTime + 1, Derived::ColsAtCompileTime>& X_normed,
                     Eigen::Matrix<double, Derived::RowsAtCompileTime + 1, Derived::RowsAtCompileTime + 1>& T);

// Normalizes a matrix for numerical stability
template <class Derived>
Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> normalizeMatrix(
    const Eigen::DenseBase<Derived>& matrix);

// Finds the fundamental matrix given point correspondences
template <class DerivedSrc, class DerivedDst>
Eigen::Matrix3d findFundamentalMatrix(const Eigen::DenseBase<DerivedSrc>& src_points,
                                      const Eigen::DenseBase<DerivedDst>& dst_points);

// Finds the fundamental matrix using RANSAC
template <class DerivedSrc, class DerivedDst>
Eigen::Matrix3d findFundamentalMatrixRansac(const Eigen::DenseBase<DerivedSrc>& src_points,
                                            const Eigen::DenseBase<DerivedDst>& dst_points,
                                            double ransac_thr,
                                            double confidence,
                                            size_t max_iters,
                                            size_t seed = std::random_device()());

// Finds the projection matrix given world and image points
template <class Derived3D, class Derived2D>
Eigen::Matrix<double, 3, 4> findProjectionMatrix(const Eigen::DenseBase<Derived3D>& world_points,
                                                 const Eigen::DenseBase<Derived2D>& image_points);

// Finds the projection matrix using RANSAC
template <class Derived3D, class Derived2D>
Eigen::Matrix<double, 3, 4> findProjectionMatrixRansac(const Eigen::DenseBase<Derived3D>& world_points,
                                                       const Eigen::DenseBase<Derived2D>& image_points,
                                                       double ransac_thr,
                                                       double confidence,
                                                       size_t max_iters,
                                                       size_t seed = std::random_device()());

// Triangulates 3D points from two sets of image points and projection matrices
template <class Derived1, class Derived2>
Eigen::Matrix<double, 3, Derived1::ColsAtCompileTime> triangulatePoints(const Eigen::DenseBase<Derived1>& image_points1,
                                                                        const Eigen::DenseBase<Derived2>& image_points2,
                                                                        const Eigen::Matrix<double, 3, 4>& P1,
                                                                        const Eigen::Matrix<double, 3, 4>& P2);

// Triangulates a single 3D point from multiple image points and projection matrices
template <class Derived2D, class DerivedPFlat>
Eigen::Vector3d triangulatePoint(const Eigen::DenseBase<Derived2D>& image_points,
                                 const Eigen::DenseBase<DerivedPFlat>& Ps_flattened);

// Triangulates a single 3D point using RANSAC
template <class Derived2D, class DerivedPFlat>
Eigen::Vector3d triangulatePointRansac(const Eigen::DenseBase<Derived2D>& image_points,
                                       const Eigen::DenseBase<DerivedPFlat>& Ps_flattened,
                                       double ransac_thr,
                                       double confidence,
                                       size_t max_iters,
                                       size_t seed = std::random_device()());

} // namespace calib3d

#include <calib3d/calib_utils.hpp>
