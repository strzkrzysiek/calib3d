// Copyright 2024 Krzysztof Wrobel

#pragma once

#include <Eigen/Core>
#include <random>

namespace calib3d {

// Computes the skew-symmetric matrix of a vector that represents
// a vector cross product, For 3D vectors v and u:
// skewSymmetric(v) * u = crossProduct(v, u)
// See: equation (A4.5) in Hartley and Zisserman (2003)
template <class Derived>
inline Eigen::Matrix3d skewSymmetric(const Eigen::DenseBase<Derived>& v);

// Computes the Sampson distance for a set of points given a fundamental matrix
// See: equation (11.9) in Hartley and Zisserman (2003)
template <class DerivedSrc, class DerivedDst>
Eigen::Vector<double, DerivedSrc::ColsAtCompileTime> sampsonDistanceFromFundamentalMatrix(
    const Eigen::DenseBase<DerivedSrc>& src_points,
    const Eigen::DenseBase<DerivedDst>& dst_points,
    const Eigen::Matrix3d& F);

// Computes the symmetric epipolar distance for a set of points given a fundamental matrix
// See: equation (11.10) in Hartley and Zisserman (2003)
// The only difference here is that the function below returns the maximum of the two components
// of the sum defined in the equation in the book. So, we may say, that it returns the maximum
// of the epipolar distances in the two images. This is a better metric when it comes to rejecting
// the outliers. It can be interpreted that a point shouldn't be further away than a^2 pixels
// from the epipolar line in any of the images.
template <class DerivedSrc, class DerivedDst>
Eigen::Vector<double, DerivedSrc::ColsAtCompileTime> symmetricEpipolarDistance(
    const Eigen::DenseBase<DerivedSrc>& src_points,
    const Eigen::DenseBase<DerivedDst>& dst_points,
    const Eigen::Matrix3d& F);

// Normalizes points for numerical stability in DLT algorithms.
// The normalized point set is zero-mean and scaled, so that the average distance
// from the origin is equal to sqrt(Dim), where Dim is the dimension of the point.
// Furthermore, the resulting points are in homogeneous coordinates.
// The normalizing transformation is returned, as well.
// See: Section 4.4.4 Normalizing transformations in Hartley and Zisserman (2003)
template <class Derived>
void normalizePoints(const Eigen::DenseBase<Derived>& points,
                     Eigen::Matrix<double, Derived::RowsAtCompileTime + 1, Derived::ColsAtCompileTime>& X_normed,
                     Eigen::Matrix<double, Derived::RowsAtCompileTime + 1, Derived::RowsAtCompileTime + 1>& T);

// Normalizes a matrix for numerical stability
// The Frobenius norm of the resulting matrix is equal to sqrt(N*M)
// where NxM are the dimensions of the input matrix
template <class Derived>
Eigen::Matrix<double, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime> normalizeMatrix(
    const Eigen::DenseBase<Derived>& matrix);

// Finds the fundamental matrix given at least 8 point correspondences
// It implements the normalized 8-point algorithm.
// See: Section 11.2 The normalized 8-point algorithm in Hartley and Zisserman (2003)
template <class DerivedSrc, class DerivedDst>
Eigen::Matrix3d findFundamentalMatrix(const Eigen::DenseBase<DerivedSrc>& src_points,
                                      const Eigen::DenseBase<DerivedDst>& dst_points);

// Implements a robust method of finding the fundamental matrix using RANSAC
// See: step (iii) in the Algorithm 11.4 in Hartley and Zisserman (2003)
// However, the base algorithm used for estimating the fundamental matrix from a minimal
// sample is the 8-point algorithm (not 7-point algorithm as in the book).
// The distance measure is the squared maximum of the epipolar distances in both of the images.
template <class DerivedSrc, class DerivedDst>
Eigen::Matrix3d findFundamentalMatrixRansac(const Eigen::DenseBase<DerivedSrc>& src_points,
                                            const Eigen::DenseBase<DerivedDst>& dst_points,
                                            double ransac_thr,
                                            double confidence,
                                            size_t max_iters,
                                            size_t seed = std::random_device()());

// Finds the projection matrix given world and image points given at least 6
// 3D-2D correspondences. It implements a normalized DLT algorithm.
// See: step (i) & (iii) in the Algorithm 7.1 in Hartley and Zisserman (2003)
template <class Derived3D, class Derived2D>
Eigen::Matrix<double, 3, 4> findProjectionMatrix(const Eigen::DenseBase<Derived3D>& world_points,
                                                 const Eigen::DenseBase<Derived2D>& image_points);

// Implements a robust method of finding the projection matrix using RANSAC
// See: step (iii) in the Algorithm 11.4 in Hartley and Zisserman (2003)
// The base algorithm used for estimating the projection matrix from a minimal
// sample is the normalized DLT algorithm.
// The distance measure is the squared reprojection error.
template <class Derived3D, class Derived2D>
Eigen::Matrix<double, 3, 4> findProjectionMatrixRansac(const Eigen::DenseBase<Derived3D>& world_points,
                                                       const Eigen::DenseBase<Derived2D>& image_points,
                                                       double ransac_thr,
                                                       double confidence,
                                                       size_t max_iters,
                                                       size_t seed = std::random_device()());

// Triangulates 3D points from two sets of image points and projection matrices
// It implements a linear homogeneous triangulation method (DLT).
// See: Section 12.2 in Hartley and Zisserman (2003)
// Note: this function triangulates many 3D points having many pairs of 2D-point correspondences.
template <class Derived1, class Derived2>
Eigen::Matrix<double, 3, Derived1::ColsAtCompileTime> triangulatePoints(const Eigen::DenseBase<Derived1>& image_points1,
                                                                        const Eigen::DenseBase<Derived2>& image_points2,
                                                                        const Eigen::Matrix<double, 3, 4>& P1,
                                                                        const Eigen::Matrix<double, 3, 4>& P2);

// Triangulates a single 3D point from multiple sets of image points and projection matrices
// As above, it implements a linear homogeneous triangulation method (DLT).
// At least two sets of image point and projection matrix are required.
// Not to use Eigen tensors, the provided projection matrices should be flattened. For:
// Eigen::Matrix<double, 3, 4> P;
// Eigen::Vector<double, 12> P_flat = P.reshaped<Eigen::ColMajor>();
template <class Derived2D, class DerivedPFlat>
Eigen::Vector3d triangulatePoint(const Eigen::DenseBase<Derived2D>& image_points,
                                 const Eigen::DenseBase<DerivedPFlat>& Ps_flattened);

// Robust method for triangulation of a single 3D point with RANSAC given
// multiple sets of image points and projection matrices.
// The base algorithm used for estimating the 3D point is the above linear homogeneous method.
// The distance measure is the squared reprojection error.
template <class Derived2D, class DerivedPFlat>
Eigen::Vector3d triangulatePointRansac(const Eigen::DenseBase<Derived2D>& image_points,
                                       const Eigen::DenseBase<DerivedPFlat>& Ps_flattened,
                                       double ransac_thr,
                                       double confidence,
                                       size_t max_iters,
                                       size_t seed = std::random_device()());

} // namespace calib3d

#include <calib3d/calib_utils.hpp>
