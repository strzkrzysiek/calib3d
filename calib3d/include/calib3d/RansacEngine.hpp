#pragma once

#include <array>
#include <glog/logging.h>
#include <ranges>
#include <set>

namespace calib3d {

template <class Spec>
template <class Derived1, class Derived2>
typename Spec::ModelMatrix RansacEngine<Spec>::fit(const Eigen::DenseBase<Derived1>& points1,
                                                   const Eigen::DenseBase<Derived2>& points2,
                                                   double ransac_thr,
                                                   double confidence,
                                                   size_t max_iters,
                                                   size_t seed) {
  static_assert(Derived1::RowsAtCompileTime == Spec::Dim1);
  static_assert(Derived2::RowsAtCompileTime == Spec::Dim2);

  CHECK_EQ(points1.cols(), points2.cols());

  const size_t n_points = points1.cols();
  CHECK_GE(n_points, Spec::MinSampleSize);

  std::mt19937 rnd_gen(seed);

  const double squared_thr = ransac_thr * ransac_thr;

  std::vector<size_t> consensus_set_ids;
  double consensus_set_variance = 0.;

  for (size_t i = 0; i < max_iters; i++) {
    auto [sample1, sample2] = getRandomSample(points1, points2, rnd_gen);
    auto model_candidate = Spec::template fitModel(sample1, sample2);

    auto squared_distances = Spec::template distance(points1, points2, model_candidate);
    auto inlier_ids = fromBinMask((squared_distances.array() < squared_thr).eval());
    const size_t n_inliers = inlier_ids.size();

    if (n_inliers >= consensus_set_ids.size() && n_inliers > 0) {
      double inlier_variance = squared_distances(inlier_ids).sum() / n_inliers;

      if (n_inliers > consensus_set_ids.size() || inlier_variance < consensus_set_variance) {
        // New consensus set found
        consensus_set_ids = inlier_ids;
        consensus_set_variance = inlier_variance;

        VLOG(3) << "Iteration: " << i;
        VLOG(3) << "New consensus set: " << n_inliers << " (var: " << inlier_variance << ")";

        // Calculate new max_iters
        double inlier_ratio = static_cast<double>(n_inliers) / n_points;
        VLOG(3) << "Inlier ratio: " << inlier_ratio;

        auto new_max_iters = static_cast<size_t>(
            std::ceil(std::log(1. - confidence) / std::log(1. - std::pow(inlier_ratio, Spec::MinSampleSize))));
        if (new_max_iters < max_iters) {
          max_iters = new_max_iters;
          VLOG(3) << "New max iters: " << max_iters;
        }
      }
    }
  }

  if (consensus_set_ids.size() < Spec::MinSampleSize) {
    LOG(WARNING) << "Consensus set (" << consensus_set_ids.size()
                 << ") is smaller than the minimal required sample size";
    LOG(WARNING) << "Consider increasing the max_iters argument.";
    LOG(WARNING) << "Fitting with all provided data.";

    return Spec::template fitModel(points1, points2);
  }

  VLOG(2) << "Final estimation using " << consensus_set_ids.size() << " inliers";

  return Spec::template fitModel(points1(Eigen::all, consensus_set_ids), points2(Eigen::all, consensus_set_ids));
}

template <class Spec>
template <class Derived1, class Derived2, class RNG>
auto RansacEngine<Spec>::getRandomSample(const Eigen::DenseBase<Derived1>& points1,
                                         const Eigen::DenseBase<Derived2>& points2,
                                         RNG& rnd) {
  const size_t n_points = points1.cols();
  std::uniform_int_distribution<size_t> idx_dist(0, n_points - 1);

  std::set<size_t> index_set;
  while (index_set.size() < Spec::MinSampleSize) {
    index_set.insert(idx_dist(rnd));
  }
  std::array<size_t, Spec::MinSampleSize> index_array;
  std::copy(index_set.begin(), index_set.end(), index_array.begin());

  return std::make_pair(points1(Eigen::all, index_array), points2(Eigen::all, index_array));
}

template <class Spec>
std::vector<size_t> RansacEngine<Spec>::fromBinMask(const Eigen::Array<bool, Eigen::Dynamic, 1>& mask) {
  auto indices = std::views::iota(0, mask.size()) | std::views::filter([&mask](const auto& i) { return mask(i); });
  std::vector<size_t> result;
  std::ranges::copy(indices, std::back_inserter(result));
  return result;
}

} // namespace calib3d
