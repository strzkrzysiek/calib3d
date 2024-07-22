#include <calib3d/Dataset.h>

#include <exception>
#include <fstream>
#include <glog/logging.h>
#include <nlohmann/json.hpp>

namespace calib3d {

bool Dataset::loadFromJson(const std::string& filename) {
  cameras.clear();
  world_points.clear();

  std::ifstream file(filename);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return false;
  }

  try {
    nlohmann::json root_json;
    file >> root_json;

    for (const auto& [cam_id_str, camera_json] : root_json["cameras"].items()) {
      size_t cam_id = std::stoul(cam_id_str);

      CameraBundle bundle;
      auto& calib = bundle.calib;

      const auto& calib_json = camera_json["calib"];

      const auto& extrinsics_json = calib_json["extrinsics"];
      Mat3x4 pose;
      for (int i = 0; i < 3; i++) {
        const auto& row = extrinsics_json[i];
        for (int j = 0; j < 4; j++) {
          pose(i, j) = row[j].get<double>();
        }
      }

      calib.world2cam.setRotationMatrix(pose.leftCols<3>());
      calib.world2cam.translation() = pose.col(3);

      const auto& intrinsics_json = calib_json["intrinsics"];
      Mat3 K;
      for (int i = 0; i < 3; i++) {
        const auto& row = intrinsics_json[i];
        for (int j = 0; j < 3; j++) {
          K(i, j) = row[j].get<double>();
        }
      }
      calib.intrinsics.principal_point = K.topRightCorner<2, 1>();
      calib.intrinsics.focal_length = K(0, 0);

      const auto& size_json = calib_json["size"];
      calib.size << size_json[0].get<int>(), size_json[1].get<int>();

      for (auto& [obs_id_str, obs_json] : camera_json["observations"].items()) {
        size_t obs_id = std::stoul(obs_id_str);

        bundle.observations[obs_id] << obs_json[0].get<double>(), obs_json[1].get<double>();
      }

      cameras.emplace(cam_id, std::move(bundle));
    }

    for (auto& [pt_id_str, pt_json] : root_json["worldPoints"].items()) {
      size_t pt_id = std::stoul(pt_id_str);

      Vec3 world_point;
      world_point(0) = pt_json[0];
      world_point(1) = pt_json[1];
      world_point(2) = pt_json[2];
      world_points[pt_id] = world_point;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();

    return false;
  }

  return true;
}

void Dataset::addObservationNoise(double noise, size_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> gaussian_noise(0.0, noise);

  for (auto& [cam_id, camera_bundle] : cameras) {
    for (auto& [pt_id, image_pt] : camera_bundle.observations) {
      image_pt.x() += gaussian_noise(rng);
      image_pt.y() += gaussian_noise(rng);
    }
  }
}

void Dataset::addObservationOutliers(double inlier_prob, size_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> inlier_dist(0., 1.0);

  for (auto& [cam_id, camera_bundle] : cameras) {
    std::uniform_real_distribution<double> x_dist(0., camera_bundle.calib.size.x());
    std::uniform_real_distribution<double> y_dist(0., camera_bundle.calib.size.y());

    for (auto& [pt_id, image_pt] : camera_bundle.observations) {
      if (inlier_dist(rng) < inlier_prob) {
        continue;
      }

      image_pt.x() = x_dist(rng);
      image_pt.y() = y_dist(rng);
    }
  }
}

std::vector<PointId> Dataset::getCommonPointIds(const std::vector<CamId>& cam_ids) const {
  CHECK(!cam_ids.empty());

  std::vector<PointId> common_pt_ids;

  const auto& cam0_obs = cameras.at(cam_ids[0]).observations;
  for (const auto& [pt_id, image_pt] : cam0_obs) {
    bool is_common_point = true;
    for (size_t i = 1; i < cam_ids.size(); i++) {
      if (!cameras.at(cam_ids[i]).observations.contains(pt_id)) {
        is_common_point = false;
        break;
      }
    }
    if (is_common_point) {
      common_pt_ids.push_back(pt_id);
    }
  }

  return common_pt_ids;
}

Mat2X Dataset::getImagePointArray(calib3d::CamId cam_id, const std::vector<PointId>& pt_ids) const {
  Mat2X image_pts(2, pt_ids.size());
  const auto& observations = cameras.at(cam_id).observations;

  for (size_t i = 0; i < pt_ids.size(); i++) {
    image_pts.col(i) = observations.at(pt_ids[i]);
  }

  return image_pts;
}

Mat3X Dataset::getWorldPointArray(const std::vector<PointId>& pt_ids) const {
  Mat3X world_pts(3, pt_ids.size());

  for (size_t i = 0; i < pt_ids.size(); i++) {
    world_pts.col(i) = world_points.at(pt_ids[i]);
  }

  return world_pts;
}


} // namespace calib3d
