#include <calib3d/dataset.h>

#include <exception>
#include <fstream>
#include <glog/logging.h>
#include <nlohmann/json.hpp>

namespace calib3d {

bool loadJsonDataset(const std::string& filename, Dataset& dataset) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return false;
  }

  try {
    nlohmann::json json_data;
    file >> json_data;

    for (const auto& [cam_id_str, camera_data] : json_data["cameras"].items()) {
      size_t cam_id = std::stoul(cam_id_str);

      CameraBundle bundle;

      const auto& calib_data = camera_data["calib"];

      const auto& extrinsics_data = calib_data["extrinsics"];
      Eigen::Matrix<double, 3, 4> pose;
      for (int i = 0; i < 3; i++) {
        const auto& row = extrinsics_data[i];
        for (int j = 0; j < 4; j++) {
          pose(i, j) = row[j].get<double>();
        }
      }
      bundle.extrinsics.world2cam_rot = Eigen::Quaterniond(pose.leftCols<3>());
      bundle.extrinsics.world_in_cam_pos = pose.col(3);

      const auto& intrinsics_data = calib_data["intrinsics"];
      Eigen::Matrix<double, 3, 3> K;
      for (int i = 0; i < 3; i++) {
        const auto& row = intrinsics_data[i];
        for (int j = 0; j < 3; j++) {
          K(i, j) = row[j].get<double>();
        }
      }
      bundle.intrinsics.principal_point = K.topRightCorner<2, 1>();
      bundle.intrinsics.f = K(0, 0);

      const auto& size_data = calib_data["size"];
      bundle.size << size_data[0].get<int>(), size_data[1].get<int>();

      for (auto& [obs_id_str, obs_data] : camera_data["observations"].items()) {
        size_t obs_id = std::stoul(obs_id_str);

        bundle.observations[obs_id] << obs_data[0].get<double>(), obs_data[1].get<double>();
      }

      dataset.cameras.emplace(cam_id, std::move(bundle));
    }

    for (auto& [pt_id_str, pt_data] : json_data["worldPoints"].items()) {
      size_t pt_id = std::stoul(pt_id_str);

      Eigen::Vector3d world_point;
      world_point(0) = pt_data[0];
      world_point(1) = pt_data[1];
      world_point(2) = pt_data[2];
      dataset.world_points[pt_id] = world_point;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return false;
  }

  return true;
}

} // namespace calib3d
