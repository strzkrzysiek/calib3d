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

      dataset.cameras.emplace(cam_id, std::move(bundle));
    }

    for (auto& [pt_id_str, pt_json] : root_json["worldPoints"].items()) {
      size_t pt_id = std::stoul(pt_id_str);

      Vec3 world_point;
      world_point(0) = pt_json[0];
      world_point(1) = pt_json[1];
      world_point(2) = pt_json[2];
      dataset.world_points[pt_id] = world_point;
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
    return false;
  }

  return true;
}

} // namespace calib3d
