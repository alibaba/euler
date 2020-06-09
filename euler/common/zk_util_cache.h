/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef EULER_COMMON_ZK_UTIL_CACHE_H_
#define EULER_COMMON_ZK_UTIL_CACHE_H_

#include <functional>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <unordered_map>

namespace euler {

struct ZkInfo {
  ZkInfo(const std::string &addr, const std::string &path)
      : addr(addr), path(path) { }

  bool operator==(const ZkInfo &other) const {
    return addr == other.addr && path == other.path;
  }

  std::string addr;
  std::string path;
};

struct HashZkInfo {
  size_t operator()(const ZkInfo &zk_info) const {
    return std::hash<std::string>()(zk_info.addr + ":" + zk_info.path);
  }
};

template <typename ZkUtil>
std::shared_ptr<ZkUtil> GetOrCreate(
    const std::string &zk_addr, const std::string &zk_path) {
  ZkInfo zk_info(zk_addr, zk_path);

  static std::mutex zk_utils_mu;
  std::lock_guard<std::mutex> lock(zk_utils_mu);

  static std::unordered_map<ZkInfo, std::shared_ptr<ZkUtil>,
                            HashZkInfo> zk_utils;
  auto iter = zk_utils.find(zk_info);
  if (iter != zk_utils.end()) {
    return iter->second;
  }

  std::shared_ptr<ZkUtil> zk_util(new ZkUtil(zk_addr, zk_path));
  if (zk_util->Initialize()) {
    zk_utils.emplace(zk_info, zk_util);
    return zk_util;
  } else {
    return nullptr;
  }
}

}  // namespace euler

#endif  // EULER_COMMON_ZK_UTIL_CACHE_H_
