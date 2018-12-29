/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef EULER_COMMON_ZK_SERVER_REGISTER_H_
#define EULER_COMMON_ZK_SERVER_REGISTER_H_

#include <unordered_map>
#include <string>
#include <mutex>  // NOLINT

#include "zookeeper.h"  // NOLINT

#include "euler/common/server_register.h"
#include "euler/common/zk_util_cache.h"

namespace euler {
namespace common {

class ZkServerRegister : public ServerRegister {
  friend std::shared_ptr<ZkServerRegister> GetOrCreate<ZkServerRegister>(
      const std::string &, const std::string &);

 public:
  ~ZkServerRegister() override;

  bool RegisterShard(size_t shard_index, const Server &server,
                     const Meta &meta, const Meta &shard_meta) override;
  bool DeregisterShard(size_t shard_index, const Server &server) override;

 private:
  static void Watcher(zhandle_t *zh, int type, int state, const char *path,
                      void *data);

  ZkServerRegister(const std::string &zk_addr, const std::string &zk_path)
      : zk_addr_(zk_addr), zk_path_(zk_path), zk_handle_(nullptr) { }

  bool Initialize() override;

  std::string zk_addr_;
  std::string zk_path_;

  std::mutex zk_mu_;
  zhandle_t *zk_handle_;

  std::mutex mu_;
  std::unordered_map<std::string, std::string> registered_;
};

}  // namespace common
}  // namespace euler

#endif  // EULER_COMMON_ZK_SERVER_REGISTER_H_
