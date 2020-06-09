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

#ifndef EULER_CLIENT_CLIENT_MANAGER_H_
#define EULER_CLIENT_CLIENT_MANAGER_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <unordered_set>

#include "euler/common/logging.h"
#include "euler/common/server_monitor.h"
#include "euler/client/rpc_client.h"
#include "euler/core/graph/graph_meta.h"

namespace euler {
class ClientManager {
 public:
  static void Init(const GraphConfig& config) {
    static ClientManager* temp = new ClientManager(config);
    instance_ = temp;
  }

  static ClientManager* GetInstance() {
    if (instance_ == nullptr || !init_succ_) {
      EULER_LOG(ERROR) << "Init failed";
      return nullptr;
    }
    return instance_;
  }

  ClientManager(ClientManager const&) = delete;

  void operator=(ClientManager const&) = delete;

  std::shared_ptr<RpcClient> GetClient(int32_t shard_id) {
    if (shard_id < 0 ||
        static_cast<size_t>(shard_id) >= clients_.size()) {
      EULER_LOG(ERROR) << "Invalid shard id";
      return nullptr;
    }
    return clients_[shard_id];
  }

  bool RetrieveShardMeta(int32_t shard_index, const std::string& key,
                         std::unordered_set<std::string>* graph_label);

  bool RetrieveShardMeta(int32_t shard_index, const std::string& key,
                         std::vector<std::vector<float>>* weights);

  bool RetrieveMeta(const std::string& key, std::string* value);

  const GraphMeta& graph_meta() const { return meta_; }

 private:
  explicit ClientManager(const GraphConfig& config) {
    std::string zk_server, zk_path;
    config.Get("zk_server", &zk_server);
    config.Get("zk_path", &zk_path);
    server_monitor_ = GetServerMonitor(zk_server, zk_path);
    int32_t shard_number = 0;
    if (!server_monitor_->GetNumShards(&shard_number) || shard_number == 0) {
      EULER_LOG(ERROR) << "Retrieve shard info from server failed!";
    } else {
      if (!InitGraphMeta()) {
        return;
      }

      for (int32_t i = 0; i < shard_number; ++i) {
        clients_.push_back(
            std::move(NewRpcClient(server_monitor_, i, config)));
      }
      EULER_LOG(INFO) << "shard number: " << shard_number;
      init_succ_ = true;
    }
  }

  bool InitGraphMeta();

  static bool init_succ_;

  static ClientManager* instance_;

  std::shared_ptr<ServerMonitor> server_monitor_;

  std::vector<std::shared_ptr<RpcClient>> clients_;

  GraphMeta meta_;
};

}  // namespace euler

#endif  // EULER_CLIENT_CLIENT_MANAGER_H_
