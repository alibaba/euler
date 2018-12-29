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

#ifndef EULER_CLIENT_TESTING_SIMPLE_SERVER_MONITOR_H_
#define EULER_CLIENT_TESTING_SIMPLE_SERVER_MONITOR_H_

#include <unordered_map>
#include <string>

#include "euler/common/server_monitor.h"
#include "euler/common/server_register.h"

namespace euler {
namespace client {
namespace testing {

class SimpleServerMonitor : public common::ServerMonitorBase,
                            public common::ServerRegister {
 public:
  SimpleServerMonitor() = default;

  bool Initialize() override { return true; }

  bool RegisterShard(size_t shard_index, const common::Server &server,
                     const common::Meta &meta,
                     const common::Meta &shard_meta) override {
    bool success;
    {
      std::lock_guard<std::mutex> lock(servers_mu_);
      success = servers_[shard_index].emplace(server).second;
    }

    if (success) {
      AddShardServer(shard_index, server);
      UpdateMeta(meta);
      UpdateShardMeta(shard_index, shard_meta);
      return true;
    } else {
      return false;
    }
  }

  bool DeregisterShard(size_t shard_index,
                       const common::Server &server) override {
    bool success;
    {
      std::lock_guard<std::mutex> lock(servers_mu_);
      success = servers_[shard_index].erase(server) > 0;
    }

    if (success) {
      RemoveShardServer(shard_index, server);
      return true;
    } else {
      return false;
    }
  }

 private:
  std::mutex servers_mu_;
  std::unordered_map<size_t, std::unordered_set<std::string>> servers_;
};

bool NewSimpleMonitor(std::shared_ptr<common::ServerRegister> *regs,
                      std::shared_ptr<common::ServerMonitor> *monitor) {
  auto simple_monitor = std::make_shared<SimpleServerMonitor>();
  if (simple_monitor->Initialize()) {
    *regs = simple_monitor;
    *monitor = simple_monitor;
    return true;
  } else {
    return false;
  }
}

}  // namespace testing
}  // namespace client
}  // namespace euler

#endif  // EULER_CLIENT_TESTING_SIMPLE_SERVER_MONITOR_H_
