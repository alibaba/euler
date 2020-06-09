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

#include <stdlib.h>

#include <unordered_map>
#include <string>

#include "euler/service/grpc_server.h"
#include "euler/common/net_util.h"
#include "euler/common/logging.h"

namespace euler {

extern "C" {

  void* StartService(
      const char* directory,
      const char* shard_idx,
      const char* shard_num,
      const char* zk_addr,
      const char* zk_path,
      const char* load_data_type,
      const char* global_sampler_type,
      const char* server_thread_num) {
    ServerDef server_def =
        {"grpc", atoi(shard_idx), atoi(shard_num), {}};

    server_def.options.emplace("port", std::to_string(GetFreePort()));
    server_def.options.emplace("data_path", directory);
    server_def.options.emplace("zk_server", zk_addr);
    server_def.options.emplace("zk_path", zk_path);
    server_def.options.emplace("load_data_type", load_data_type);
    server_def.options.emplace("global_sampler_type", global_sampler_type);
    server_def.options.emplace("num_threads", server_thread_num);

    std::unique_ptr<ServerInterface> server_;
    auto s = NewServer(server_def, &server_);
    if (!s.ok()) {
      EULER_LOG(ERROR) << "Create euler server failed, status: " << s;
      return nullptr;
    }

    s = server_->Start();
    if (!s.ok()) {
      EULER_LOG(ERROR) << "Start euler server failed, status: " << s;
      return nullptr;
    }

    return server_.release();
  }

}

}  // namespace euler
