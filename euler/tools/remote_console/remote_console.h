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

#ifndef EULER_TOOLS_REMOTE_CONSOLE_REMOTE_CONSOLE_H_
#define EULER_TOOLS_REMOTE_CONSOLE_REMOTE_CONSOLE_H_

#include <string>
#include <vector>

#include "euler/client/query_proxy.h"
#include "euler/client/query.h"
#include "euler/common/data_types.h"
#include "euler/core/framework/op_kernel.h"

namespace euler {

class RemoteConsole {
 public:
  RemoteConsole(std::string zk_server,
                std::string zk_path,
                int32_t shard_num) {
    GraphConfig graph_config;
    graph_config.Add("zk_server", zk_server);
    graph_config.Add("zk_path", zk_path);
    graph_config.Add("num_retries", 1);
    graph_config.Add("shard_num", shard_num);
    graph_config.Add("mode", "remote");
    QueryProxy::Init(graph_config);
    query_proxy_ = QueryProxy::GetInstance();
  }

  void GetNodeNeighbor(uint64_t node_id, std::string nb_type,
      std::vector<uint64_t>* nbs, std::vector<float>* edge_weights,
      std::vector<int32_t>* types);

  void GetNodeFeature(uint64_t node_id, std::string feature_name,
                      std::vector<uint64_t>* feature);

  void GetNodeFeature(uint64_t node_id, std::string feature_name,
                      std::vector<float>* feature);

 private:
  QueryProxy* query_proxy_;
};

}  // namespace euler

#endif  // EULER_TOOLS_REMOTE_CONSOLE_REMOTE_CONSOLE_H_
