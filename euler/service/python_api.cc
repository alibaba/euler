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

#include <unordered_map>
#include <string>

#include "euler/service/graph_service.h"

namespace euler {
namespace service {

extern "C" {

  extern void StartService(
      const char* directory,
      const char* loader_type,
      const char* hdfs_addr,
      const char* hdfs_port,
      const char* shard_idx,
      const char* shard_num,
      const char* zk_addr,
      const char* zk_path,
      const char* global_sampler_type,
      const char* graph_type,
      const char* server_thread_num) {
    std::unordered_map<std::string, std::string> conf;
    conf["directory"] = directory;
    conf["loader_type"] = loader_type;
    conf["hdfs_addr"] = hdfs_addr;
    conf["hdfs_port"] = hdfs_port;
    conf["shard_idx"] = shard_idx;
    conf["shard_num"] = shard_num;
    conf["zk_addr"] = zk_addr;
    conf["zk_path"] = zk_path;
    conf["global_sampler_type"] = global_sampler_type;
    conf["graph_type"] = graph_type;
    conf["server_thread_num"] = server_thread_num;
    euler::service::StartService(conf);
  }
}

}  // namespace service
}  // namespace euler
