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

#ifndef EULER_CLIENT_QUERY_PROXY_H_
#define EULER_CLIENT_QUERY_PROXY_H_

#include <unordered_map>
#include <vector>
#include <string>

#include "euler/client/graph_config.h"
#include "euler/common/logging.h"
#include "euler/common/env.h"
#include "euler/common/str_util.h"
#include "euler/core/graph/graph.h"
#include "euler/core/graph/graph_builder.h"
#include "euler/core/framework/types.h"
#include "euler/core/graph/graph_meta.h"
#include "euler/parser/optimize_type.h"

namespace euler {

class Compiler;
class Query;
class Tensor;

class QueryProxy {
 public:
  typedef std::function<void()> DoneCallback;

  QueryProxy(const QueryProxy&) = delete;
  void operator=(const QueryProxy&) = delete;

  static bool Init(const GraphConfig& config);

  std::unordered_map<std::string, Tensor*>
  RunGremlin(
      Query* query, const std::vector<std::string>& result_name);

  void RunAsyncGremlin(Query* query, DoneCallback callback);

  static QueryProxy* GetInstance() {
    if (instance_ == nullptr) {
      EULER_LOG(ERROR) << "Init failed";
    }
    return instance_;
  }

  const GraphMeta& graph_meta() const { return meta_; }

  int32_t GetShardNum() {
    return shard_num_;
  }

  const std::vector<std::vector<float>>& GetShardNodeWeight() {
    return shard_node_weight_;
  }

  const std::vector<std::vector<float>>& GetShardEdgeWeight() {
    return shard_edge_weight_;
  }

  const std::vector<std::string>& GetGraphLabel() {
    return graph_label_;
  }

 private:
  int32_t shard_num_;
  Compiler* compiler_;
  Env* env_;
  ThreadPool* tp_;
  GraphMeta meta_;
  // [node_types_num + 1, shards_num + 1]
  std::vector<std::vector<float>> shard_node_weight_;
  // [edge_types_num + 1, shards_num + 1]
  std::vector<std::vector<float>> shard_edge_weight_;
  std::vector<std::string> graph_label_;
  static QueryProxy* instance_;

  explicit QueryProxy(int32_t shard_num);
};

}  // namespace euler

#endif  // EULER_CLIENT_QUERY_PROXY_H_
