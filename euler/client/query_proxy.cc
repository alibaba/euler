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

#include "euler/client/query_proxy.h"

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>

#include "euler/common/env.h"
#include "euler/core/framework/executor.h"
#include "euler/core/framework/tensor.h"
#include "euler/core/index/index_manager.h"
#include "euler/client/query.h"
#include "euler/client/client_manager.h"
#include "euler/parser/compiler.h"

namespace euler {

static const char* kRemoteGraphMode = "remote";
static const char* kGraphPartitionMode = "graph_partition";

QueryProxy* QueryProxy::instance_ = nullptr;

bool QueryProxy::Init(const GraphConfig& config) {
  std::string opt_type_str = "";
  if (!config.Get("mode", &opt_type_str)) {
    EULER_LOG(FATAL) <<"no mode in graph config";
    return false;
  }

  OptimizerType type;
  const GraphMeta* meta = nullptr;
  std::vector<std::vector<float>> shard_node_weight, shard_edge_weight;
  std::vector<std::string> graph_label;
  if (opt_type_str == kRemoteGraphMode) {
    type = distribute;
  } else if (opt_type_str == kGraphPartitionMode) {
    type = graph_partition;
  } else {
    type = local;
  }

  int32_t shard_num = 1;
  std::string index_info;
  if (type == distribute || type == graph_partition) {
    ClientManager::Init(config);
    if (!config.Get("shard_num", &shard_num)) {
      EULER_LOG(ERROR) << "no shard_num in graph_config";
      return false;
    }

    ClientManager* cm = ClientManager::GetInstance();
    meta = &cm->graph_meta();

    if (!cm->RetrieveMeta("index_info", &index_info)) {
      EULER_LOG(ERROR) << "index info error";
    }

    Compiler::Init(shard_num, type, index_info);

    // graph label
    std::unordered_set<std::string> graph_label_set;
    for (int32_t i = 0; i < shard_num; ++i) {
      if (!cm->RetrieveShardMeta(i, "graph_label", &graph_label_set)) {
        EULER_LOG(ERROR) << "no graph label info, shard no: " << i;
      }
    }

    graph_label.reserve(graph_label_set.size());
    for (auto it = graph_label_set.begin(); it != graph_label_set.end();
         ++it) {
      EULER_LOG(INFO) << "graph label: " << *it;
      graph_label.push_back(*it);
    }

    // shard node weight
    bool shard_meta_ok = true;
    for (int32_t i = 0; i < shard_num; ++i) {
      if (!cm->RetrieveShardMeta(
          i, "node_sum_weight", &shard_node_weight)) {
        shard_meta_ok = false;
        EULER_LOG(ERROR) << "node sum weight error";
      }
    }
    if (shard_meta_ok) {
      std::vector<float> n_weight_sum(shard_num, 0.0);
      for (int32_t i = 0; i < shard_num; ++i) {
        n_weight_sum[i] = 0.0;
        for (size_t j = 0; j < shard_node_weight.size(); ++j) {
          n_weight_sum[i] += shard_node_weight[j][i];
        }
      }
      shard_node_weight.push_back(n_weight_sum);
      for (size_t i = 0; i < shard_node_weight.size(); ++i) {
        float sum_weight = 0;
        for (int32_t j = 0; j < shard_num; ++j) {
          sum_weight += shard_node_weight[i][j];
        }
        shard_node_weight[i].resize(shard_num + 1);
        shard_node_weight[i][shard_num] = sum_weight;
      }
    }
    // shard edge weight
    shard_meta_ok = true;
    for (int32_t i = 0; i < shard_num; ++i) {
      if (!cm->RetrieveShardMeta(
          i, "edge_sum_weight", &shard_edge_weight)) {
        shard_meta_ok = false;
        EULER_LOG(ERROR) << "edge sum weight error";
      }
    }
    if (shard_meta_ok) {
      std::vector<float> e_weight_sum(shard_num, 0.0);
      for (int32_t i = 0; i < shard_num; ++i) {
        e_weight_sum[i] = 0.0;
        for (size_t j = 0; j < shard_edge_weight.size(); ++j) {
          e_weight_sum[i] += shard_edge_weight[j][i];
        }
      }
      shard_edge_weight.push_back(e_weight_sum);
      for (size_t i = 0; i < shard_edge_weight.size(); ++i) {
        float sum_weight = 0;
        for (int32_t j = 0; j < shard_num; ++j) {
          sum_weight += shard_edge_weight[i][j];
        }
        shard_edge_weight[i].resize(shard_num + 1);
        shard_edge_weight[i][shard_num] = sum_weight;
      }
    }
  } else {
    // init graph
    std::string data_path = "";
    if (!config.Get("data_path", &data_path)) {
      EULER_LOG(ERROR) << "no data_path in graph_config";
      return false;
    }

    std::string sampler_type_info = "";
    if (!config.Get("sampler_type", &sampler_type_info)) {
      return false;
    }

    std::string data_type_info = "";
    if (!config.Get("data_type", &data_type_info)) {
      return false;
    }

    std::unique_ptr<FileIO> data_dir;
    if (!Env::Default()->NewFileIO(data_path, true, &data_dir).ok() ||
        !data_dir->initialized() || !data_dir->IsDirectory()) {
      EULER_LOG(ERROR) << "No such directory found, path: " << data_path;
      return false;
    }

    auto& graph = Graph::Instance();
    if (!graph.Init(0, 1, sampler_type_info, data_path, data_type_info).ok()) {
      EULER_LOG(FATAL) << "graph data error!";
      return false;
    }

    auto& index_manager = IndexManager::Instance();
    if (!data_dir->ListDirectory([](const std::string &filename) {
          return filename == "Index";
        }).empty()) {
      std::string index_dir = JoinPath(data_path, "Index");
      if (!index_manager.Deserialize(index_dir).ok()) {
        EULER_LOG(FATAL) << "index data error!";
      }
    }

    index_info = index_manager.GetIndexInfo()[0]["index_info"];
    Compiler::Init(shard_num, local, index_info);
    meta = &Graph::Instance().graph_meta();
    graph_label = graph.GetGraphLabel();
  }

  static QueryProxy* temp = new QueryProxy(shard_num);
  instance_ = temp;
  instance_->meta_ = *meta;
  instance_->shard_edge_weight_ = shard_edge_weight;
  instance_->shard_node_weight_ = shard_node_weight;
  instance_->graph_label_ = graph_label;

  EULER_LOG(INFO) << "QueryProxy load successfully!\n"
                  << "GraphMeta {\n" << meta->ToString() << "}\n";

  return true;
}

QueryProxy::QueryProxy(int32_t shard_num) {
  shard_num_ = shard_num;
  compiler_ = Compiler::GetInstance();
  env_ = Env::Default();
  tp_ = env_->StartThreadPool("client_thread_pool", 8);
}

std::unordered_map<std::string, Tensor*>
QueryProxy::RunGremlin(Query* query,
                       const std::vector<std::string>& result_name) {
  DAG* dag = nullptr;
  if (query->SingleOpQuery()) {
    dag = compiler_->Op2DAG(query->op_name_,
                            query->alias_,
                            query->output_num_,
                            query->input_tensor_names_,
                            query->norm_attr_names_);
  } else {
    dag = compiler_->Compile(query->gremlin_);
  }
  if (dag == nullptr) {
    EULER_LOG(FATAL) << "parse error: " << query->gremlin_;
    std::unordered_map<std::string, Tensor*> error_map;
    return error_map;
  }
  Executor executor(dag, tp_, query->ctx_.get());
  executor.Run();
  return query->GetResult(result_name);
}

void QueryProxy::RunAsyncGremlin(Query* query, DoneCallback callback) {
  DAG* dag = nullptr;
  if (query->SingleOpQuery()) {
    dag = compiler_->Op2DAG(query->op_name_,
                            query->alias_,
                            query->output_num_,
                            query->input_tensor_names_,
                            query->norm_attr_names_);
  } else {
    dag = compiler_->Compile(query->gremlin_);
  }
  if (dag == nullptr) {
    EULER_LOG(FATAL) << "parse error: " << query->gremlin_;
    return;
  }
  Executor* executor = new Executor(dag, tp_, query->ctx_.get());
  executor->Run(
      [executor, callback](){
        callback();
        delete executor;
      });
}

}  // namespace euler
