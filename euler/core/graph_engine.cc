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

#include "euler/core/graph_engine.h"

#include <stdlib.h>
#include <dirent.h>
#include <unordered_set>
#include <unordered_map>
#include <iostream>

#include "glog/logging.h"

#include "euler/common/string_util.h"
#include "euler/common/file_io_factory.h"
#include "euler/core/compact_graph_factory.h"
#include "euler/core/fast_graph_factory.h"

namespace euler {
namespace core {
int32_t GetPartitionIdx(std::string file_path) {
  std::vector<std::string> file_info;
  euler::common::split_string(file_path, '/', &file_info);
  std::vector<std::string> file_name;
  euler::common::split_string(file_info.back(), '.', &file_name);
  std::vector<std::string> partition_info;
  euler::common::split_string(file_name[0], '_', &partition_info);
  return atoi(partition_info.back().c_str());
}

bool GraphEngine::Initialize(std::unordered_map<std::string,
     std::string> conf) {
  std::string directory = conf["directory"];
  LoaderType loader_type = conf["loader_type"] == "local" ? local : hdfs;
  std::string hdfs_addr = conf["hdfs_addr"];
  int32_t hdfs_port = atoi(conf["hdfs_port"].c_str());
  int32_t shard_idx = atoi(conf["shard_idx"].c_str());
  int32_t shard_num = atoi(conf["shard_num"].c_str());
  GlobalSamplerType global_sampler_type =
      conf["global_sampler_type"] == "node" ? node : (
      conf["global_sampler_type"] == "edge" ? edge : (
      conf["global_sampler_type"] == "none" ? none : all));
  // using graph builder init graph
  euler::common::FileIOFactory* file_io_factory = nullptr;
  std::function<bool(std::string input)> full_file_filter_fn = {};
  if (loader_type == local) {
    if (euler::common::factory_map.find("local") ==
        euler::common::factory_map.end()) {
      LOG(ERROR) << "no local file io factory register";
      return false;
    }
    file_io_factory = euler::common::factory_map["local"];
    full_file_filter_fn = [](std::string input) {
      std::vector<std::string> file_name_vec;
      int32_t cnt = euler::common::split_string(input, '.',
                                                &file_name_vec);
      if (cnt == 2 && file_name_vec.back() == "dat") return true;
      return false;
    };
  } else {
    if (euler::common::factory_map.find("hdfs") ==
        euler::common::factory_map.end()) {
      LOG(ERROR) << "no hdfs file io factory register";
      return false;
    }
    file_io_factory = euler::common::factory_map["hdfs"];
  }
  std::vector<std::string> file_list;
  std::vector<std::string> full_file_list =
      file_io_factory->ListFile(hdfs_addr, hdfs_port, directory,
                                full_file_filter_fn);
  // using shard idx and shard num to get the files that need to be loaded
  partition_num_ = 0;
  for (size_t i = 0; i < full_file_list.size(); ++i) {
    int32_t temp = GetPartitionIdx(full_file_list[i]) + 1;
    partition_num_ = temp > partition_num_ ? temp : partition_num_;
  }
  if (shard_idx == -1) {
    file_list = full_file_list;
  } else {
    std::unordered_set<int32_t> filter;
    int32_t p_idx = 0;
    for (int32_t m = 0; p_idx < partition_num_; ++m) {
      p_idx = m * shard_num + shard_idx;
      if (p_idx < partition_num_) {
        filter.insert(p_idx);
      }
    }
    for (size_t i = 0; i < full_file_list.size(); ++i) {
      int32_t partition_idx = GetPartitionIdx(full_file_list[i]);
      if (filter.find(partition_idx) != filter.end()) {
        file_list.push_back(full_file_list[i]);
      }
    }
  }
  if (file_list.empty()) {
    LOG(ERROR) << "no file valid in dir: " << directory;
    return false;
  }
  GraphFactory* graph_factory = nullptr;
  if (graph_type_ == compact) {
    graph_factory = new CompactGraphFactory();
  } else {
    graph_factory = new FastGraphFactory();
  }
  GraphBuilder graph_builder(graph_factory);
  graph_ = graph_builder.BuildGraph(
      file_list, loader_type, hdfs_addr, hdfs_port, global_sampler_type);

  if (graph_ != nullptr) {
    return true;
  } else {
    return false;
  }
}

bool GraphEngine::Initialize(const std::string& directory) {
  std::unordered_map<std::string, std::string> conf;
  conf["directory"] = directory;
  conf["loader_type"] = "local";
  conf["hdfs_addr"] = "";
  conf["hdfs_port"] = "0";
  conf["shard_idx"] = "-1";
  conf["shard_num"] = "0";
  conf["global_sampler_type"] = "all";
  return Initialize(conf);
}

std::vector<euler::common::NodeID>
GraphEngine::SampleNode(int32_t node_type, int32_t count) const {
  if (!graph_->GlobalNodeSamplerOk()) {
    graph_->BuildGlobalSampler();
  }
  return graph_->SampleNode(node_type, count);
}

std::vector<euler::common::EdgeID>
GraphEngine::SampleEdge(int32_t edge_type, int32_t count) const {
  if (!graph_->GlobalEdgeSamplerOk()) {
    graph_->BuildGlobalEdgeSampler();
  }
  return graph_->SampleEdge(edge_type, count);
}

std::vector<int32_t> GraphEngine::GetNodeType(
    const std::vector<euler::common::NodeID>& node_ids) const {
  std::vector<int32_t> types(node_ids.size());
  for (size_t i = 0; i < node_ids.size(); ++i) {
    Node* node = graph_->GetNodeByID(node_ids[i]);
    if (node == nullptr) {
      types[i] = -1;
    } else {
      types[i] = node->GetType();
    }
  }
  return types;
}

#define GET_FEATURE(GET_STRUCT_METHOD, GET_FEATURE_METHOD, STRUCT_TYPE, \
                    F_NUMS_PTR, F_VALUES_PTR, IDS, FIDS, GET_FV_NUM) {  \
  STRUCT_TYPE* scout = nullptr;                                         \
  int32_t fv_num = 1;                                                   \
  for (size_t i = 0; i < IDS.size() && scout == nullptr; ++i) {         \
    scout = graph_->GET_STRUCT_METHOD(IDS[i]);                          \
  }                                                                     \
  if (scout != nullptr) {                                               \
    fv_num = scout->GET_FV_NUM();                                       \
  }                                                                     \
  F_NUMS_PTR->reserve(IDS.size() * FIDS.size());                        \
  F_VALUES_PTR->reserve(IDS.size() * FIDS.size() * fv_num);             \
  for (size_t i = 0; i < IDS.size(); ++i) {                             \
    STRUCT_TYPE* strct = graph_->GET_STRUCT_METHOD(IDS[i]);             \
    if (strct != nullptr) {                                             \
      strct->GET_FEATURE_METHOD(FIDS, F_NUMS_PTR, F_VALUES_PTR);        \
    } else {                                                            \
      for (size_t i = 0; i < FIDS.size(); ++i) {                        \
        F_NUMS_PTR->push_back(0);                                       \
      }                                                                 \
    }                                                                   \
  }                                                                     \
}

// Get certain float32 features of the intended nodes
void GraphEngine::GetNodeFloat32Feature(
    const std::vector<euler::common::NodeID>& node_ids,
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<float>* feature_values) const {
  GET_FEATURE(GetNodeByID, GetFloat32Feature, Node, feature_nums,
              feature_values, node_ids, fids, GetFloat32FeatureValueNum);
}

void GraphEngine::GetNodeUint64Feature(
    const std::vector<euler::common::NodeID>& node_ids,
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<uint64_t>* feature_values) const {
  GET_FEATURE(GetNodeByID, GetUint64Feature, Node, feature_nums,
              feature_values, node_ids, fids, GetUint64FeatureValueNum);
}

void GraphEngine::GetNodeBinaryFeature(
    const std::vector<euler::common::NodeID>& node_ids,
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<char>* feature_values) const {
  GET_FEATURE(GetNodeByID, GetBinaryFeature, Node, feature_nums,
              feature_values, node_ids, fids, GetBinaryFeatureValueNum);
}

void GraphEngine::GetEdgeFloat32Feature(
    const std::vector<euler::common::EdgeID>& edge_ids,
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<float>* feature_values) const {
  GET_FEATURE(GetEdgeByID, GetFloat32Feature, Edge, feature_nums,
              feature_values, edge_ids, fids, GetFloat32FeatureValueNum);
}

void GraphEngine::GetEdgeUint64Feature(
    const std::vector<euler::common::EdgeID>& edge_ids,
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<uint64_t>* feature_values) const {
  GET_FEATURE(GetEdgeByID, GetUint64Feature, Edge, feature_nums,
              feature_values, edge_ids, fids, GetUint64FeatureValueNum);
}

void GraphEngine::GetEdgeBinaryFeature(
    const std::vector<euler::common::EdgeID>& edge_ids,
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<char>* feature_values) const {
  GET_FEATURE(GetEdgeByID, GetBinaryFeature, Edge, feature_nums,
              feature_values, edge_ids, fids, GetBinaryFeatureValueNum);
}

#define GET_NODE_NEIGHBOR(METHOD, NEIGHBORS_PTR, NEIGHBOR_NUMS_PTR, \
                          NODE_IDS, EDGE_TYPES, ...) {              \
  NEIGHBOR_NUMS_PTR->resize(NODE_IDS.size(), 0);                    \
  NEIGHBORS_PTR->reserve(NODE_IDS.size());                          \
  for (size_t i = 0; i < NODE_IDS.size(); ++i) {                    \
    Node* node = graph_->GetNodeByID(NODE_IDS[i]);                  \
    if (node != nullptr) {                                          \
      std::vector<euler::common::IDWeightPair> node_neighbor =      \
          node->METHOD(EDGE_TYPES, ##__VA_ARGS__);                  \
      (*NEIGHBOR_NUMS_PTR)[i] = node_neighbor.size();               \
      NEIGHBORS_PTR->insert(NEIGHBORS_PTR->end(),                   \
                            node_neighbor.begin(),                  \
                            node_neighbor.end());                   \
    }                                                               \
  }                                                                 \
}

void GraphEngine::GetFullNeighbor(
    const std::vector<euler::common::NodeID>& node_ids,
    const std::vector<int32_t>& edge_types,
    std::vector<euler::common::IDWeightPair>* neighbors,
    std::vector<uint32_t>* neighbor_nums) const {
  GET_NODE_NEIGHBOR(GetFullNeighbor, neighbors, neighbor_nums,
                    node_ids, edge_types);
}

void GraphEngine::GetSortedFullNeighbor(
    const std::vector<euler::common::NodeID>& node_ids,
    const std::vector<int32_t>& edge_types,
    std::vector<euler::common::IDWeightPair>* neighbors,
    std::vector<uint32_t>* neighbor_nums) const {
  GET_NODE_NEIGHBOR(GetSortedFullNeighbor, neighbors, neighbor_nums,
                    node_ids, edge_types);
}

void GraphEngine::GetTopKNeighbor(
    const std::vector<euler::common::NodeID>& node_ids,
    const std::vector<int32_t>& edge_types, int32_t k,
    std::vector<euler::common::IDWeightPair>* neighbors,
    std::vector<uint32_t>* neighbor_nums) const {
  GET_NODE_NEIGHBOR(GetTopKNeighbor, neighbors, neighbor_nums,
                    node_ids, edge_types, k);
}

void GraphEngine::SampleNeighbor(
    const std::vector<euler::common::NodeID>& node_ids,
    const std::vector<int32_t>& edge_types, int32_t count,
    std::vector<euler::common::IDWeightPair>* neighbors,
    std::vector<uint32_t>* neighbor_nums) const {
  GET_NODE_NEIGHBOR(SampleNeighbor, neighbors, neighbor_nums,
                    node_ids, edge_types, count);
}

}  // namespace core
}  // namespace euler
