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

#include "euler/core/compact_graph.h"

#include "glog/logging.h"
#include "euler/common/data_types.h"

namespace euler {
namespace core {

CompactGraph::CompactGraph() {
  global_sampler_ok_ = false;
}

CompactGraph::~CompactGraph() {
}

std::vector<euler::common::NodeID>
CompactGraph::SampleNode(int32_t node_type, int32_t count) const {
  if (!global_sampler_ok_) {
    LOG(ERROR) << "global sampler is not initialized!";
  }
  std::vector<euler::common::NodeID> vec;
  vec.reserve(count);
  if (node_type == -1) {
    if (node_type_collection_.GetSumWeight() == 0) {
      return vec;
    }
    for (int32_t i = 0; i < count; i++) {
      node_type = node_type_collection_.Sample().first;
      vec.push_back(node_samplers_[node_type].Sample().first);
    }
  }
  else {
    if (node_samplers_[node_type].GetSumWeight() == 0) {
      return vec;
    }
    for (int32_t i = 0; i < count; i++) {
      vec.push_back(node_samplers_[node_type].Sample().first);
    }
  }
  return vec;
}

std::vector<euler::common::EdgeID>
CompactGraph::SampleEdge(int32_t edge_type, int32_t count) const {
  if (!global_edge_sampler_ok_) {
    LOG(ERROR) << "global edge sampler is not initialized!";
  }
  std::vector<euler::common::EdgeID> vec;
  vec.reserve(count);
  if (edge_samplers_[edge_type].GetSumWeight() == 0) {
    return vec;
  }
  for (int32_t i = 0; i < count; i++) {
    vec.push_back(edge_samplers_[edge_type].Sample().first);
  }
  return vec;
}

bool CompactGraph::BuildGlobalSampler() {
  std::vector<std::vector<float>> norm_weights;
  std::vector<std::vector<euler::common::NodeID>> node_ids;
  std::vector<int32_t> node_type_ids;
  norm_weights.resize(node_type_num_);
  node_ids.resize(node_type_num_);
  node_type_ids.resize(node_type_num_);
  node_weight_sums_.resize(node_type_num_);
  node_type_ids.resize(node_type_num_);
  for (auto &it : node_map_) {
    int32_t type = it.second->GetType();
    node_ids[type].push_back(it.first);
    norm_weights[type].push_back(it.second->GetWeight());
    node_weight_sums_[type] += it.second->GetWeight();
  }

  for (size_t type = 0; type < node_weight_sums_.size(); type++) {
    for (size_t idx = 0; idx < norm_weights[type].size(); idx++) {
      norm_weights[type][idx] /= node_weight_sums_[type];
    }
    node_type_ids[type] = type;
  }

  node_samplers_.resize(node_ids.size());
  for (size_t type = 0; type < node_ids.size(); type++) {
    node_samplers_[type].Init(node_ids[type], norm_weights[type]);
  }
  node_type_collection_.Init(node_type_ids, node_weight_sums_);
  global_sampler_ok_ = true;
  return true;
}

bool CompactGraph::BuildGlobalEdgeSampler() {
  std::vector<std::vector<float>> norm_weights;
  std::vector<std::vector<euler::common::EdgeID>> edge_ids;
  norm_weights.resize(edge_type_num_);
  edge_ids.resize(edge_type_num_);
  edge_weight_sums_.resize(edge_type_num_);
  for (auto &it : edge_map_) {
    int type = it.second->GetType();
    edge_ids[type].push_back(it.first);
    norm_weights[type].push_back(it.second->GetWeight());
    edge_weight_sums_[type] += it.second->GetWeight();
  }

  for (size_t type = 0; type < edge_weight_sums_.size(); type++) {
    for (size_t idx = 0; idx < norm_weights[type].size(); idx++) {
      norm_weights[type][idx] /= edge_weight_sums_[type];
    }
  }

  edge_samplers_.resize(edge_ids.size());
  for (size_t type = 0; type < edge_ids.size(); type++) {
    edge_samplers_[type].Init(edge_ids[type], norm_weights[type]);
  }
  global_edge_sampler_ok_ = true;
  return true;
}

}  // namespace core
}  // namespace euler
