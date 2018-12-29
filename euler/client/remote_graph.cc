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

#include "euler/client/remote_graph.h"

#include <atomic>
#include <memory>

#include "glog/logging.h"

#include "euler/client/remote_graph_shard.h"
#include "euler/common/server_monitor.h"
#include "euler/common/compact_weighted_collection.h"
#include "euler/common/string_util.h"

namespace euler {
namespace client {

///////////////////////// MergeCallback /////////////////////////

template <class T>
class MergeCallback {
 public:
  MergeCallback(
      T* final_result, std::function<void(const T&)> callback,
      std::atomic<int>* counter, int target, const std::vector<int>& index)
      : final_result_(final_result), callback_(callback),
        counter_(counter), target_(target), index_(index) {
  }

  void operator()(const T& partial_result);

 private:
  T* final_result_;  // Free by callback
  std::function<void(const T&)> callback_;
  std::atomic<int>* counter_;  // Free by callback
  int target_;
  std::vector<int> index_;
};

template<class T>
void MergeCallback<T>::operator()(const T& partial_result) {
  // Copy partial reuslt to final result
  for (size_t i = 0; i < index_.size() && i < partial_result.size(); ++i) {
    final_result_->at(index_[i]) = partial_result[i];
  }

  auto counter = ++*counter_;
  if (counter == target_) {
    callback_(*final_result_);
    delete final_result_;
    delete counter_;
  }
}

///////////////////////// RemoteGraph /////////////////////////

RemoteGraph::RemoteGraph(): server_monitor_(nullptr) {
}

RemoteGraph::~RemoteGraph() {
  shards_.clear();
}

bool RemoteGraph::Initialize(const GraphConfig& config) {
  // Initialize server monitor
  if (server_monitor_ == nullptr) {
    std::string zk_server;
    std::string zk_path;
    if (!config.Get("zk_server", &zk_server)) {
      LOG(ERROR) << "No zookeeper server specified";
      return false;
    }
    if (!config.Get("zk_path", &zk_path)) {
      LOG(ERROR) << "No zookeeper path specified";
      return false;
    }

    LOG(INFO) << "Initialize RemoteGraph, connect to server monitor: ["
              << zk_server << ", " << zk_path << "]";
    server_monitor_ = euler::common::GetServerMonitor(zk_server, zk_path);
    if (server_monitor_ == nullptr) {
      LOG(ERROR) << "Create server monitor failed, config: "
                 << config.DebugString();
      return false;
    }
  }

  int shard_number = 0;
  if (!server_monitor_->GetNumShards(&shard_number) || shard_number == 0) {
    LOG(ERROR) << "Retrieve shard info from server failed!";
    return false;
  }
  LOG(INFO) << "Retrieve meta info success, shard number: " << shard_number;

  std::string meta_value;
  if (!server_monitor_->GetMeta("num_partitions", &meta_value)) {
    LOG(ERROR) << "Retrieve partition number from server failed!";
    return false;
  }

  partition_number_ = atoi(meta_value.c_str());
  if (partition_number_ == 0) {
    LOG(ERROR) << "Invalid partition number";
    return false;
  }
  LOG(INFO) << "Retrieve meta info success, partition number: "
            << partition_number_;

  for (auto i = 0; i < shard_number; ++i) {
    if (!RetrieveShardMeta(i, "node_sum_weight", &node_weight_sum_) ||
        !RetrieveShardMeta(i, "edge_sum_weight", &edge_weight_sum_)) {
      return false;
    }

    std::shared_ptr<RemoteGraphShard>
        shard(new RemoteGraphShard(server_monitor_, i));
    if (!shard->Initialize(config)) {
      shards_.clear();
      return false;
    }
    shards_.push_back(shard);
  }

  // Summary node weights on all node_types for each shard
  std::vector<float> weight_sum(shard_number, 0.0);
  for (auto i = 0; i < shard_number; ++i) {
    weight_sum[i] = 0.0;
    for (size_t j = 0; j < node_weight_sum_.size(); ++j) {
      weight_sum[i] += node_weight_sum_[j][i];
    }
  }
  node_weight_sum_.push_back(weight_sum);

  // Summary edge weights on all edge_types for each shard
  for (auto i = 0; i < shard_number; ++i) {
    weight_sum[i] = 0.0;
    for (size_t j = 0; j < edge_weight_sum_.size(); ++j) {
      weight_sum[i] += edge_weight_sum_[j][i];
    }
  }
  edge_weight_sum_.push_back(weight_sum);

  return true;
}

bool RemoteGraph::RetrieveShardMeta(int shard_index, const std::string& key,
                                    std::vector<std::vector<float>>* weights) {
  std::string weight;
  if (!server_monitor_->GetShardMeta(shard_index, key, &weight)) {
    LOG(ERROR) << "Retrieve shard meta failed, key: " << key
               << " shard index: " << shard_index;
    return false;
  }

  std::vector<std::string> weight_vec;
  euler::common::split_string(weight, ',', &weight_vec);
  if (weight_vec.empty()) {
    LOG(ERROR) << "Invalid weight meta failed, shard: " << shard_index
               << " weight meta: " << weight;
    return false;
  }

  // Resize weights vector to fit type if necessary
  if (weights->size() < weight_vec.size()) {
    weights->resize(weight_vec.size());
  }

  for (auto& wv : *weights) {
    wv.resize(shard_index + 1, 0.0);
  }

  for (size_t j = 0; j < weight_vec.size(); ++j) {
    auto& wv = weights->at(j);
    wv[shard_index] = atof(weight_vec[j].c_str());
  }

  LOG(INFO) << "Retrieve Shard Meta Info successfully, shard: " << shard_index
            << ", Key: " << key << ", Meta Info: " << weight;
  return true;
}

#define REMOTE_SAMPLE(RESULT_TYPE, METHOD, TYPE, WEIGHT)        \
  auto ids = new RESULT_TYPE(count);                            \
  auto counter = new std::atomic<int>(0);                       \
  std::vector<int> job_size(shards_.size(), 0);                 \
  std::vector<int> shard_index;                                 \
  for (size_t i = 0; i < shards_.size(); ++i) {                 \
    shard_index.push_back(i);                                   \
  }                                                             \
  using CWC = common::CompactWeightedCollection<int>;           \
  CWC sampler;                                                  \
  sampler.Init(shard_index, WEIGHT);                            \
  for (int i = 0; i < count; ++i) {                             \
    auto si = sampler.Sample().first;                           \
    ++job_size[si];                                             \
  }                                                             \
  int last = 0;                                                 \
  for (size_t i = 0; i < shards_.size(); ++i) {                 \
    std::vector<int> index;                                     \
    index.reserve(shards_.size());                              \
    for (int j = 0; j < job_size[i]; ++j) {                     \
      index.push_back(last);                                    \
      ++last;                                                   \
    }                                                           \
    MergeCallback<RESULT_TYPE> mcallback(                       \
        ids, callback, counter, shards_.size(), index);         \
    shards_[i]->METHOD(TYPE, index.size(), mcallback);          \
  }

void RemoteGraph::SampleNode(
    int node_type,
    int count,
    std::function<void(const NodeIDVec&)> callback) const {
  auto type = (node_type == -1) ? (node_weight_sum_.size() - 1) : node_type;
  REMOTE_SAMPLE(NodeIDVec, SampleNode, node_type, node_weight_sum_[type]);
}

void RemoteGraph::SampleEdge(
    int edge_type,
    int count,
    std::function<void(const EdgeIDVec&)> callback) const {
  auto type = (edge_type == -1) ? (edge_weight_sum_.size() - 1) : edge_type;
  REMOTE_SAMPLE(EdgeIDVec, SampleEdge, edge_type, edge_weight_sum_[type]);
}

#undef REMOTE_SAMPLE  // REMOTE_SAMPLE

#define PARTITION_NODE_CALL(ID_TYPE, IDS, PARTITION,                    \
                            RESULT_TYPE, METHOD, ...)                   \
  std::vector<std::vector<ID_TYPE>> shard_node_ids(shards_.size());     \
  std::vector<std::vector<int>> shard_index(shards_.size());            \
  for (size_t i = 0; i < shard_node_ids.size(); ++i) {                  \
    shard_node_ids[i].reserve(IDS.size());                              \
    shard_index[i].reserve(IDS.size());                                 \
  }                                                                     \
  for (size_t i = 0; i < IDS.size(); ++i) {                             \
    int index = PARTITION(IDS[i]);                                      \
    shard_node_ids[index].push_back(IDS[i]);                            \
    shard_index[index].push_back(i);                                    \
  }                                                                     \
  auto features = new RESULT_TYPE(IDS.size());                          \
  auto counter = new std::atomic<int>();                                \
  for (size_t i = 0; i < shards_.size(); ++i) {                         \
    MergeCallback<RESULT_TYPE> mcallback(                               \
        features, callback, counter, shards_.size(), shard_index[i]);   \
    shards_[i]->METHOD(                                                 \
        shard_node_ids[i], ##__VA_ARGS__, mcallback);                   \
  }

void RemoteGraph::GetNodeType(
    const std::vector<NodeID>& node_ids,
    std::function<void(const TypeVec&)> callback) const {
  PARTITION_NODE_CALL(NodeID, node_ids, NodePartition(),
                      TypeVec, GetNodeType);
}

void RemoteGraph::GetNodeFloat32Feature(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& fids,
    std::function<void(const FloatFeatureVec&)> callback) const {
  PARTITION_NODE_CALL(NodeID, node_ids, NodePartition(),
                      FloatFeatureVec, GetNodeFloat32Feature, fids);
}

void RemoteGraph::GetNodeUint64Feature(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& fids,
    std::function<void(const UInt64FeatureVec&)> callback) const {
  PARTITION_NODE_CALL(NodeID, node_ids, NodePartition(),
                      UInt64FeatureVec, GetNodeUint64Feature, fids);
}

void RemoteGraph::GetNodeBinaryFeature(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& fids,
    std::function<void(const BinaryFatureVec&)> callback) const {
  PARTITION_NODE_CALL(NodeID, node_ids, NodePartition(),
                      BinaryFatureVec, GetNodeBinaryFeature, fids);
}

void RemoteGraph::GetEdgeFloat32Feature(
    const std::vector<EdgeID>& edge_ids,
    const std::vector<int>& fids,
    std::function<void(const FloatFeatureVec&)> callback) const {
  PARTITION_NODE_CALL(EdgeID, edge_ids, EdgePartition(),
                      FloatFeatureVec, GetEdgeFloat32Feature, fids);
}

void RemoteGraph::GetEdgeUint64Feature(
    const std::vector<EdgeID>& edge_ids,
    const std::vector<int32_t>& fids,
    std::function<void(const UInt64FeatureVec&)> callback) const {
  PARTITION_NODE_CALL(EdgeID, edge_ids, EdgePartition(),
                      UInt64FeatureVec, GetEdgeUint64Feature, fids);
}

void RemoteGraph::GetEdgeBinaryFeature(
    const std::vector<EdgeID>& edge_ids,
    const std::vector<int>& fids,
    std::function<void(const BinaryFatureVec&)> callback) const {
  PARTITION_NODE_CALL(EdgeID, edge_ids, EdgePartition(),
                      BinaryFatureVec, GetEdgeBinaryFeature, fids);
}

void RemoteGraph::GetFullNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    std::function<void(const IDWeightPairVec&)> callback) const {
  PARTITION_NODE_CALL(NodeID, node_ids, NodePartition(),
                      IDWeightPairVec, GetFullNeighbor, edge_types);
}

void RemoteGraph::GetSortedFullNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    std::function<void(const IDWeightPairVec&)> callback) const {
  PARTITION_NODE_CALL(NodeID, node_ids, NodePartition(),
                      IDWeightPairVec, GetSortedFullNeighbor, edge_types);
}

void RemoteGraph::GetTopKNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    int k,
    std::function<void(const IDWeightPairVec&)> callback) const {
  PARTITION_NODE_CALL(NodeID, node_ids, NodePartition(),
                      IDWeightPairVec, GetTopKNeighbor, edge_types, k);
}

void RemoteGraph::SampleNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    int count,
    std::function<void(const IDWeightPairVec&)> callback) const {
  PARTITION_NODE_CALL(NodeID, node_ids, NodePartition(),
                      IDWeightPairVec, SampleNeighbor, edge_types, count);
}

#undef PARTITION_NODE_CALL

}  // namespace client
}  // namespace euler
