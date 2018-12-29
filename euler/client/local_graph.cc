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

#include "euler/client/local_graph.h"

#include <atomic>

#include "glog/logging.h"

namespace euler {
namespace client {

LocalGraph::LocalGraph() {
}

bool LocalGraph::Initialize(const GraphConfig& config) {
  std::string directory;
  if (!config.Get("directory", &directory)) {
    LOG(ERROR) << "Directory must be specified in local mode";
    return false;
  }

  euler::core::GraphType graph_type = euler::core::compact;
  std::string load_type;
  if (config.Get("load_type", &load_type) && load_type == "fast") {
    graph_type = euler::core::fast;
  }
  engine_ = new euler::core::GraphEngine(graph_type);
  if (!engine_->Initialize(directory)) {
    LOG(ERROR) << "Initialize local graph engine failed, config: "
               << config.DebugString();
    engine_ = nullptr;
    return false;
  }

  return true;
}


//////////////////////////// Sample Node/Edge ////////////////////////////

void LocalGraph::SampleNode(
    int node_type,
    int count,
    std::function<void(const NodeIDVec&)> callback) const {
  auto node_ids = engine_->SampleNode(node_type, count);
  callback(node_ids);
}

void LocalGraph::SampleEdge(
    int edge_type,
    int count,
    std::function<void(const EdgeIDVec&)> callback) const {
  auto edge_ids = engine_->SampleEdge(edge_type, count);
  callback(edge_ids);
}

////////////////////////////// Get Type /////////////////////////////

void LocalGraph::GetNodeType(
    const std::vector<NodeID>& node_ids,
    std::function<void(const TypeVec&)> callback) const {
  auto types = engine_->GetNodeType(node_ids);
  callback(types);
}

//////////////////////////// Get Feature ////////////////////////////

namespace {

template <typename ValueType, typename FeatureVec>
void BuildFeatureVec(const std::vector<ValueType>& values,
                     const std::vector<uint32_t>& value_nums,
                     int ids_size, int fids_size, FeatureVec* results) {
  auto data = values.data();
  results->resize(ids_size);
  for (int i = 0; i < ids_size; ++i) {
    auto& result = results->at(i);
    result.resize(fids_size);
    for (int j = 0; j < fids_size; ++j) {
      auto& item = result[j];
      item.resize(value_nums[i * fids_size + j]);
      memcpy(&item[0], data, sizeof(item[0]) * item.size());
      data += item.size();
    }
  }
}

}  // namespace

void LocalGraph::GetNodeFloat32Feature(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& fids,
    std::function<void(const FloatFeatureVec&)> callback) const {
  std::vector<float> values;
  std::vector<uint32_t> value_nums;
  FloatFeatureVec features;
  engine_->GetNodeFloat32Feature(node_ids, fids, &value_nums, &values);
  BuildFeatureVec(values, value_nums, node_ids.size(), fids.size(), &features);
  callback(features);
}

void LocalGraph::GetNodeUint64Feature(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& fids,
    std::function<void(const UInt64FeatureVec&)> callback) const {
  std::vector<uint64_t> values;
  std::vector<uint32_t> value_nums;
  UInt64FeatureVec features;
  engine_->GetNodeUint64Feature(node_ids, fids, &value_nums, &values);
  BuildFeatureVec(values, value_nums, node_ids.size(), fids.size(), &features);
  callback(features);
}

void LocalGraph::GetNodeBinaryFeature(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& fids,
    std::function<void(const BinaryFatureVec&)> callback) const {
  std::vector<char> values;
  std::vector<uint32_t> value_nums;
  BinaryFatureVec features;
  engine_->GetNodeBinaryFeature(node_ids, fids, &value_nums, &values);
  BuildFeatureVec(values, value_nums, node_ids.size(), fids.size(), &features);
  callback(features);
}

void LocalGraph::GetEdgeFloat32Feature(
    const std::vector<EdgeID>& edge_ids,
    const std::vector<int>& fids,
    std::function<void(const FloatFeatureVec&)> callback) const {
  std::vector<float> values;
  std::vector<uint32_t> value_nums;
  FloatFeatureVec features;
  engine_->GetEdgeFloat32Feature(edge_ids, fids, &value_nums, &values);
  BuildFeatureVec(values, value_nums, edge_ids.size(), fids.size(), &features);
  callback(features);
}

void LocalGraph::GetEdgeUint64Feature(
    const std::vector<EdgeID>& edge_ids,
    const std::vector<int32_t>& fids,
    std::function<void(const UInt64FeatureVec&)> callback) const {
  std::vector<uint64_t> values;
  std::vector<uint32_t> value_nums;
  UInt64FeatureVec features;
  engine_->GetEdgeUint64Feature(edge_ids, fids, &value_nums, &values);
  BuildFeatureVec(values, value_nums, edge_ids.size(), fids.size(), &features);
  callback(features);
}

void LocalGraph::GetEdgeBinaryFeature(
    const std::vector<EdgeID>& edge_ids,
    const std::vector<int>& fids,
    std::function<void(const BinaryFatureVec&)> callback) const {
  std::vector<char> values;
  std::vector<uint32_t> value_nums;
  BinaryFatureVec features;
  engine_->GetEdgeBinaryFeature(edge_ids, fids, &value_nums, &values);
  BuildFeatureVec(values, value_nums, edge_ids.size(), fids.size(), &features);
  callback(features);
}


//////////////////////////// Get Neighbor ////////////////////////////

namespace {

IDWeightPairVec NeighborsBuilder(const std::vector<IDWeightPair>& neighbors,
                                 const std::vector<uint32_t>& neighbor_nums,
                                 int node_number) {
  IDWeightPairVec neighbor_vec(node_number);
  auto data = neighbors.data();
  for (int i = 0; i < node_number; ++i) {
    neighbor_vec[i].resize(neighbor_nums[i]);
    memcpy(neighbor_vec[i].data(), data, sizeof(data[0]) * neighbor_nums[i]);
    data += neighbor_nums[i];
  }
  return neighbor_vec;
}

}  // namespace

void LocalGraph::GetFullNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    std::function<void(const IDWeightPairVec&)> callback) const {
  std::vector<IDWeightPair> neighbors;
  std::vector<uint32_t> nums;
  engine_->GetFullNeighbor(node_ids, edge_types, &neighbors, &nums);
  callback(NeighborsBuilder(neighbors, nums, node_ids.size()));
}

void LocalGraph::GetSortedFullNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    std::function<void(const IDWeightPairVec&)> callback) const {
  std::vector<IDWeightPair> neighbors;
  std::vector<uint32_t> nums;
  engine_->GetSortedFullNeighbor(node_ids, edge_types, &neighbors, &nums);
  callback(NeighborsBuilder(neighbors, nums, node_ids.size()));
}

void LocalGraph::GetTopKNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    int k,
    std::function<void(const IDWeightPairVec&)> callback) const {
  std::vector<IDWeightPair> neighbors;
  std::vector<uint32_t> nums;
  engine_->GetTopKNeighbor(node_ids, edge_types, k, &neighbors, &nums);
  callback(NeighborsBuilder(neighbors, nums, node_ids.size()));
}

void LocalGraph::SampleNeighbor(
    const std::vector<NodeID>& node_ids,
    const std::vector<int>& edge_types,
    int count,
    std::function<void(const IDWeightPairVec&)> callback) const {
  std::vector<IDWeightPair> neighbors;
  std::vector<uint32_t> nums;
  engine_->SampleNeighbor(node_ids, edge_types, count, &neighbors, &nums);
  callback(NeighborsBuilder(neighbors, nums, node_ids.size()));
}

}  // namespace client
}  // namespace euler
