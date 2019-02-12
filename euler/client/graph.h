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

#ifndef EULER_CLIENT_GRAPH_H_
#define EULER_CLIENT_GRAPH_H_

#include <vector>
#include <string>
#include <utility>
#include <memory>
#include <functional>

#include "euler/common/data_types.h"
#include "euler/client/graph_config.h"

namespace euler {
namespace client {

using NodeID = euler::common::NodeID;
using EdgeID = euler::common::EdgeID;
using IDWeightPair = euler::common::IDWeightPair;

using IDWeightPairVec = std::vector<std::vector<IDWeightPair>>;
using NodeIDVec = std::vector<NodeID>;
using EdgeIDVec = std::vector<EdgeID>;
using TypeVec = std::vector<int32_t>;
using FloatFeatureVec = std::vector<std::vector<std::vector<float>>>;
using UInt64FeatureVec = std::vector<std::vector<std::vector<uint64_t>>>;
using BinaryFatureVec =  std::vector<std::vector<std::string>>;

// Represents a user view of the Graph.
// Note: the solid Graph may exist on remote or locale server.
// Example:
//    Graph* g = Graph::NewGraph("remote.ini");
class Graph {
 public:
  virtual ~Graph() {
  }

  static std::unique_ptr<Graph> NewGraph(const std::string& config_file);
  static std::unique_ptr<Graph> NewGraph(const GraphConfig& config);

  // Initialize the user Graph view by a specified GraphConfig.
  // Note: an Graph instance is vaild only after the
  // initialization has been successed.
  virtual bool Initialize(const GraphConfig& config) = 0;

  virtual void BiasedSampleNeighbor(
      const std::vector<NodeID>& node_ids,
      const std::vector<NodeID>& parent_node_ids,
      const std::vector<int>& edge_types,
      const std::vector<int>& parent_edge_types,
      int count,
      float p,
      float q,
      std::function<void(const IDWeightPairVec&)> callback) const;

  virtual void SampleNode(
      int node_type,
      int count,
      std::function<void(const NodeIDVec&)> callback) const = 0;

  virtual void SampleEdge(
      int edge_type,
      int count,
      std::function<void(const EdgeIDVec&)> callback) const = 0;

  virtual void GetNodeType(
      const std::vector<NodeID>& node_ids,
      std::function<void(const TypeVec&)> callback) const = 0;

  virtual void GetNodeFloat32Feature(
      const std::vector<NodeID>& node_ids,
      const std::vector<int>& fids,
      std::function<void(const FloatFeatureVec&)> callback) const = 0;

  virtual void GetNodeUint64Feature(
      const std::vector<NodeID>& node_ids,
      const std::vector<int>& fids,
      std::function<void(const UInt64FeatureVec&)> callback) const = 0;

  virtual void GetNodeBinaryFeature(
      const std::vector<NodeID>& node_ids,
      const std::vector<int>& fids,
      std::function<void(const BinaryFatureVec&)> callback) const = 0;

  virtual void GetEdgeFloat32Feature(
      const std::vector<EdgeID>& edge_ids,
      const std::vector<int>& fids,
      std::function<void(const FloatFeatureVec&)> callback) const = 0;

  virtual void GetEdgeUint64Feature(
      const std::vector<EdgeID>& edge_ids,
      const std::vector<int32_t>& fids,
      std::function<void(const UInt64FeatureVec&)> callback) const = 0;

  virtual void GetEdgeBinaryFeature(
      const std::vector<EdgeID>& edge_ids,
      const std::vector<int>& fids,
      std::function<void(const BinaryFatureVec&)> callback) const = 0;

  virtual void GetFullNeighbor(
      const std::vector<NodeID>& node_ids,
      const std::vector<int>& edge_types,
      std::function<void(const IDWeightPairVec&)> callback) const = 0;

  virtual void GetSortedFullNeighbor(
      const std::vector<NodeID>& node_ids,
      const std::vector<int>& edge_types,
      std::function<void(const IDWeightPairVec&)> callback) const = 0;

  virtual void GetTopKNeighbor(
      const std::vector<NodeID>& node_ids,
      const std::vector<int>& edge_types,
      int k,
      std::function<void(const IDWeightPairVec&)> callback) const = 0;

  virtual void SampleNeighbor(
      const std::vector<NodeID>& node_ids,
      const std::vector<int>& edge_types,
      int count,
      std::function<void(const IDWeightPairVec&)> callback) const = 0;
};

}   // namespace client
}   // namespace euler

#endif  // EULER_CLIENT_GRAPH_H_
