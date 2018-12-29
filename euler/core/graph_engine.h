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

#ifndef EULER_CORE_GRAPH_ENGINE_H_
#define EULER_CORE_GRAPH_ENGINE_H_

#include <stdio.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>

#include "euler/core/graph.h"
#include "euler/core/graph_builder.h"
#include "euler/common/data_types.h"

namespace euler {
namespace core {
enum GraphType {compact, fast};

class GraphEngine {
 public:
  explicit GraphEngine(GraphType graph_type) {
    graph_type_ = graph_type;
    partition_num_ = 0;
    graph_ = nullptr;
  }

  virtual ~GraphEngine() {
    if (graph_ != nullptr) {
      delete graph_;
    }
  }

  virtual bool Initialize(std::unordered_map<std::string, std::string> conf);

  virtual bool Initialize(const std::string& directory);

  // Randomly Sample nodes with the specified type
  virtual std::vector<euler::common::NodeID>
  SampleNode(int32_t node_type, int32_t count) const;

  // Randomly sample edges with the specified type
  virtual std::vector<euler::common::EdgeID>
  SampleEdge(int32_t edge_type, int32_t count) const;

  virtual std::vector<int32_t> GetNodeType(
      const std::vector<euler::common::NodeID>& node_ids) const;

  // Get certain float32 features of the intended nodes
  virtual void GetNodeFloat32Feature(
      const std::vector<euler::common::NodeID>& node_ids,
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<float>* feature_values) const;

  // Get certain uint64 features of the intended nodes
  virtual void GetNodeUint64Feature(
      const std::vector<euler::common::NodeID>& node_ids,
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<uint64_t>* feature_values) const;

  // Get certain binary features of the intended nodes
  virtual void GetNodeBinaryFeature(
      const std::vector<euler::common::NodeID>& node_ids,
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<char>* feature_values) const;

  // Get certain float features of the intended edges
  virtual void GetEdgeFloat32Feature(
      const std::vector<euler::common::EdgeID>& edge_ids,
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<float>* feature_values) const;

  // Get certain uint64 features of the intended edges
  virtual void GetEdgeUint64Feature(
      const std::vector<euler::common::EdgeID>& edge_ids,
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<uint64_t>* feature_values) const;

  // Get certain binary features of the intended edges
  virtual void GetEdgeBinaryFeature(
      const std::vector<euler::common::EdgeID>& edge_ids,
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<char>* feature_values) const;

  // Get all the neighbor nodes of the specified edge types
  virtual void GetFullNeighbor(
      const std::vector<euler::common::NodeID>& node_ids,
      const std::vector<int32_t>& edge_types,
      std::vector<euler::common::IDWeightPair>* neighbors,
      std::vector<uint32_t>* neighbor_nums) const;

  // Get all the neighbor nodes of the specified edge types, sorted by node ids
  virtual void GetSortedFullNeighbor(
      const std::vector<euler::common::NodeID>& node_ids,
      const std::vector<int32_t>& edge_types,
      std::vector<euler::common::IDWeightPair>* neighbors,
      std::vector<uint32_t>* neighbor_nums) const;

  // Get top K neighbor nodes of the specified edge types
  virtual void GetTopKNeighbor(
      const std::vector<euler::common::NodeID>& node_ids,
      const std::vector<int32_t>& edge_types, int32_t k,
      std::vector<euler::common::IDWeightPair>* neighbors,
      std::vector<uint32_t>* neighbor_nums) const;

  // Randomly sample neighbors with the specified edge types
  virtual void SampleNeighbor(
      const std::vector<euler::common::NodeID>& node_ids,
      const std::vector<int32_t>& edge_types, int32_t count,
      std::vector<euler::common::IDWeightPair>* neighbors,
      std::vector<uint32_t>* neighbor_nums) const;

  virtual int32_t GetPartitionNum() {
    return partition_num_;
  }

  virtual std::string GetNodeSumWeight() {
    std::string result = "";
    std::vector<float> sums = graph_->GetNodeWeightSums();
    for (size_t i = 0; i < sums.size(); ++i) {
      char buffer[50];
      snprintf(buffer, sizeof(buffer), "%.6f", sums[i]);
      std::string weight_str(buffer);
      if (i > 0) {
        result += ",";
      }
      result += weight_str;
    }
    return result;
  }

  virtual std::string GetEdgeSumWeight() {
    std::string result = "";
    std::vector<float> sums = graph_->GetEdgeWeightSums();
    for (size_t i = 0; i < sums.size(); ++i) {
      char buffer[50];
      snprintf(buffer, sizeof(buffer), "%.6f", sums[i]);
      std::string weight_str(buffer);
      if (i > 0) {
        result += ",";
      }
      result += weight_str;
    }
    return result;
  }

  void ShowGraph() {
    graph_->ShowGraph();
  }

 private:
  GraphType graph_type_;

  Graph* graph_;

  int32_t partition_num_;
};

}  // namespace core
}  // namespace euler

#endif  // EULER_CORE_GRAPH_ENGINE_H_
