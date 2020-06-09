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

#ifndef EULER_CORE_GRAPH_NODE_H_
#define EULER_CORE_GRAPH_NODE_H_

#include <vector>
#include <string>
#include <utility>

#include "euler/common/data_types.h"
#include "euler/common/timmer.h"
#include "euler/common/weighted_collection.h"
#include "euler/common/compact_weighted_collection.h"

namespace euler {

bool id_cmp(std::pair<euler::common::NodeID, float> a,
            std::pair<euler::common::NodeID, float> b);

class NodeComparison {
 public:
  bool operator()(const std::pair<euler::common::NodeID, int32_t>& a,
                  const std::pair<euler::common::NodeID, int32_t>& b) {
    return a.first > b.first;
  }
};

class NodeWeightComparision {
 public:
  bool operator()(const euler::common::IDWeightPair& a,
                  const euler::common::IDWeightPair& b) {
    return std::get<1>(a) > std::get<1>(b);
  }
};

class NeighborInfo{
 public:
  NeighborInfo() {
  }
  euler::common::CompactWeightedCollection<int32_t> edge_group_collection;
  std::vector<int32_t> neighbor_groups_idx;
  std::vector<euler::common::NodeID> neighbors;
  std::vector<float> neighbors_weight;
};

class Node {
 public:
  Node(euler::common::NodeID id, float weight, int32_t type)
      : id_(id), weight_(weight), type_(type) {}

  Node() {}

  virtual ~Node() {
  }

  virtual bool Init(const std::vector<std::vector<uint64_t>>& neighbor_ids,
            const std::vector<std::vector<float>>& neighbor_weights,
            const std::vector<std::vector<uint64_t>>& uint64_features,
            const std::vector<std::vector<float>>& float_features,
            const std::vector<std::string>& binary_features);

  euler::common::NodeID GetID() const {return id_;}

  int32_t GetType() const {return type_;}

  float GetWeight() const {return weight_;}

  // Randomly sample neighbors with the specified edge types
  virtual std::vector<euler::common::IDWeightPair>
  SampleNeighbor(const std::vector<int32_t>& edge_types, int32_t count) const;

  // Randomly sample in-neighbors with the specified edge types
  virtual std::vector<euler::common::IDWeightPair>
  SampleInNeighbor(const std::vector<int32_t>& edge_types, int32_t count) const;

  // Get all the neighbor nodes of the specified edge types
  virtual std::vector<euler::common::IDWeightPair>
  GetFullNeighbor(const std::vector<int32_t>& edge_types) const;

  // Get all the in-neighbor nodes of the specified edge types
  virtual std::vector<euler::common::IDWeightPair>
  GetFullInNeighbor(const std::vector<int32_t>& edge_types) const;

  // Get all the neighbor nodes of the specified edge types, sorted by node ids
  virtual std::vector<euler::common::IDWeightPair>
  GetSortedFullNeighbor(const std::vector<int32_t>& edge_types) const;

  // Get all the in-neighbor nodes of the specified edge types,
  // sorted by node ids
  virtual std::vector<euler::common::IDWeightPair>
  GetSortedFullInNeighbor(const std::vector<int32_t>& edge_types) const;

  // Get top K neighbor nodes of the specified edge types
  virtual std::vector<euler::common::IDWeightPair>
  GetTopKNeighbor(const std::vector<int32_t>& edge_types, int32_t k) const;

  // Get top K in-neighbor nodes of the specified edge types
  virtual std::vector<euler::common::IDWeightPair>
  GetTopKInNeighbor(const std::vector<int32_t>& edge_types, int32_t k) const;

  virtual int32_t GetFloat32FeatureValueNum() const;

  virtual int32_t GetUint64FeatureValueNum() const;

  virtual int32_t GetBinaryFeatureValueNum() const;

  virtual void GetUint64Feature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<uint64_t>* feature_values) const;

  virtual void GetUint64Feature(
      const std::vector<int32_t>& fids,
      std::vector<std::vector<uint64_t>>* feature_values) const;

  virtual void GetFloat32Feature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<float>* feature_values) const;

  virtual void GetFloat32Feature(
      const std::vector<int32_t>& fids,
      std::vector<std::vector<float>>* feature_values) const;

  virtual void GetBinaryFeature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<char>* feature_values) const;

  virtual void GetBinaryFeature(
      const std::vector<int32_t>& fids,
      std::vector<std::string>* feature_values) const;

  virtual bool DeSerialize(const char* s, size_t size);

  bool DeSerialize(const std::string& data) {
    return DeSerialize(data.c_str(), data.size());
  }

  virtual bool Serialize(std::string* s) const;

  virtual uint32_t SerializeSize() const;

 protected:
  euler::common::NodeID id_;

  float weight_;

  int32_t type_;

 private:
  std::vector<int32_t> uint64_features_idx_;

  std::vector<uint64_t> uint64_features_;

  std::vector<int32_t> float_features_idx_;

  std::vector<float> float_features_;

  std::vector<int32_t> binary_features_idx_;

  std::string binary_features_;

  NeighborInfo neighbor_info_;

  NeighborInfo in_neighbor_info_;

  inline std::vector<euler::common::IDWeightPair> __SampleNeighbor(
    const std::vector<int32_t>& edge_types,
    int32_t count,
    const NeighborInfo& ni) const;

  inline std::vector<euler::common::IDWeightPair> __GetFullNeighbor(
    const std::vector<int32_t>& edge_types,
    const NeighborInfo& ni) const;

  inline std::vector<euler::common::IDWeightPair> __GetSortedFullNeighbor(
    const std::vector<int32_t>& edge_types,
    const NeighborInfo& ni) const;

  inline std::vector<euler::common::IDWeightPair> __GetTopKNeighbor(
    const std::vector<int32_t>& edge_types,
    int32_t k,
    const NeighborInfo& ni) const;
};

}  // namespace euler

#endif  // EULER_CORE_GRAPH_NODE_H_
