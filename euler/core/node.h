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

#ifndef EULER_CORE_NODE_H_
#define EULER_CORE_NODE_H_

#include <vector>
#include <string>
#include <utility>

#include "euler/common/data_types.h"
#include "euler/common/timmer.h"
#include "euler/common/weighted_collection.h"
#include "euler/common/compact_weighted_collection.h"

namespace euler {
namespace core {

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

class Node {
 public:
  Node(euler::common::NodeID id, float weight) : id_(id), weight_(weight) {}

  Node() {}

  virtual ~Node() {
  }

  euler::common::NodeID GetID() const {return id_;}

  int32_t GetType() const {return type_;}

  float GetWeight() const {return weight_;}

  // Randomly sample neighbors with the specified edge types
  virtual std::vector<euler::common::IDWeightPair>
  SampleNeighbor(const std::vector<int32_t>& edge_types, int32_t count) const = 0;

  // Get all the neighbor nodes of the specified edge types
  virtual std::vector<euler::common::IDWeightPair>
  GetFullNeighbor(const std::vector<int32_t>& edge_types) const = 0;

  // Get all the neighbor nodes of the specified edge types, sorted by node ids
  virtual std::vector<euler::common::IDWeightPair>
  GetSortedFullNeighbor(const std::vector<int32_t>& edge_types) const = 0;

  // Get top K neighbor nodes of the specified edge types
  virtual std::vector<euler::common::IDWeightPair>
  GetTopKNeighbor(const std::vector<int32_t>& edge_types, int32_t k) const = 0;

  virtual int32_t GetFloat32FeatureValueNum() const = 0;

  virtual int32_t GetUint64FeatureValueNum() const = 0;

  virtual int32_t GetBinaryFeatureValueNum() const = 0;

  virtual void GetUint64Feature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<uint64_t>* feature_values) const = 0;

  virtual void GetFloat32Feature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<float>* feature_values) const = 0;

  virtual void GetBinaryFeature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<char>* feature_values) const = 0;

  virtual bool DeSerialize(const char* s, size_t size) = 0;

  bool DeSerialize(const std::string& data) {
    return DeSerialize(data.c_str(), data.size());
  }

  virtual std::string Serialize() const = 0;

 protected:
  euler::common::NodeID id_;

  int32_t type_;

  float weight_;

  euler::common::CompactWeightedCollection<int32_t> edge_group_collection_;

};

}  // namespace core
}  // namespace euler

#endif  // EULER_CORE_NODE_H_
