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

#ifndef EULER_CORE_FAST_NODE_H_
#define EULER_CORE_FAST_NODE_H_

#include <string>

#include "euler/core/node.h"
#include "euler/common/data_types.h"
#include "euler/common/fast_weighted_collection.h"

namespace euler {
namespace core {

class FastNode : public Node {
 public:
  FastNode(euler::common::NodeID id, float weight);

  FastNode();

  ~FastNode();

  // Randomly sample neighbors with the specified edge types
  std::vector<euler::common::IDWeightPair>
  SampleNeighbor(const std::vector<int32_t>& edge_types, int32_t count) const override;

  // Get all the neighbor nodes of the specified edge types
  std::vector<euler::common::IDWeightPair>
  GetFullNeighbor(const std::vector<int32_t>& edge_types) const override;

  // Get all the neighbor nodes of the specified edge types, sorted by node ids
  std::vector<euler::common::IDWeightPair>
  GetSortedFullNeighbor(const std::vector<int32_t>& edge_types) const override;

  // Get top K neighbor nodes of the specified edge types
  std::vector<euler::common::IDWeightPair>
  GetTopKNeighbor(const std::vector<int32_t>& edge_types, int32_t k) const override;

  int32_t GetFloat32FeatureValueNum() const override {
    int32_t num = 1;
    for (size_t i = 0; i < float_features_.size(); ++i) {
      num = std::max(static_cast<int32_t>(float_features_[i].size()), num);
    }
    return num;
  }

  int32_t GetUint64FeatureValueNum() const override {
    int32_t num = 1;
    for (size_t i = 0; i < uint64_features_.size(); ++i) {
      num = std::max(static_cast<int32_t>(uint64_features_[i].size()), num);
    }
    return num;
  }

  int32_t GetBinaryFeatureValueNum() const override {
    int32_t num = 1;
    for (size_t i = 0; i < binary_features_.size(); ++i) {
      num = std::max(static_cast<int32_t>(binary_features_[i].size()), num);
    }
    return num;
  }

  void GetUint64Feature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<uint64_t>* feature_values) const override;

  void GetFloat32Feature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<float>* feature_values) const override;

  void GetBinaryFeature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<char>* feature_values) const override;

  bool DeSerialize(const char* s, size_t size) override;

  std::string Serialize() const override;

 private:
  std::vector<euler::common::FastWeightedCollection<euler::common::NodeID>*>
      neighbor_collection_list_;

  std::vector<std::vector<uint64_t>> uint64_features_;

  std::vector<std::vector<float>> float_features_;

  std::vector<std::string> binary_features_;
};

}  // namespace core
}  // namespace euler

#endif  // EULER_CORE_FAST_NODE_H_
