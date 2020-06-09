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

#ifndef EULER_CORE_GRAPH_EDGE_H_
#define EULER_CORE_GRAPH_EDGE_H_

#include <vector>
#include <string>
#include <algorithm>

#include "euler/common/data_types.h"

namespace euler {

class Edge {
 public:
  Edge(euler::common::NodeID src_id,
       euler::common::NodeID dst_id,
       int32_t type, float weight)
      : id_(std::make_tuple(src_id, dst_id, type)),
        type_(type), weight_(weight) {}

  Edge() {}

  virtual ~Edge() {}

  bool Init(const std::vector<std::vector<uint64_t>>& uint64_features,
            const std::vector<std::vector<float>>& float_features,
            const std::vector<std::string>& binary_features);

  euler::common::EdgeID GetID() const {return id_;}

  int32_t GetType() const {return type_;}

  float GetWeight() const {return weight_;}

  virtual int32_t GetFloat32FeatureValueNum() const {
    int32_t num = 1, pre = 0;
    for (size_t i = 0; i < float_features_idx_.size(); ++i) {
      num = std::max(float_features_idx_[i] - pre, num);
      pre = float_features_idx_[i];
    }
    return num;
  }

  virtual int32_t GetUint64FeatureValueNum() const {
    int32_t num = 1, pre = 0;
    for (size_t i = 0; i < uint64_features_idx_.size(); ++i) {
      num = std::max(uint64_features_idx_[i] - pre, num);
      pre = uint64_features_idx_[i];
    }
    return num;
  }

  virtual int32_t GetBinaryFeatureValueNum() const {
    int32_t num = 1, pre = 0;
    for (size_t i = 0; i < binary_features_idx_.size(); ++i) {
      num = std::max(binary_features_idx_[i] - pre, num);
      pre = binary_features_idx_[i];
    }
    return num;
  }

  // Get certain uint64 features
  virtual void GetUint64Feature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<uint64_t>* feature_values) const;

  virtual void GetUint64Feature(
      const std::vector<int32_t>& fids,
      std::vector<std::vector<uint64_t>>* feature_values) const;

  // Get certain float32 features
  virtual void GetFloat32Feature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<float>* feature_values) const;

  virtual void GetFloat32Feature(
      const std::vector<int32_t>& fids,
      std::vector<std::vector<float>>* feature_values) const;

  // Get certain binary features
  virtual void GetBinaryFeature(
      const std::vector<int>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<char>* feature_values) const;

  virtual void GetBinaryFeature(
      const std::vector<int>& fids,
      std::vector<std::string>* feature_values) const;

  virtual bool DeSerialize(const char* s, size_t size);

  bool DeSerialize(const std::string& data) {
    return DeSerialize(data.c_str(), data.size());
  }

  virtual bool Serialize(std::string* s) const;
  virtual uint32_t SerializeSize() const;

 protected:
  euler::common::EdgeID id_;

  int32_t type_;

  float weight_;

  std::vector<int32_t> uint64_features_idx_;

  std::vector<uint64_t> uint64_features_;

  std::vector<int32_t> float_features_idx_;

  std::vector<float> float_features_;

  std::vector<int32_t> binary_features_idx_;

  std::string binary_features_;
};

}  // namespace euler

#endif  // EULER_CORE_GRAPH_EDGE_H_
