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

#ifndef EULER_CORE_EDGE_H_
#define EULER_CORE_EDGE_H_

#include <vector>
#include <string>

#include "euler/common/data_types.h"

namespace euler {
namespace core {

class Edge {
 public:
  Edge(euler::common::EdgeID id, float weight)
      : id_(id), weight_(weight) {}

  Edge() {}

  virtual ~Edge() {}

  euler::common::EdgeID GetID() const {return id_;}

  int32_t GetType() const {return type_;}

  float GetWeight() const {return weight_;}

  // Get certain uint64 features
  virtual void GetUint64Feature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<uint64_t>* feature_values) const;

  // Get certain float32 features
  virtual void GetFloat32Feature(
      const std::vector<int32_t>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<float>* feature_values) const;

  // Get certain binary features
  virtual void GetBinaryFeature(
      const std::vector<int>& fids,
      std::vector<uint32_t>* feature_nums,
      std::vector<char>* feature_values) const;

  virtual bool DeSerialize(const char* s, size_t size) = 0;

  bool DeSerialize(const std::string& data) {
    return DeSerialize(data.c_str(), data.size());
  }

  virtual std::string Serialize() const = 0;

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

}  // namespace core
}  // namespace euler

#endif  // EULER_CORE_EDGE_H_
