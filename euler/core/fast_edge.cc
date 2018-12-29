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

#include "euler/core/fast_edge.h"

#include <string>
#include <vector>

#include "euler/common/bytes_reader.h"
namespace euler {
namespace core {
FastEdge::FastEdge(euler::common::EdgeID id, float weight)
  : Edge(id, weight) {
}

FastEdge::FastEdge() {
}

FastEdge::~FastEdge() {
}

bool FastEdge::DeSerialize(const char* s, size_t size) {
  uint64_t src_id = 0;
  uint64_t dst_id = 0;
  euler::common::BytesReader bytes_reader(s, size);
  // parse id
  if (!bytes_reader.GetUInt64(&src_id) ||
      !bytes_reader.GetUInt64(&dst_id)) {
    return false;
  }

  if (!bytes_reader.GetInt32(&type_) ||  // parse type
      !bytes_reader.GetFloat(&weight_)) {  // parse weight
    return false;
  }

  id_ = std::make_tuple(src_id, dst_id, type_);

  // parse uint64 feature
  int32_t uint64_feature_type_num = 0;
  if (!bytes_reader.GetInt32(&uint64_feature_type_num)) {
    return false;
  }
  if (!bytes_reader.GetInt32List(uint64_feature_type_num,
                                 &uint64_features_idx_)) {
    return false;
  }
  int32_t uint64_fv_num = 0;
  for (int32_t i = 0; i < uint64_feature_type_num; ++i) {
    uint64_fv_num += uint64_features_idx_[i];
    uint64_features_idx_[i] = uint64_fv_num;
  }
  if (!bytes_reader.GetUInt64List(uint64_fv_num, &uint64_features_)) {
    return false;
  }

  // parse float feature
  int32_t float_feature_type_num = 0;
  if (!bytes_reader.GetInt32(&float_feature_type_num)) {
    return false;
  }
  if (!bytes_reader.GetInt32List(float_feature_type_num,
                                 &float_features_idx_)) {
    return false;
  }
  int32_t float_fv_num = 0;
  for (int32_t i = 0; i < float_feature_type_num; ++i) {
    float_fv_num += float_features_idx_[i];
    float_features_idx_[i] = float_fv_num;
  }
  if (!bytes_reader.GetFloatList(float_fv_num, &float_features_)) {
    return false;
  }

  // parse binary feature
  int32_t binary_feature_type_num = 0;
  if (!bytes_reader.GetInt32(&binary_feature_type_num)) {
    return false;
  }
  if (!bytes_reader.GetInt32List(binary_feature_type_num,
                                 &binary_features_idx_)) {
    return false;
  }
  int32_t binary_fv_num = 0;
  for (int32_t i = 0; i < binary_feature_type_num; ++i) {
    binary_fv_num += binary_features_idx_[i];
    binary_features_idx_[i] = binary_fv_num;
  }
  if (!bytes_reader.GetString(binary_fv_num, &binary_features_)) {
    return false;
  }

  return true;
}

std::string FastEdge::Serialize() const {
  std::string result;
  return result;
}

}  // namespace core
}  // namespace euler
