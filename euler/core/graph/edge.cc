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

#include "euler/core/graph/edge.h"


#include "euler/common/logging.h"
#include "euler/common/bytes_io.h"
#include "euler/common/bytes_compute.h"

namespace euler {

bool Edge::Init(const std::vector<std::vector<uint64_t>>& uint64_features,
                const std::vector<std::vector<float>>& float_features,
                const std::vector<std::string>& binary_features) {
  int32_t idx = 0;
  for (size_t i = 0; i < uint64_features.size(); ++i) {
    idx += uint64_features[i].size();
    uint64_features_idx_.push_back(idx);
    std::copy(uint64_features[i].begin(), uint64_features[i].end(),
              back_inserter(uint64_features_));
  }

  idx = 0;
  for (size_t i = 0; i < float_features.size(); ++i) {
    idx += float_features[i].size();
    float_features_idx_.push_back(idx);
    std::copy(float_features[i].begin(), float_features[i].end(),
              back_inserter(float_features_));
  }

  idx = 0;
  for (size_t i = 0; i < binary_features.size(); ++i) {
    idx += binary_features[i].size();
    binary_features_idx_.push_back(idx);
    std::copy(binary_features[i].begin(), binary_features[i].end(),
              back_inserter(binary_features_));
  }
  return true;
}

#define GET_EDGE_FEATURE(F_NUMS_PTR, F_VALUES_PTR, FEATURES,            \
                         FEATURES_IDX, FIDS) {                          \
    for (size_t i = 0; i < FIDS.size(); ++i) {                          \
      int32_t fid = FIDS[i];                                            \
      if (fid >= 0 && fid < static_cast<int32_t>(FEATURES_IDX.size())) { \
        int32_t pre = fid == 0 ? 0 : FEATURES_IDX[fid - 1];             \
        F_NUMS_PTR->push_back(FEATURES_IDX[fid] - pre);                 \
      } else {                                                          \
        F_NUMS_PTR->push_back(0);                                       \
      }                                                                 \
    }                                                                   \
    for (size_t i = 0; i < FIDS.size(); ++i) {                          \
      int32_t fid = FIDS[i];                                            \
      if (fid >= 0 && fid < static_cast<int32_t>(FEATURES_IDX.size())) { \
        int32_t pre = fid == 0 ? 0 : FEATURES_IDX[fid - 1];             \
        int32_t now = FEATURES_IDX[fid];                                \
        F_VALUES_PTR->insert(F_VALUES_PTR->end(),                       \
                             FEATURES.begin() + pre,                    \
                             FEATURES.begin() + now);                   \
      }                                                                 \
    }                                                                   \
  }

#define GET_EDGE_FEATURE_VEC(F_VALUES_PTR, FEATURES,                    \
                             FEATURES_IDX, FIDS, TYPE) {                \
    F_VALUES_PTR->resize(FIDS.size());                                  \
    for (size_t i = 0; i < FIDS.size(); ++i) {                          \
      int32_t fid = FIDS[i];                                            \
      if (fid >= 0 && fid < static_cast<int32_t>(FEATURES_IDX.size())) { \
        int32_t pre = fid == 0 ? 0 : FEATURES_IDX[fid - 1];             \
        int32_t now = FEATURES_IDX[fid];                                \
        (*F_VALUES_PTR)[i] = TYPE(FEATURES.begin() + pre,               \
                                  FEATURES.begin() + now);              \
      }                                                                 \
    }                                                                   \
  }

void Edge::GetUint64Feature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<uint64_t>* feature_values) const {
  GET_EDGE_FEATURE(feature_nums, feature_values, uint64_features_,
                   uint64_features_idx_, fids);
}

void Edge::GetUint64Feature(
    const std::vector<int32_t>& fids,
    std::vector<std::vector<uint64_t>>* feature_values) const {
  GET_EDGE_FEATURE_VEC(feature_values, uint64_features_,
                       uint64_features_idx_, fids, std::vector<uint64_t>);
}

void Edge::GetFloat32Feature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<float>* feature_values) const {
  GET_EDGE_FEATURE(feature_nums, feature_values, float_features_,
                   float_features_idx_, fids);
}

void Edge::GetFloat32Feature(
    const std::vector<int32_t>& fids,
    std::vector<std::vector<float>>* feature_values) const {
  GET_EDGE_FEATURE_VEC(feature_values, float_features_,
                       float_features_idx_, fids, std::vector<float>);
}

void Edge::GetBinaryFeature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<char>* feature_values) const {
  GET_EDGE_FEATURE(feature_nums, feature_values, binary_features_,
                   binary_features_idx_, fids);
}

void Edge::GetBinaryFeature(
    const std::vector<int32_t>& fids,
    std::vector<std::string>* feature_values) const {
  GET_EDGE_FEATURE_VEC(feature_values, binary_features_,
                       binary_features_idx_, fids, std::string);
}

bool Edge::DeSerialize(const char* s, size_t size) {
  uint64_t src_id = 0;
  uint64_t dst_id = 0;
  BytesReader bytes_reader(s, size);

  // parse id
  if (!bytes_reader.Read(&src_id) ||
      !bytes_reader.Read(&dst_id)) {
    EULER_LOG(ERROR) << "edge id error";
    return false;
  }
  if (!bytes_reader.Read(&type_) ||  // parse type
      !bytes_reader.Read(&weight_)) {  // parse weight
    EULER_LOG(ERROR) << "edge info error, edge_id: " << src_id << "," << dst_id;
    return false;
  }

  id_ = std::make_tuple(src_id, dst_id, type_);

  // parse uint64 feature
  if (!bytes_reader.Read(&uint64_features_idx_)) {
    EULER_LOG(ERROR) << "uint64 feature idx list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  if (!bytes_reader.Read(&uint64_features_)) {
    EULER_LOG(ERROR) << "uint64 feature value list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  if (!bytes_reader.Read(&float_features_idx_)) {
    EULER_LOG(ERROR) << "float feature idx list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  if (!bytes_reader.Read(&float_features_)) {
    EULER_LOG(ERROR) << "float feature value list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  if (!bytes_reader.Read(&binary_features_idx_)) {
    EULER_LOG(ERROR) << "binary feature idx list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  if (!bytes_reader.Read(&binary_features_)) {
    EULER_LOG(ERROR) << "binary feature value list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  return true;
}

bool Edge::Serialize(std::string* s) const {
  BytesWriter bytes_writer;

  uint64_t src_id = std::get<0>(id_);
  uint64_t dst_id = std::get<1>(id_);
  if (!bytes_writer.Write(src_id) ||
      !bytes_writer.Write(dst_id)) {
    EULER_LOG(ERROR) << "edge id error";
    return false;
  }
  if (!bytes_writer.Write(type_) || !bytes_writer.Write(weight_)) {
    EULER_LOG(ERROR) << "edge info error, edge_id: " << src_id << "," << dst_id;
    return false;
  }

  if (!bytes_writer.Write(uint64_features_idx_)) {
    EULER_LOG(ERROR) << "uint64 feature idx list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  if (!bytes_writer.Write(uint64_features_)) {
    EULER_LOG(ERROR) << "uint64 feature value list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  if (!bytes_writer.Write(float_features_idx_)) {
    EULER_LOG(ERROR) << "float feature idx list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  if (!bytes_writer.Write(float_features_)) {
    EULER_LOG(ERROR) << "float feature value list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  if (!bytes_writer.Write(binary_features_idx_)) {
    EULER_LOG(ERROR) << "binary feature idx list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  if (!bytes_writer.Write(binary_features_)) {
    EULER_LOG(ERROR) << "binary feature value list error, edge_id: "
                     << src_id << "," << dst_id << "," << type_;
    return false;
  }

  *s = bytes_writer.data();
  return true;
}

uint32_t Edge::SerializeSize() const {
  uint32_t total = 0;
  total+= BytesSize(std::get<0>(id_));
  total+= BytesSize(std::get<1>(id_));
  total+= BytesSize(type_);
  total+= BytesSize(weight_);
  total+= BytesSize(uint64_features_idx_);
  total+= BytesSize(uint64_features_);
  total+= BytesSize(float_features_idx_);
  total+= BytesSize(float_features_);
  total+= BytesSize(binary_features_idx_);
  total+= BytesSize(binary_features_);

  return total;
}

}  // namespace euler
