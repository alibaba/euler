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

#include "euler/core/graph/node.h"

#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <sstream>
#include <utility>

#include "euler/common/logging.h"
#include "euler/common/bytes_io.h"
#include "euler/common/bytes_compute.h"

namespace euler {

bool id_cmp(std::pair<euler::common::NodeID, float> a,
            std::pair<euler::common::NodeID, float> b) {
  return (a.first < b.first);
}


bool Node::Init(const std::vector<std::vector<uint64_t>>& neighbor_ids,
                    const std::vector<std::vector<float>>& neighbor_weights,
                    const std::vector<std::vector<uint64_t>>& uint64_features,
                    const std::vector<std::vector<float>>& float_features,
                    const std::vector<std::string>& binary_features) {
    if (neighbor_ids.size() != neighbor_weights.size()) {
      EULER_LOG(ERROR) << "ids not equal weights";
      return false;
    }
    float sum_weight = 0;
    std::vector<int32_t> type_ids;
    std::vector<float> type_weights;
    int32_t idx = 0;
    for (size_t i = 0; i < neighbor_ids.size(); ++i) {
      if (neighbor_ids[i].size() != neighbor_weights[i].size()) {
        EULER_LOG(ERROR) << "ids not equal weights";
        return false;
      }

      idx += neighbor_ids[i].size();
      neighbor_info_.neighbor_groups_idx.push_back(idx);

      float type_weight = 0;
      for (size_t j = 0; j < neighbor_ids[i].size(); ++j) {
        neighbor_info_.neighbors.push_back(neighbor_ids[i][j]);
        sum_weight += neighbor_weights[i][j];
        type_weight += neighbor_weights[i][j];
        neighbor_info_.neighbors_weight.push_back(sum_weight);
      }
      type_ids.push_back(i);
      type_weights.push_back(type_weight);
    }
    neighbor_info_.edge_group_collection.Init(type_ids, type_weights);

    idx = 0;
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

inline std::vector<euler::common::IDWeightPair> Node::__SampleNeighbor(
    const std::vector<int32_t>& edge_types,
    int32_t count,
    const NeighborInfo& ni) const {
  std::vector<euler::common::IDWeightPair> err_vec;
  std::vector<euler::common::IDWeightPair> empty_vec;
  std::vector<euler::common::IDWeightPair> vec(count);
  euler::common::CompactWeightedCollection<int32_t> sub_edge_group_collection_;
  if (edge_types.size() > 1 &&
      edge_types.size() < ni.edge_group_collection.GetSize()) {
    std::vector<std::pair<int32_t, float>> edge_type_weight(edge_types.size());
    // rebuild weighted collection
    for (size_t i = 0; i < edge_types.size(); ++i) {
      int32_t edge_type = edge_types[i];
      if (edge_type >= 0 && edge_type <
          static_cast<int32_t>(ni.edge_group_collection.GetSize())) {
        edge_type_weight[i] = ni.edge_group_collection.Get(edge_type);
      } else {
        EULER_LOG(ERROR) << "input edge types vec error:" << edge_type;
        return err_vec;
      }
    }
    sub_edge_group_collection_.Init(edge_type_weight);
  }

  for (int32_t i = 0; i < count; ++i) {
    int32_t edge_type = 0;
    if (edge_types.size() == 1) {
      edge_type = edge_types[0];
      if (edge_type < 0 || edge_type >=
          static_cast<int32_t>(ni.edge_group_collection.GetSize())) {
        return err_vec;
      }
      int32_t pre_idx = edge_type == 0 ? 0 :
                        ni.neighbor_groups_idx[edge_type - 1];
      int32_t cur_idx = ni.neighbor_groups_idx[edge_type] - 1;
      if (cur_idx < pre_idx) {
        return empty_vec;
      }
    } else if (edge_types.size() > 1 &&
               edge_types.size() < ni.edge_group_collection.GetSize()) {
      if (sub_edge_group_collection_.GetSumWeight() == 0) {
        return empty_vec;
      }
      edge_type = sub_edge_group_collection_.Sample().first;
    } else {  // sampling in all edge groups
      if (ni.edge_group_collection.GetSumWeight() == 0) {
        return empty_vec;
      }
      edge_type = ni.edge_group_collection.Sample().first;
    }
    // sample neighbor
    int32_t interval_idx_begin = edge_type == 0 ? 0 :
                                 ni.neighbor_groups_idx[edge_type - 1];
    int32_t interval_idx_end = ni.neighbor_groups_idx[edge_type] - 1;
    size_t mid = euler::common::RandomSelect<euler::common::NodeID>(
        ni.neighbors_weight, interval_idx_begin, interval_idx_end);
    float pre_sum_weight = mid <= 0 ? 0 : ni.neighbors_weight[mid - 1];
    vec[i] = std::make_tuple(ni.neighbors[mid],
                             ni.neighbors_weight[mid] - pre_sum_weight,
                             edge_type);
  }
  return vec;
}

std::vector<euler::common::IDWeightPair>
Node::SampleNeighbor(const std::vector<int32_t>& edge_types,
                     int32_t count) const {
  return __SampleNeighbor(edge_types, count, neighbor_info_);
}

std::vector<euler::common::IDWeightPair>
Node::SampleInNeighbor(const std::vector<int32_t>& edge_types,
                       int32_t count) const {
  return __SampleNeighbor(edge_types, count, in_neighbor_info_);
}


inline std::vector<euler::common::IDWeightPair> Node::__GetFullNeighbor(
  const std::vector<int32_t>& edge_types,
  const NeighborInfo& ni) const {
  std::vector<euler::common::IDWeightPair> vec;
  vec.reserve(edge_types.size() * 2);
  for (size_t i = 0; i < edge_types.size(); ++i) {
    int32_t edge_type = edge_types[i];
    if (edge_type >= 0 &&
        edge_type < static_cast<int32_t>(ni.edge_group_collection.GetSize())) {
      int32_t begin_idx = (edge_type == 0) ?
                          0 : ni.neighbor_groups_idx[edge_type - 1];
      int32_t end_idx = ni.neighbor_groups_idx[edge_type];
      for (int32_t j = begin_idx; j < end_idx; ++j) {
        float pre_sum_weight = j == 0 ? 0 : ni.neighbors_weight[j - 1];
        vec.push_back(
            euler::common::IDWeightPair(ni.neighbors[j],
                                        ni.neighbors_weight[j] - pre_sum_weight,
                                        edge_type));
      }
    }
  }
  return vec;
}

std::vector<euler::common::IDWeightPair>
Node::GetFullNeighbor(const std::vector<int32_t>& edge_types) const {
  return __GetFullNeighbor(edge_types, neighbor_info_);
}

std::vector<euler::common::IDWeightPair>
Node::GetFullInNeighbor(const std::vector<int32_t>& edge_types) const {
  return __GetFullNeighbor(edge_types, in_neighbor_info_);
}

inline std::vector<euler::common::IDWeightPair> Node::__GetSortedFullNeighbor(
  const std::vector<int32_t>& edge_types,
  const NeighborInfo& ni) const {
  std::vector<euler::common::IDWeightPair> vec;
  vec.reserve(edge_types.size() * 2);
  if (edge_types.size() == 0) {
    return vec;
  }
  std::vector<int32_t> ptr_list(ni.edge_group_collection.GetSize());
  std::priority_queue<std::pair<euler::common::NodeID, int32_t>,
      std::vector<std::pair<euler::common::NodeID, int32_t>>,
      NodeComparison> min_heap;
  // init min_heap and ptr_list
  for (size_t i = 0; i < edge_types.size(); ++i) {
    int32_t edge_type = edge_types[i];
    int32_t begin_idx = edge_type == 0 ? 0 :
                        ni.neighbor_groups_idx[edge_type - 1];
    if (edge_type >= 0 &&
        edge_type < static_cast<int32_t>(ni.edge_group_collection.GetSize()) &&
        ni.neighbor_groups_idx[edge_type] - begin_idx > 0) {
      // this kind of edge exist neighbor
      std::pair<euler::common::NodeID, int32_t> id_edge_type(
          ni.neighbors[begin_idx], edge_type);
      min_heap.push(id_edge_type);
      ptr_list[edge_type] = begin_idx;
    }
  }
  // merge
  while (!min_heap.empty()) {
    // get smallest node id
    std::pair<euler::common::NodeID, int32_t> smallest = min_heap.top();
    min_heap.pop();
    int32_t edge_type = smallest.second;
    size_t ptr = ptr_list[edge_type]++;  // get and move ptr
    // put into result
    float pre = ptr == 0 ? 0 : ni.neighbors_weight[ptr - 1];
    vec.push_back(euler::common::IDWeightPair(
        ni.neighbors[ptr], ni.neighbors_weight[ptr] - pre, edge_type));
    // update min_heap
    if (ptr_list[edge_type] < ni.neighbor_groups_idx[edge_type]) {
      std::pair<euler::common::NodeID, int32_t> id_edge_type(
          ni.neighbors[ptr_list[edge_type]], edge_type);
      min_heap.push(id_edge_type);
    }
  }
  return vec;
}

std::vector<euler::common::IDWeightPair>
Node::GetSortedFullNeighbor(const std::vector<int32_t>& edge_types) const {
  return __GetSortedFullNeighbor(edge_types, neighbor_info_);
}

std::vector<euler::common::IDWeightPair>
Node::GetSortedFullInNeighbor(const std::vector<int32_t>& edge_types) const {
  return __GetSortedFullNeighbor(edge_types, in_neighbor_info_);
}

inline std::vector<euler::common::IDWeightPair> Node::__GetTopKNeighbor(
  const std::vector<int32_t>& edge_types,
  int32_t k,
  const NeighborInfo& ni) const {
  std::vector<euler::common::IDWeightPair> vec;
  vec.reserve(k);
  if (k <= 0 || edge_types.size() == 0) {
    return vec;
  }
  bool fail = false;
  std::priority_queue<euler::common::IDWeightPair,
      std::vector<euler::common::IDWeightPair>, NodeWeightComparision> min_heap;
  for (size_t i = 0; !fail && i < edge_types.size(); ++i) {
    int32_t edge_type = edge_types[i];
    if (edge_type >= 0 &&
        edge_type < static_cast<int32_t>(ni.edge_group_collection.GetSize())) {
      int32_t begin_idx = edge_type == 0 ? 0 :
                          ni.neighbor_groups_idx[edge_type - 1];
      for (int32_t j = begin_idx; j <
           ni.neighbor_groups_idx[edge_type]; ++j) {
        float pre = j == 0 ? 0 : ni.neighbors_weight[j - 1];
        if (static_cast<int32_t>(min_heap.size()) < k) {
          min_heap.push(euler::common::IDWeightPair(ni.neighbors[j],
                         ni.neighbors_weight[j] - pre, edge_type));
        } else {
          if (std::get<1>(min_heap.top()) < ni.neighbors_weight[j] - pre) {
            min_heap.pop();
            min_heap.push(euler::common::IDWeightPair(
                ni.neighbors[j], ni.neighbors_weight[j] - pre, edge_type));
          }
        }
      }
    } else {
      EULER_LOG(ERROR) << "input edge types vec error:"<< edge_type;
      fail = true;
    }
  }
  if (!fail) {
    vec.resize(min_heap.size());
    while (!min_heap.empty()) {
      vec[min_heap.size() - 1] = min_heap.top();
      min_heap.pop();
    }
    return vec;
  } else {
    std::vector<euler::common::IDWeightPair> error_vec;
    return error_vec;
  }
}

std::vector<euler::common::IDWeightPair>
Node::GetTopKNeighbor(const std::vector<int32_t>& edge_types,
      int32_t k) const {
  return __GetTopKNeighbor(edge_types, k, neighbor_info_);
}

std::vector<euler::common::IDWeightPair>
Node::GetTopKInNeighbor(const std::vector<int32_t>& edge_types,
      int32_t k) const {
  return __GetTopKNeighbor(edge_types, k, neighbor_info_);
}

#define GET_NODE_FEATURE(F_NUMS_PTR, F_VALUES_PTR, FEATURES, FEATURES_IDX, \
                         FIDS) {                                           \
  for (size_t i = 0; i < FIDS.size(); ++i) {                               \
    int32_t fid = FIDS[i];                                                 \
    if (fid >= 0 && fid < static_cast<int32_t>(FEATURES_IDX.size())) {     \
      int32_t pre = fid == 0 ? 0 : FEATURES_IDX[fid - 1];                  \
      F_NUMS_PTR->push_back(FEATURES_IDX[fid] - pre);                      \
    } else {                                                               \
      F_NUMS_PTR->push_back(0);                                            \
    }                                                                      \
  }                                                                        \
  for (size_t i = 0; i < FIDS.size(); ++i) {                               \
    int32_t fid = FIDS[i];                                                 \
    if (fid >= 0 && fid < static_cast<int32_t>(FEATURES_IDX.size())) {     \
      int32_t pre = fid == 0 ? 0 : FEATURES_IDX[fid - 1];                  \
      int32_t now = FEATURES_IDX[fid];                                     \
      F_VALUES_PTR->insert(F_VALUES_PTR->end(),                            \
                           FEATURES.begin() + pre,                         \
                           FEATURES.begin() + now);                        \
    }                                                                      \
  }                                                                        \
}                                                                          \

#define GET_NODE_FEATURE_VEC(F_VALUES_PTR, FEATURES,                       \
                             FEATURES_IDX, FIDS, TYPE) {                   \
  for (size_t i = 0; i < FIDS.size(); ++i) {                               \
    int32_t fid = FIDS[i];                                                 \
    if (fid >= 0 && fid < static_cast<int32_t>(FEATURES_IDX.size())) {     \
      int32_t pre = fid == 0 ? 0 : FEATURES_IDX[fid - 1];                  \
      int32_t now = FEATURES_IDX[fid];                                     \
      F_VALUES_PTR->emplace_back(TYPE(FEATURES.begin() + pre,              \
                                 FEATURES.begin() + now));                 \
    }                                                                      \
  }                                                                        \
}

void Node::GetUint64Feature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<uint64_t>* feature_values) const {
  GET_NODE_FEATURE(feature_nums, feature_values, uint64_features_,
                   uint64_features_idx_, fids);
}

void Node::GetUint64Feature(
      const std::vector<int32_t>& fids,
      std::vector<std::vector<uint64_t>>* feature_values) const {
  GET_NODE_FEATURE_VEC(feature_values, uint64_features_,
                       uint64_features_idx_, fids, std::vector<uint64_t>);
}

void Node::GetFloat32Feature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<float>* feature_values) const {
  GET_NODE_FEATURE(feature_nums, feature_values, float_features_,
                   float_features_idx_, fids);
}

void Node::GetFloat32Feature(
      const std::vector<int32_t>& fids,
      std::vector<std::vector<float>>* feature_values) const {
  GET_NODE_FEATURE_VEC(feature_values, float_features_,
                       float_features_idx_, fids, std::vector<float>);
}

void Node::GetBinaryFeature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<char>* feature_values) const {
  GET_NODE_FEATURE(feature_nums, feature_values, binary_features_,
                   binary_features_idx_, fids);
}

void Node::GetBinaryFeature(
      const std::vector<int32_t>& fids,
      std::vector<std::string>* feature_values) const {
  GET_NODE_FEATURE_VEC(feature_values, binary_features_,
                       binary_features_idx_, fids, std::string);
}

#undef GET_NODE_FEATURE
#undef GET_NODE_FEATURE_VEC

bool Node::DeSerialize(const char* s, size_t size) {
  BytesReader bytes_reader(s, size);
  if (!bytes_reader.Read(&id_) ||  // parse node id
      !bytes_reader.Read(&type_) ||  // parse node type
      !bytes_reader.Read(&weight_)) {  // parse node weight
    EULER_LOG(ERROR) << "node info error";
    return false;
  }

  std::vector<int32_t> edge_group_ids;
  std::vector<float> edge_group_weights;
  if (!bytes_reader.Read(&edge_group_ids)) {
    EULER_LOG(ERROR) << "edge group id list error, node_id: " << id_;
    return false;
  }

  if (!bytes_reader.Read(&edge_group_weights)) {
    EULER_LOG(ERROR) << "edge group weight list error, node_id: " << id_;
    return false;
  }

  // build edge_group_collection_
  if (!neighbor_info_.edge_group_collection.Init(edge_group_ids,
                                                 edge_group_weights)) {
    EULER_LOG(ERROR) << "neighbor edge group collection error, node_id: "
                     << id_;
    return false;
  }

  // build neighbors info
  if (!bytes_reader.Read(&neighbor_info_.neighbor_groups_idx)) {
    EULER_LOG(ERROR) << "neighbor groups idx error, node_id: " << id_;
    return false;
  }

  if (!bytes_reader.Read(&neighbor_info_.neighbors)) {
    EULER_LOG(ERROR) << "neighbors error, node_id: " << id_;
    return false;
  }

  if (!bytes_reader.Read(&neighbor_info_.neighbors_weight)) {
    EULER_LOG(ERROR) << "neighbors weights error, node_id: " << id_;
    return false;
  }

  edge_group_ids.clear();
  edge_group_weights.clear();
  if (!bytes_reader.Read(&edge_group_ids)) {
    EULER_LOG(ERROR) << "edge group id list error, node_id: " << id_;
    return false;
  }

  if (!bytes_reader.Read(&edge_group_weights)) {
    EULER_LOG(ERROR) << "edge group weight list error, node_id: " << id_;
    return false;
  }

  // build edge_group_collection_
  if (!in_neighbor_info_.edge_group_collection.Init(edge_group_ids,
                                                    edge_group_weights)) {
    EULER_LOG(ERROR) << "in neighbor edge group collection error, node_id: "
                     << id_;
    return false;
  }
  // build in_neighbors info
  if (!bytes_reader.Read(&in_neighbor_info_.neighbor_groups_idx)) {
    EULER_LOG(ERROR) << "in neighbor groups idx error, node_id: " << id_;
    return false;
  }

  if (!bytes_reader.Read(&in_neighbor_info_.neighbors)) {
    EULER_LOG(ERROR) << "in neighbors error, node_id: " << id_;
    return false;
  }

  if (!bytes_reader.Read(&in_neighbor_info_.neighbors_weight)) {
    EULER_LOG(ERROR) << "in neighbors weights error, node_id: " << id_;
    return false;
  }

  // parse uint64 feature
  if (!bytes_reader.Read(&uint64_features_idx_)) {
    EULER_LOG(ERROR) << "uint64 feature idx list error, node_id: " << id_;
    return false;
  }
  if (!bytes_reader.Read(&uint64_features_)) {
    EULER_LOG(ERROR) << "uint64 feature value list error, node_id: " << id_;
    return false;
  }

  // parse float feature
  if (!bytes_reader.Read(&float_features_idx_)) {
    EULER_LOG(ERROR) << "float feature idx list error, node_id: " << id_;
    return false;
  }

  if (!bytes_reader.Read(&float_features_)) {
    EULER_LOG(ERROR) << "float feature value list error, node_id: " << id_;
    return false;
  }

  // parse binary feature
  if (!bytes_reader.Read(&binary_features_idx_)) {
    EULER_LOG(ERROR) << "binary feature idx list error, node_id: " << id_;
    return false;
  }
  if (!bytes_reader.Read(&binary_features_)) {
    EULER_LOG(ERROR) << "binary feature value list error, node_id: " << id_;
    return false;
  }

  return true;
}

bool Node::Serialize(std::string* s) const {
  BytesWriter bytes_writer;

  if (!bytes_writer.Write(id_) || !bytes_writer.Write(type_) ||
      !bytes_writer.Write(weight_)) {
    EULER_LOG(ERROR) << "node info error";
    return false;
  }

  // parse neighbor info
  if (!bytes_writer.Write(neighbor_info_.edge_group_collection.GetIds())) {
    EULER_LOG(ERROR) << "edge group id list error, node_id: " << id_;
    return false;
  }

  if (!bytes_writer.Write(neighbor_info_.edge_group_collection.GetWeights())) {
    EULER_LOG(ERROR) << "edge group weight list error, node_id: " << id_;
    return false;
  }

  if (!bytes_writer.Write(neighbor_info_.neighbor_groups_idx)) {
    EULER_LOG(ERROR) << "neighbor groups idx error, node_id: " << id_;
    return false;
  }

  if (!bytes_writer.Write(neighbor_info_.neighbors)) {
    EULER_LOG(ERROR) << "neighbors error, node_id: " << id_;
    return false;
  }

  if (!bytes_writer.Write(neighbor_info_.neighbors_weight)) {
    EULER_LOG(ERROR) << "neighbors weights error, node_id: " << id_;
    return false;
  }

  // parse in-neighbor info
  if (!bytes_writer.Write(in_neighbor_info_.edge_group_collection.GetIds())) {
    EULER_LOG(ERROR) << "in edge group id list error, node_id: " << id_;
    return false;
  }

  if (!bytes_writer.Write(
         in_neighbor_info_.edge_group_collection.GetWeights())) {
    EULER_LOG(ERROR) << "in edge group weight list error, node_id: " << id_;
    return false;
  }

  if (!bytes_writer.Write(in_neighbor_info_.neighbor_groups_idx)) {
    EULER_LOG(ERROR) << "in neighbor groups idx error, node_id: " << id_;
    return false;
  }

  if (!bytes_writer.Write(in_neighbor_info_.neighbors)) {
    EULER_LOG(ERROR) << "neighbors error, node_id: " << id_;
    return false;
  }

  if (!bytes_writer.Write(in_neighbor_info_.neighbors_weight)) {
    EULER_LOG(ERROR) << "in neighbors weights error, node_id: " << id_;
    return false;
  }
  // parse uint64 feature
  if (!bytes_writer.Write(uint64_features_idx_)) {
    EULER_LOG(ERROR) << "uint64 feature idx list error, node_id: " << id_;
    return false;
  }
  if (!bytes_writer.Write(uint64_features_)) {
    EULER_LOG(ERROR) << "uint64 feature value list error, node_id: " << id_;
    return false;
  }

  // parse float feature
  if (!bytes_writer.Write(float_features_idx_)) {
    EULER_LOG(ERROR) << "float feature idx list error, node_id: " << id_;
    return false;
  }

  if (!bytes_writer.Write(float_features_)) {
    EULER_LOG(ERROR) << "float feature value list error, node_id: " << id_;
    return false;
  }

  // parse binary feature
  if (!bytes_writer.Write(binary_features_idx_)) {
    EULER_LOG(ERROR) << "binary feature idx list error, node_id: " << id_;
    return false;
  }
  if (!bytes_writer.Write(binary_features_)) {
    EULER_LOG(ERROR) << "binary feature value list error, node_id: " << id_;
    return false;
  }

  *s = bytes_writer.data();
  return true;
}

uint32_t Node::SerializeSize() const {
  uint32_t total = 0;

  total += BytesSize(id_);
  total += BytesSize(type_);
  total += BytesSize(weight_);
  total += BytesSize(neighbor_info_.edge_group_collection.GetIds());
  total += BytesSize(neighbor_info_.edge_group_collection.GetWeights());
  total += BytesSize(neighbor_info_.neighbor_groups_idx);
  total += BytesSize(neighbor_info_.neighbors);
  total += BytesSize(neighbor_info_.neighbors_weight);
  total += BytesSize(in_neighbor_info_.edge_group_collection.GetIds());
  total += BytesSize(in_neighbor_info_.edge_group_collection.GetWeights());
  total += BytesSize(in_neighbor_info_.neighbor_groups_idx);
  total += BytesSize(in_neighbor_info_.neighbors);
  total += BytesSize(in_neighbor_info_.neighbors_weight);
  total += BytesSize(uint64_features_idx_);
  total += BytesSize(uint64_features_);
  total += BytesSize(float_features_idx_);
  total += BytesSize(float_features_);
  total += BytesSize(binary_features_idx_);
  total += BytesSize(binary_features_);

  return total;
}

int32_t Node::GetFloat32FeatureValueNum() const {
  int32_t num = 1, pre = 0;
  for (size_t i = 0; i < float_features_idx_.size(); ++i) {
    num = std::max(float_features_idx_[i] - pre, num);
    pre = float_features_idx_[i];
  }
  return num;
}

int32_t Node::GetUint64FeatureValueNum() const {
  int32_t num = 1, pre = 0;
  for (size_t i = 0; i < uint64_features_idx_.size(); ++i) {
    num = std::max(uint64_features_idx_[i] - pre, num);
    pre = uint64_features_idx_[i];
  }
  return num;
}

int32_t Node::GetBinaryFeatureValueNum() const {
  int32_t num = 1, pre = 0;
  for (size_t i = 0; i < binary_features_idx_.size(); ++i) {
    num = std::max(binary_features_idx_[i] - pre, num);
    pre = binary_features_idx_[i];
  }
  return num;
}

}  // namespace euler
