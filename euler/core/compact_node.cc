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

#include "euler/core/compact_node.h"

#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <sstream>
#include <utility>

#include "glog/logging.h"

#include "euler/common/bytes_reader.h"

namespace euler {
namespace core {

CompactNode::CompactNode(euler::common::NodeID id, float weight)
  : Node(id, weight) {
}

CompactNode::CompactNode() {
}

CompactNode::~CompactNode() {
}

std::vector<euler::common::IDWeightPair>
CompactNode::SampleNeighbor(
    const std::vector<int32_t>& edge_types,
    int32_t count) const {
  std::vector<euler::common::IDWeightPair> err_vec;
  std::vector<euler::common::IDWeightPair> empty_vec;
  std::vector<euler::common::IDWeightPair> vec(count);
  euler::common::CompactWeightedCollection<int32_t> sub_edge_group_collection_;
  if (edge_types.size() > 1 &&
      edge_types.size() < edge_group_collection_.GetSize()) {
    std::vector<std::pair<int32_t, float>> edge_type_weight(edge_types.size());
    // rebuild weighted collection
    for (size_t i = 0; i < edge_types.size(); ++i) {
      int32_t edge_type = edge_types[i];
      if (edge_type >= 0 &&
          edge_type < static_cast<int32_t>(edge_group_collection_.GetSize())) {
        edge_type_weight[i] = edge_group_collection_.Get(edge_type);
      } else {
        LOG(ERROR) << "edge types vec error";
        return err_vec;
      }
    }
    sub_edge_group_collection_.Init(edge_type_weight);
  }
  for (int32_t i = 0; i < count; ++i) {
    int32_t edge_type = 0;
    if (edge_types.size() == 1) {
      edge_type = edge_types[0];
      if (edge_type < 0 ||
          edge_type >= static_cast<int32_t>(edge_group_collection_.GetSize())) {
        return err_vec;
      }
      int32_t pre_idx = edge_type == 0 ? 0 :
                        neighbor_groups_idx_[edge_type - 1];
      int32_t cur_idx = neighbor_groups_idx_[edge_type];
      if (cur_idx <= pre_idx) {
        return empty_vec;
      }
    } else if (edge_types.size() > 1 &&
               edge_types.size() < edge_group_collection_.GetSize()) {
      if (sub_edge_group_collection_.GetSumWeight() == 0) {
        return empty_vec;
      }
      edge_type = sub_edge_group_collection_.Sample().first;
    } else {  // sampling in all edge groups
      if (edge_group_collection_.GetSumWeight() == 0) {
        return empty_vec;
      }
      edge_type = edge_group_collection_.Sample().first;
    }
    // sample neighbor
    int32_t interval_idx_begin = edge_type == 0 ? 0 :
                                 neighbor_groups_idx_[edge_type - 1];
    int32_t interval_idx_end = neighbor_groups_idx_[edge_type] - 1;
    size_t mid = euler::common::RandomSelect<euler::common::NodeID>(
        neighbors_weight_, interval_idx_begin, interval_idx_end);
    float pre_sum_weight = mid <= 0 ? 0 : neighbors_weight_[mid - 1];
    vec[i] = std::make_tuple(neighbors_[mid],
                             neighbors_weight_[mid] - pre_sum_weight,
                             edge_type);
  }
  return vec;
}

std::vector<euler::common::IDWeightPair>
CompactNode::GetFullNeighbor(const std::vector<int32_t>& edge_types) const {
  std::vector<euler::common::IDWeightPair> vec;
  vec.reserve(edge_types.size() * 2);
  for (size_t i = 0; i < edge_types.size(); ++i) {
    int32_t edge_type = edge_types[i];
    if (edge_type >= 0 &&
        edge_type < static_cast<int32_t>(edge_group_collection_.GetSize())) {
      int32_t begin_idx = edge_type == 0 ? 0 : neighbor_groups_idx_[edge_type - 1];
      int32_t end_idx = neighbor_groups_idx_[edge_type];
      for (int32_t j = begin_idx; j < end_idx; ++j) {
        float pre_sum_weight = j == 0 ? 0 : neighbors_weight_[j - 1];
        vec.push_back(
            euler::common::IDWeightPair(neighbors_[j],
                                        neighbors_weight_[j] - pre_sum_weight,
                                        edge_type));
      }
    }
  }
  return vec;
}

std::vector<euler::common::IDWeightPair>
CompactNode::GetSortedFullNeighbor(const std::vector<int32_t>& edge_types) const {
  std::vector<euler::common::IDWeightPair> vec;
  vec.reserve(edge_types.size() * 2);
  if (edge_types.size() == 0) {
    return vec;
  }
  std::vector<int32_t> ptr_list(edge_group_collection_.GetSize());
  std::priority_queue<std::pair<euler::common::NodeID, int32_t>,
      std::vector<std::pair<euler::common::NodeID, int32_t>>,
      NodeComparison> min_heap;
  // init min_heap and ptr_list
  for (size_t i = 0; i < edge_types.size(); ++i) {
    int32_t edge_type = edge_types[i];
    int32_t begin_idx = edge_type == 0 ? 0 :
                        neighbor_groups_idx_[edge_type - 1];
    if (edge_type >= 0 &&
        edge_type < static_cast<int32_t>(edge_group_collection_.GetSize()) &&
        neighbor_groups_idx_[edge_type] - begin_idx > 0) {
      // this kind of edge exist neighbor
      std::pair<euler::common::NodeID, int32_t> id_edge_type(
          neighbors_[begin_idx], edge_type);
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
    float pre = ptr == 0 ? 0 : neighbors_weight_[ptr - 1];
    vec.push_back(euler::common::IDWeightPair(
        neighbors_[ptr], neighbors_weight_[ptr] - pre, edge_type));
    // update min_heap
    if (ptr_list[edge_type] < neighbor_groups_idx_[edge_type]) {
      std::pair<euler::common::NodeID, int32_t> id_edge_type(
          neighbors_[ptr_list[edge_type]], edge_type);
      min_heap.push(id_edge_type);
    }
  }
  return vec;
}

std::vector<euler::common::IDWeightPair>
CompactNode::GetTopKNeighbor(const std::vector<int32_t>& edge_types,
      int32_t k) const {
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
        edge_type < static_cast<int32_t>(edge_group_collection_.GetSize())) {
      int32_t begin_idx = edge_type == 0 ? 0 :
                          neighbor_groups_idx_[edge_type - 1];
      for (int32_t j = begin_idx; j <
           neighbor_groups_idx_[edge_type]; ++j) {
        float pre = j == 0 ? 0 : neighbors_weight_[j - 1];
        if (static_cast<int32_t>(min_heap.size()) < k) {
          min_heap.push(euler::common::IDWeightPair(neighbors_[j],
                                                    neighbors_weight_[j] - pre,
                                                    edge_type));
        } else {
          if (std::get<1>(min_heap.top()) < neighbors_weight_[j] - pre) {
            min_heap.pop();
            min_heap.push(euler::common::IDWeightPair(neighbors_[j],
                                                      neighbors_weight_[j] - pre,
                                                      edge_type));
          }
        }
      }
    } else {
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

void CompactNode::GetUint64Feature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<uint64_t>* feature_values) const {
  GET_NODE_FEATURE(feature_nums, feature_values, uint64_features_,
                   uint64_features_idx_, fids);
}

void CompactNode::GetFloat32Feature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<float>* feature_values) const {
  GET_NODE_FEATURE(feature_nums, feature_values, float_features_,
                   float_features_idx_, fids);
}

void CompactNode::GetBinaryFeature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<char>* feature_values) const {
  GET_NODE_FEATURE(feature_nums, feature_values, binary_features_,
                   binary_features_idx_, fids);
}

bool CompactNode::DeSerialize(const char* s, size_t size) {
  euler::common::BytesReader bytes_reader(s, size);
  if (!bytes_reader.GetUInt64(&id_) ||  // parse node id
      !bytes_reader.GetInt32(&type_) ||  // parse node type
      !bytes_reader.GetFloat(&weight_)) {  // parse node weight
    LOG(ERROR) << "node info error";
    return false;
  }

  int32_t edge_group_num = 0;
  if (!bytes_reader.GetInt32(&edge_group_num)) {
    LOG(ERROR) << "edge group num error, node_id: " << id_;
    return false;
  }

  std::vector<int32_t> edge_group_ids(edge_group_num);
  for (int32_t i = 0; i < edge_group_num; ++i) {
    edge_group_ids[i] = i;
  }

  std::vector<int32_t> edge_group_size_list;
  if (!bytes_reader.GetInt32List(edge_group_num,
                                 &edge_group_size_list)) {
    LOG(ERROR) << "edge group size list error, node_id: " << id_;
    return false;
  }

  std::vector<float> edge_group_weight_list;
  if (!bytes_reader.GetFloatList(edge_group_num,
                                 &edge_group_weight_list)) {
    LOG(ERROR) << "edge group weight list error, node_id: " << id_;
    return false;
  }

  // build edge_group_collection_
  if (!edge_group_collection_.Init(edge_group_ids, edge_group_weight_list)) {
    LOG(ERROR) << "edge group collection error, node_id: " << id_;
    return false;
  }
  // build neighbors info
  int32_t total_neighbors_num = 0;
  std::vector<std::vector<uint64_t>> ids_list(edge_group_num);
  neighbor_groups_idx_.resize(edge_group_num);
  for (int32_t i = 0; i < edge_group_num; ++i) {
    total_neighbors_num += edge_group_size_list[i];
    neighbor_groups_idx_[i] = total_neighbors_num;
    ids_list[i] = std::vector<uint64_t>();
    if (!bytes_reader.GetUInt64List(edge_group_size_list[i],
                                    &ids_list[i])) {
      LOG(ERROR) << "neighbor id list error, node_id: " << id_;
      return false;
    }
  }

  std::vector<std::vector<float>> weights_list(edge_group_num);
  for (int32_t i = 0; i < edge_group_num; ++i) {
    weights_list[i] = std::vector<float>();
    if (!bytes_reader.GetFloatList(edge_group_size_list[i],
                                   &weights_list[i])) {
      LOG(ERROR) << "neighbor weight list error, node_id: " << id_;
      return false;
    }
  }

  neighbors_.reserve(total_neighbors_num);
  neighbors_weight_.reserve(total_neighbors_num);
  float sum_weight = 0;
  for (int32_t i = 0; i < edge_group_num; ++i) {
    // sort ids_list[i], weights_list[i]
    std::vector<std::pair<euler::common::NodeID, float>>
        id_weight_pairs(ids_list[i].size());
    uint64_t pre = 0;
    bool in_order = true;
    for (size_t j = 0; j < ids_list[i].size(); ++j) {
      id_weight_pairs[j] = std::pair<euler::common::NodeID, float>
          (ids_list[i][j], weights_list[i][j]);
      in_order = in_order && pre < id_weight_pairs[j].first;
      pre = id_weight_pairs[j].first;
    }
    if (!in_order) {
      std::sort(id_weight_pairs.begin(), id_weight_pairs.end(), id_cmp);
    }
    for (size_t j = 0; j < id_weight_pairs.size(); ++j) {
      neighbors_.push_back(id_weight_pairs[j].first);
      sum_weight += id_weight_pairs[j].second;
      neighbors_weight_.push_back(sum_weight);
    }
  }
  // parse uint64 feature
  int32_t uint64_feature_type_num = 0;
  if (!bytes_reader.GetInt32(&uint64_feature_type_num)) {
    LOG(ERROR) << "uint64 feature type num error, node_id: " << id_;
    return false;
  }
  if (!bytes_reader.GetInt32List(uint64_feature_type_num,
                                 &uint64_features_idx_)) {
    LOG(ERROR) << "uint64 feature idx list error, node_id: " << id_;
    return false;
  }
  int32_t uint64_fv_num = 0;
  for (int32_t i = 0; i < uint64_feature_type_num; ++i) {
    uint64_fv_num += uint64_features_idx_[i];
    uint64_features_idx_[i] = uint64_fv_num;
  }
  if (!bytes_reader.GetUInt64List(uint64_fv_num, &uint64_features_)) {
    LOG(ERROR) << "uint64 feature value list error, node_id: " << id_;
    return false;
  }

  // parse float feature
  int32_t float_feature_type_num = 0;
  if (!bytes_reader.GetInt32(&float_feature_type_num)) {
    LOG(ERROR) << "float feature type num error, node_id: " << id_;
    return false;
  }
  if (!bytes_reader.GetInt32List(float_feature_type_num,
                                 &float_features_idx_)) {
    LOG(ERROR) << "float feature idx list error, node_id: " << id_;
    return false;
  }
  int32_t float_fv_num = 0;
  for (int32_t i = 0; i < float_feature_type_num; ++i) {
    float_fv_num += float_features_idx_[i];
    float_features_idx_[i] = float_fv_num;
  }
  if (!bytes_reader.GetFloatList(float_fv_num, &float_features_)) {
    LOG(ERROR) << "float feature value list error, node_id: " << id_;
    return false;
  }

  // parse binary feature
  int32_t binary_feature_type_num = 0;
  if (!bytes_reader.GetInt32(&binary_feature_type_num)) {
    LOG(ERROR) << "binary feature type num error, node_id: " << id_;
    return false;
  }
  if (!bytes_reader.GetInt32List(binary_feature_type_num,
                                 &binary_features_idx_)) {
    LOG(ERROR) << "binary feature idx list error, node_id: " << id_;
    return false;
  }
  int32_t binary_fv_num = 0;
  for (int32_t i = 0; i < binary_feature_type_num; ++i) {
    binary_fv_num += binary_features_idx_[i];
    binary_features_idx_[i] = binary_fv_num;
  }
  if (!bytes_reader.GetString(binary_fv_num, &binary_features_)) {
    LOG(ERROR) << "binary feature value list error, node_id: " << id_;
    return false;
  }
  return true;
}

std::string CompactNode::Serialize() const {
  return "";
}

}  // namespace core
}  // namespace euler
