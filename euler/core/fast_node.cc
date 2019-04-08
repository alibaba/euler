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

#include "euler/core/fast_node.h"

#include <string>
#include <vector>
#include <algorithm>
#include <queue>
#include <utility>

#include "glog/logging.h"

#include "euler/common/bytes_reader.h"
#include "euler/common/fast_weighted_collection.h"

namespace euler {
namespace core {

FastNode::FastNode(euler::common::NodeID id, float weight)
    : Node(id, weight) {
}

FastNode::FastNode() {
}

FastNode::~FastNode() {
  for (size_t i = 0; i < neighbor_collection_list_.size(); ++i) {
    delete neighbor_collection_list_[i];
  }
}

std::vector<euler::common::IDWeightPair>
FastNode::SampleNeighbor(const std::vector<int32_t>& edge_types,
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

  // sampling
  for (int32_t i = 0; i < count; ++i) {
    int32_t edge_type = 0;
    if (edge_types.size() == 1) {
      edge_type = edge_types[0];
      if (edge_type < 0 ||
          edge_type >= static_cast<int32_t>(edge_group_collection_.GetSize())) {
        return err_vec;
      }
      if (neighbor_collection_list_[edge_type]->GetSize() == 0) {
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
    std::pair<euler::common::NodeID, float> id_weight =
        neighbor_collection_list_[edge_type]->Sample();
    vec[i] = std::make_tuple(id_weight.first, id_weight.second, edge_type);
  }
  return vec;
}

std::vector<euler::common::IDWeightPair>
FastNode::GetFullNeighbor(const std::vector<int32_t>& edge_types) const {
  std::vector<euler::common::IDWeightPair> vec;
  vec.reserve(edge_types.size() * 2);
  for (size_t i = 0; i < edge_types.size(); ++i) {
    int32_t edge_type = edge_types[i];
    if (edge_type >= 0 &&
        edge_type < static_cast<int32_t>(neighbor_collection_list_.size())) {
      size_t edge_group_size = neighbor_collection_list_[edge_type]->GetSize();
      for (size_t j = 0; j < edge_group_size; ++j) {
        std::pair<uint64_t, float> p =
            neighbor_collection_list_[edge_type]->Get(j);
        vec.push_back(euler::common::IDWeightPair(p.first, p.second, edge_type));
      }
    }
  }
  return vec;
}

std::vector<euler::common::IDWeightPair>
FastNode::GetSortedFullNeighbor(const std::vector<int32_t>& edge_types) const {
  std::vector<euler::common::IDWeightPair> vec;
  vec.reserve(edge_types.size() * 2);
  if (edge_types.size() == 0) {
    return vec;
  }
  std::vector<size_t> ptr_list(edge_group_collection_.GetSize());
  std::priority_queue<std::pair<euler::common::NodeID, int32_t>,
      std::vector<std::pair<euler::common::NodeID, int32_t>>,
      NodeComparison> min_heap;
  // init min_heap
  for (size_t i = 0; i < edge_types.size(); ++i) {
    int32_t edge_type = edge_types[i];
    if (edge_type >= 0 &&
        edge_type < static_cast<int32_t>(neighbor_collection_list_.size()) &&
        neighbor_collection_list_[edge_type]->GetSize() > 0) {
      // this kind of edge exist neighbor
      std::pair<euler::common::NodeID, int32_t> id_edge_type(
          neighbor_collection_list_[edge_type]->Get(0).first, edge_type);
      min_heap.push(id_edge_type);
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
    std::pair<euler::common::NodeID, float> id_weight =
        neighbor_collection_list_[edge_type]->Get(ptr);
    vec.push_back(euler::common::IDWeightPair(id_weight.first,
                                              id_weight.second,
                                              edge_type));
    // update min_heap
    if (ptr_list[edge_type] < neighbor_collection_list_[edge_type]->GetSize()) {
      std::pair<euler::common::NodeID, int32_t> id_edge_type(
          neighbor_collection_list_[edge_type]->
          Get(ptr_list[edge_type]).first, edge_type);
      min_heap.push(id_edge_type);
    }
  }
  return vec;
}

std::vector<euler::common::IDWeightPair>
FastNode::GetTopKNeighbor(const std::vector<int32_t>& edge_types,
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
      for (size_t j = 0; j <
          neighbor_collection_list_[edge_type]->GetSize(); ++j) {
        std::pair<euler::common::NodeID, float> id_weight =
            neighbor_collection_list_[edge_type]->Get(j);
        if (static_cast<int32_t>(min_heap.size()) < k) {
          min_heap.push(euler::common::IDWeightPair(id_weight.first,
                                                    id_weight.second,
                                                    edge_type));
        } else {
          if (std::get<1>(min_heap.top()) < id_weight.second) {
            min_heap.pop();
            min_heap.push(euler::common::IDWeightPair(id_weight.first,
                                                      id_weight.second,
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

#define GET_NODE_FEATURE(F_NUMS_PTR, F_VALUES_PTR, FEATURES, FIDS) { \
  for (size_t i = 0; i < FIDS.size(); ++i) {                         \
    int32_t fid = FIDS[i];                                           \
    if (fid >= 0 && fid < static_cast<int32_t>(FEATURES.size())) {   \
      F_NUMS_PTR->push_back(FEATURES[fid].size());                   \
    } else {                                                         \
      F_NUMS_PTR->push_back(0);                                      \
    }                                                                \
  }                                                                  \
  for (size_t i = 0; i < FIDS.size(); ++i) {                         \
    int32_t fid = FIDS[i];                                           \
    if (fid >= 0 && fid < static_cast<int32_t>(FEATURES.size())) {   \
      F_VALUES_PTR->insert(F_VALUES_PTR->end(),                      \
                           FEATURES[fid].begin(),                    \
                           FEATURES[fid].end());                     \
    }                                                                \
  }                                                                  \
}

void FastNode::GetUint64Feature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<uint64_t>* feature_values) const {
  GET_NODE_FEATURE(feature_nums, feature_values, uint64_features_, fids);
}

void FastNode::GetFloat32Feature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<float>* feature_values) const {
  GET_NODE_FEATURE(feature_nums, feature_values, float_features_, fids);
}

void FastNode::GetBinaryFeature(
    const std::vector<int32_t>& fids,
    std::vector<uint32_t>* feature_nums,
    std::vector<char>* feature_values) const {
  GET_NODE_FEATURE(feature_nums, feature_values, binary_features_, fids);
}

bool FastNode::DeSerialize(const char* s, size_t size) {
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
    LOG(ERROR) << "edge group weight collection error, node_id: " << id_;
    return false;
  }
  // build neighbor_collection_list_
  std::vector<std::vector<uint64_t>> ids_list(edge_group_num);
  for (int32_t i = 0; i < edge_group_num; ++i) {
    ids_list[i] = std::vector<uint64_t>();
    if (!bytes_reader.GetUInt64List(edge_group_size_list[i], &ids_list[i])) {
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
  neighbor_collection_list_.resize(edge_group_num);
  for (int32_t i = 0; i < edge_group_num; ++i) {
    neighbor_collection_list_[i] =
      new euler::common::FastWeightedCollection<euler::common::NodeID>();
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
    neighbor_collection_list_[i]->Init(id_weight_pairs);
  }

  // parse uint64 feature
  int32_t uint64_feature_type_num = 0;
  if (!bytes_reader.GetInt32(&uint64_feature_type_num)) {
    LOG(ERROR) << "uint64 feature type num error, node_id: " << id_;
    return false;
  }
  std::vector<int32_t> uint64_feature_size_list;
  uint64_features_.resize(uint64_feature_type_num);
  if (!bytes_reader.GetInt32List(uint64_feature_type_num,
                                 &uint64_feature_size_list)) {
    LOG(ERROR) << "uint64 feature idx list error, node_id: " << id_;
    return false;
  }
  for (int32_t i = 0; i < uint64_feature_type_num; ++i) {
    if (!bytes_reader.GetUInt64List(uint64_feature_size_list[i],
                                    &uint64_features_[i])) {
      LOG(ERROR) << "uint64 feature value list error, node_id: " << id_;
      return false;
    }
  }
  // parse float feature
  int32_t float_feature_type_num = 0;
  if (!bytes_reader.GetInt32(&float_feature_type_num)) {
    LOG(ERROR) << "float feature type num error, node_id_: " << id_;
    return false;
  }
  std::vector<int32_t> float_feature_size_list;
  float_features_.resize(float_feature_type_num);
  if (!bytes_reader.GetInt32List(float_feature_type_num,
                                 &float_feature_size_list)) {
    LOG(ERROR) << "float feature idx list error, node_id_: " << id_;
    return false;
  }
  for (int32_t i = 0; i < float_feature_type_num; ++i) {
    if (!bytes_reader.GetFloatList(float_feature_size_list[i],
                                   &float_features_[i])) {
      LOG(ERROR) << "float feature value list error, node_id: " << id_;
      return false;
    }
  }

  // parse binary feature
  int32_t binary_feature_type_num = 0;
  if (!bytes_reader.GetInt32(&binary_feature_type_num)) {
    LOG(ERROR) << "binary feature type num error, node_id: " << id_;
    return false;
  }
  std::vector<int32_t> binary_feature_size_list;
  binary_features_.resize(binary_feature_type_num);
  if (!bytes_reader.GetInt32List(binary_feature_type_num,
                                 &binary_feature_size_list)) {
    LOG(ERROR) << "binary feature idx list error, node_id: " << id_;
    return false;
  }
  for (int32_t i = 0; i < binary_feature_type_num; ++i) {
    if (!bytes_reader.GetString(binary_feature_size_list[i],
                                &binary_features_[i])) {
      LOG(ERROR) << "binary feature value list error, node_id: " << id_;
      return false;
    }
  }

  return true;
}

std::string FastNode::Serialize() const {
  std::string str;
  return str;
}

}  // namespace core
}  // namespace euler
