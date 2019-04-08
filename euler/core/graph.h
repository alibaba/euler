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

#ifndef EULER_CORE_GRAPH_H_
#define EULER_CORE_GRAPH_H_

#include <vector>
#include <unordered_map>
#include <iostream>
#include <string>
#include <utility>
#include <memory>
#include <mutex>

#include "glog/logging.h"

#include "euler/common/data_types.h"
#include "euler/core/node.h"
#include "euler/core/edge.h"

namespace euler {
namespace core {

class Graph {
 public:
  Graph() {
    node_map_.reserve(10000000);
    edge_map_.reserve(10000000);
    node_type_num_ = 0;
    edge_type_num_ = 0;
    global_sampler_ok_ = false;
    global_edge_sampler_ok_ = false;
  }

  virtual ~Graph() {
    std::unordered_map<euler::common::NodeID, Node*>::iterator n_map_it =
        node_map_.begin();
    while (n_map_it != node_map_.end()) {
      delete n_map_it->second;
      ++n_map_it;
    }
    std::unordered_map<euler::common::EdgeID, Edge*,
        euler::common::EdgeIDHashFunc, euler::common::EdgeIDEqualKey>::iterator
        e_map_it = edge_map_.begin();
    while (e_map_it != edge_map_.end()) {
      delete e_map_it->second;
      ++e_map_it;
    }
  }

  // Randomly Sample nodes with the specified type
  virtual std::vector<euler::common::NodeID> SampleNode(
      int node_type, int count) const = 0;

  // Randomly sample edges with the specified type
  virtual std::vector<euler::common::EdgeID>
  SampleEdge(int edge_type, int count) const = 0;

  virtual Node* GetNodeByID(euler::common::NodeID id) {
    if (node_map_.find(id) != node_map_.end()) {
      return node_map_[id];
    } else {
      return nullptr;
    }
  }

  virtual Edge* GetEdgeByID(euler::common::EdgeID id) {
    if (edge_map_.find(id) != edge_map_.end()) {
      return edge_map_[id];
    } else {
      return nullptr;
    }
  }

  virtual bool AddNode(Node* n) {
    euler::common::NodeID node_id = n->GetID();
    node_map_insert_lock_.lock();
    node_map_[node_id] = n;
    node_map_insert_lock_.unlock();
    return true;
  }

  virtual bool AddNodeFrom(const std::unordered_map<euler::common::NodeID, Node*>& map) {
    node_map_insert_lock_.lock();
    node_map_.insert(map.begin(), map.end());
    node_map_insert_lock_.unlock();
    return true;
  }

  virtual bool AddNodeFrom(const std::vector<Node*>& vec) {
    node_map_insert_lock_.lock();
    for(auto &it:vec){
       node_map_.insert({it->GetID(), it});
    }
    node_map_insert_lock_.unlock();
    return true;
  }

  virtual bool AddEdge(Edge* e) {
    euler::common::EdgeID edge_id = e->GetID();
    edge_map_insert_lock_.lock();
    edge_map_[edge_id] = e;
    edge_map_insert_lock_.unlock();
    return true;
  }

  virtual bool AddEdgeFrom(const std::unordered_map<euler::common::EdgeID, Edge*,
                            euler::common::EdgeIDHashFunc, euler::common::EdgeIDEqualKey>& map) {
    edge_map_insert_lock_.lock();
    edge_map_.insert(map.begin(), map.end());
    edge_map_insert_lock_.unlock();
    return true;
  }

  virtual bool AddEdgeFrom(const std::vector<Edge*>& vec) {
    edge_map_insert_lock_.lock();
    for(auto &it:vec){
      edge_map_.insert({it->GetID(),it});
    }
    edge_map_insert_lock_.unlock();
    return true;
  }

  int64_t getNodeSize() {
      return node_map_.size();
  }

  int64_t getEdgeSize() {
      return edge_map_.size();
  }

  void reserveNodeMap(size_t size) {
    node_map_.reserve(size);
  }

  void reserveEdgeMap(size_t size){
    edge_map_.reserve(size);
  }
  virtual bool BuildGlobalSampler() = 0;

  virtual bool BuildGlobalEdgeSampler() = 0;

  void SetNodeTypeNum(int node_type_num) {
    node_map_insert_lock_.lock();
    node_type_num_ = node_type_num > node_type_num_ ?
        node_type_num : node_type_num_;
    node_map_insert_lock_.unlock();
  }

  int GetNodeTypeNum() {
    return node_type_num_;
  }

  void SetEdgeTypeNum(int edge_type_num) {
    edge_map_insert_lock_.lock();
    edge_type_num_ = edge_type_num > edge_type_num_ ?
        edge_type_num : edge_type_num_;
    edge_map_insert_lock_.unlock();
  }

  int GetEdgeTypeNum() {
    return edge_type_num_;
  }

  bool GlobalNodeSamplerOk() {
    return global_sampler_ok_;
  }

  bool GlobalEdgeSamplerOk() {
    return global_edge_sampler_ok_;
  }

  std::vector<float> GetNodeWeightSums() {
    if (global_sampler_ok_) {
      return node_weight_sums_;
    } else {
      LOG(WARNING) << "global sampler is not ok";
      return std::vector<float>();
    }
  }

  std::vector<float> GetEdgeWeightSums() {
    if (global_edge_sampler_ok_) {
      return edge_weight_sums_;
    } else {
      LOG(WARNING) << "global sampler is not ok";
      return std::vector<float>();
    }
  }

  void ShowGraph() {
    int32_t cnt = 0;
    std::unordered_map<euler::common::NodeID, Node*>::iterator it =
        node_map_.begin();
    while (it != node_map_.end()) {
      if (cnt % 500000 == 0) {
        std::cout << it->second->Serialize() << std::endl;
      }
      ++cnt;
      ++it;
    }
  }

 protected:
  std::mutex node_map_insert_lock_;

  std::mutex edge_map_insert_lock_;

  int node_type_num_;

  int edge_type_num_;

  std::vector<float> node_weight_sums_;

  std::vector<float> edge_weight_sums_;

  std::unordered_map<euler::common::NodeID, Node*> node_map_;

  std::unordered_map<euler::common::EdgeID, Edge*,
      euler::common::EdgeIDHashFunc, euler::common::EdgeIDEqualKey> edge_map_;

  bool global_sampler_ok_;

  bool global_edge_sampler_ok_;
};

}  // namespace core
}  // namespace euler

#endif  // EULER_CORE_GRAPH_H_
