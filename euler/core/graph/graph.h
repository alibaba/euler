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

#ifndef EULER_CORE_GRAPH_GRAPH_H_
#define EULER_CORE_GRAPH_GRAPH_H_

#include <stdint.h>

#include <vector>
#include <unordered_map>
#include <iostream>
#include <string>
#include <utility>
#include <memory>

#include "euler/common/logging.h"
#include "euler/common/status.h"
#include "euler/common/data_types.h"
#include "euler/common/file_io.h"
#include "euler/common/server_monitor.h"
#include "euler/core/graph/node.h"
#include "euler/core/graph/edge.h"
#include "euler/core/graph/graph_meta.h"
#include "euler/common/fast_weighted_collection.h"
#include "euler/common/compact_weighted_collection.h"

namespace euler {

class Graph {
 public:
  Graph()
      : global_sampler_ok_(0),
        global_edge_sampler_ok_(0),
        initialized_(false),
        shard_index_(0),
        shard_number_(0) {
  }

  ~Graph();

  Status Init(int shard_index, int shard_number,
              const std::string& sampler_type,
              const std::string& data_path,
              const std::string& load_data_type);

  std::vector<Meta> GetRegisterInfo();

  Status DeregisterRemote(const std::string& host_port,
                          const std::string& zk_server,
                          const std::string& zk_path);

  static Graph& Instance() {
    static Graph instance;
    return instance;
  }

  typedef uint64_t UID;

  UID EdgeIdToUID(const euler::common::EdgeID& eid);

  euler::common::EdgeID UIDToEdgeId(UID uid);

  std::vector<euler::common::NodeID>
  SampleNode(int node_type, int count) const;

  std::vector<euler::common::NodeID>
  SampleNode(const std::vector<int>& node_types, int count) const;

  std::vector<euler::common::EdgeID>
  SampleEdge(int edge_type, int count) const;

  std::vector<euler::common::EdgeID>
  SampleEdge(const std::vector<int>& edge_types, int count) const;

  Node* GetNodeByID(euler::common::NodeID id) const {
    auto it = node_map_.find(id);
    if (it != node_map_.end()) {
      return it->second;
    }
    return nullptr;
  }

  Edge* GetEdgeByID(euler::common::EdgeID id) const {
    if (edge_map_.size() == 0) {
      EULER_LOG(FATAL) << "Edge Map size is 0. "
                       << "Edges need to be loaded before use GetEdgeByID.";
    }

    auto it = edge_map_.find(id);
    if (it != edge_map_.end()) {
      return it->second;
    }

    return nullptr;
  }

  bool AddNode(Node* n);

  bool AddNodeFrom(const std::unordered_map<euler::common::NodeID, Node*>& map);

  bool AddNodeFrom(const std::vector<Node*>& vec);

  bool AddEdge(Edge* e);

  bool AddEdgeFrom(const std::unordered_map<
                   euler::common::EdgeID,
                   Edge*,
                   euler::common::EdgeIDHashFunc,
                   euler::common::EdgeIDEqualKey>& map);

  bool AddEdgeFrom(const std::vector<Edge*>& vec);

  int64_t getNodeSize();

  int64_t getEdgeSize();

  void reserveNodeMap(size_t size);

  void reserveEdgeMap(size_t size);

  bool BuildGlobalSampler();

  bool BuildGlobalEdgeSampler();

  size_t GetEdgeTypeNum();

  size_t GetNodeTypeNum();

  std::vector<std::string> GetGraphLabel();

  bool GlobalNodeSamplerOk();

  bool GlobalEdgeSamplerOk();

  std::vector<float> GetNodeWeightSums();

  std::vector<float> GetEdgeWeightSums();

  void ShowGraph();

  bool Dump(euler::FileIO* file_io) const;

  FeatureType GetNodeFeatureType(const std::string &feature_name) const;

  int32_t GetNodeFeatureId(const std::string &feature_name) const;

  FeatureType GetEdgeFeatureType(const std::string &feature_name) const;

  int32_t GetEdgeFeatureId(const std::string &feature_name) const;

  void set_meta(const GraphMeta& meta) { meta_ = meta; }

  const GraphMeta& graph_meta() const { return meta_; }

  bool GetNodeTypeByName(const std::string& name, int* type_id) {
    auto it = meta_.node_type_map_.find(name);
    if (it == meta_.node_type_map_.end()) {
      return false;
    }

    *type_id = it->second;
    return true;
  }

  bool GetEdgeTypeByName(const std::string& name, int* type_id) {
    auto it = meta_.edge_type_map_.find(name);
    if (it == meta_.edge_type_map_.end()) {
      return false;
    }

    *type_id = it->second;
    return true;
  }

 protected:
  std::vector<float> node_weight_sums_;
  std::vector<float> edge_weight_sums_;
  std::unordered_map<euler::common::NodeID, Node*> node_map_;
  std::unordered_map<euler::common::EdgeID, Edge*,
      euler::common::EdgeIDHashFunc, euler::common::EdgeIDEqualKey> edge_map_;
  std::unordered_map<UID, euler::common::EdgeID> edge_id_map_;
  euler::common::EdgeIDHashFunc eid_hash_;
  bool global_sampler_ok_;
  bool global_edge_sampler_ok_;

 private:
  bool initialized_;
  int shard_index_;
  int shard_number_;
  euler::GraphMeta meta_;
  euler::common::FastWeightedCollection<int32_t> node_type_collection_;
  euler::common::FastWeightedCollection<int32_t> edge_type_collection_;
  std::vector<euler::common::FastWeightedCollection<euler::common::NodeID>>
      node_samplers_;
  std::vector<euler::common::FastWeightedCollection<euler::common::EdgeID>>
      edge_samplers_;
};

}  // namespace euler

#endif  // EULER_CORE_GRAPH_GRAPH_H_
