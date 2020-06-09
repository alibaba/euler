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


#ifndef EULER_CORE_DAG_DEF_DAG_DEF_H_
#define EULER_CORE_DAG_DEF_DAG_DEF_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <vector>

#include "euler/core/dag_def/dag_node_def.h"
#include "euler/core/framework/dag.pb.h"
#include "euler/core/framework/attr_value.h"

namespace euler {

struct FusionRule {
  std::string fusion_name_;
  std::vector<FusionOutput> fusion_output_map_;
  std::vector<int32_t> fusion_nodes_;
  FusionRule() {}
  FusionRule(
      const std::string& fusion_name,
      const std::unordered_map<int32_t, int32_t>& pattern,
      const std::vector<std::vector<std::string>>& fusion_infos,
      const std::vector<int32_t>& fusion_nodes);
};

struct ShardRule {
  std::vector<SplitOpInfo> split_op_info_list_;
  std::vector<MergeOpInfo> merge_op_info_list_;
  int32_t split_num_;
  ShardRule() {}
  ShardRule(const std::vector<std::vector<std::string>>& split_infos,
            const std::vector<std::vector<std::string>>& merge_infos,
            int32_t split_num);
};

struct UniqueGatherRule {
  std::vector<UniqueOpInfo> unique_op_info_list_;
  std::vector<GatherOpInfo> gather_op_info_list_;
  UniqueGatherRule(
      const std::vector<std::vector<std::string>>& unique_op_infos,
      const std::vector<std::vector<std::string>>& gather_op_infos);
};

class DAGDef {
 public:
  DAGDef() {
    id_counter_ = 0;
  }

  std::shared_ptr<NodeDef> ProduceNodeDef(const std::string& name,
                                          int32_t output_num) {
    return std::make_shared<NodeDef>(name, GetIdCounter(), output_num);
  }

  bool AddNodeDef(std::shared_ptr<NodeDef> node,
                  const std::unordered_set<int32_t>& pre,
                  const std::unordered_set<int32_t>& succ);

  int32_t FusionNodes(const std::unordered_set<int32_t>& nodes,
                      const FusionRule& fusion_rule);

  bool ShardRemoteNodeDef(int32_t remote_node_id,
                          const ShardRule& shard_rule);

  bool UniqueAndGatherNode(int32_t pattern_node_id,
                           const UniqueGatherRule& unique_gather_rule);

  std::shared_ptr<NodeDef> GetNodeById(int32_t id) const {
    if (node_map_.find(id) == node_map_.end()) return nullptr;
    return node_map_.at(id);
  }

  void EraseNodeById(int32_t id) {
    node_map_.erase(id);
  }

  std::unordered_map<int32_t, std::shared_ptr<NodeDef>> GetNodeMap() const {
    return node_map_;
  }

  void CutRelation(const std::unordered_set<int32_t>& nodes,
                   const std::unordered_set<int32_t>& in_nodes,
                   const std::unordered_set<int32_t>& out_nodes);

  std::vector<int32_t> TopologicSort(const std::unordered_set<int32_t>& nodes);

  std::vector<int32_t> TopologicSort();

  void GetInAndOutNode(const std::unordered_set<int32_t>& sub_nodes,
                       std::unordered_set<int32_t>* in_nodes,
                       std::unordered_set<int32_t>* out_nodes);

  std::unordered_set<int32_t> MacroFusionProcess(
      const std::unordered_set<std::string>& local_only_ops);

  DAGProto ToProto();

 private:
  int32_t GetIdCounter() {
    return id_counter_++;
  }

  bool FusionAvailable(
      const std::unordered_set<int32_t>& nodes,
      const std::unordered_set<int32_t>& in_nodes,
      const std::unordered_set<int32_t>& out_nodes,
      const std::unordered_map<std::string, int32_t>& fusion_outputs);

  void InsertSplitNode(
      const std::vector<SplitOpInfo>& split_op_info_list,
      int32_t shard_num, const RemoteNodeDef& remote_node_prototype,
      std::vector<std::shared_ptr<NodeDef>>* split_nodes);

  void InsertRemoteNode(
      int32_t shard_num, const RemoteNodeDef& remote_node_prototype,
      const std::vector<std::shared_ptr<NodeDef>>& split_nodes,
      std::vector<std::shared_ptr<RemoteNodeDef>>* remote_nodes);

  void InsertMergeNode(
      const std::vector<MergeOpInfo>& merge_op_info_list,
      const std::vector<std::shared_ptr<NodeDef>>& split_nodes,
      const std::vector<std::shared_ptr<RemoteNodeDef>>& remote_nodes,
      const std::unordered_set<int32_t>& out_nodes);

  void InsertUniqueNode(
      const std::vector<UniqueOpInfo>& unique_op_info_list,
      const NodeDef& pattern_node,
      std::vector<std::shared_ptr<NodeDef>>* unique_nodes);

  void InsertPatternNode(
      const std::vector<UniqueOpInfo>& unique_op_info_list,
      const std::vector<std::shared_ptr<NodeDef>>& unique_nodes,
      std::shared_ptr<NodeDef> pattern_node);

  void InsertGatherNode(
      const std::vector<GatherOpInfo>& gather_op_info_list,
      const std::vector<std::shared_ptr<NodeDef>>& unique_nodes,
      const NodeDef& pattern_node,
      const std::unordered_set<int32_t>& out_nodes);

  std::unordered_map<int32_t, std::shared_ptr<NodeDef>> node_map_;

  int32_t id_counter_;  // dag node inner id counter

  friend class OptimizeRule;
};

}  // namespace euler

#endif  // EULER_CORE_DAG_DEF_DAG_DEF_H_
