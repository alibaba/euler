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

#ifndef EULER_PARSER_OPTIMIZE_RULE_H_
#define EULER_PARSER_OPTIMIZE_RULE_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

#include "euler/common/str_util.h"
#include "euler/core/dag_def/dag_def.h"
#include "euler/parser/optimize_type.h"

namespace euler {

struct DynamicSplit {
  int32_t pattern_node_id;
  void(*AddSplit)(const NodeDef& node_m,
                  std::vector<std::vector<std::string>>* split_op_info);
};

struct DynamicOutput {
  int32_t pattern_node_id;
  void(*AddOutput)(const NodeDef& node_m,
                   const std::vector<std::vector<std::string>>& split_op_info,
                   std::vector<std::vector<std::string>>* fusion_output_map,
                   std::vector<std::vector<std::string>>* merge_op_info);
};

enum OptimizeRuleType: int32_t {
  fusion_and_shard,
  unique_and_gather
};

struct OptimizeRule {
  OptimizerType opt_type_;

  OptimizeRuleType type_;

  /*
  adj_info:
  [
    "a:0 c:2",
    "b:1 c:2",
    "c:2 d:3 4:e"
    "d:3 f:5",
    "e:4 f:5",
    "f:5"
  ]
  */
  DAGDef sub_dag_;

  /* extra condition for match node key="node_name,pattern_node_id" */
  std::unordered_map<std::string, bool(*)(const NodeDef& node_m)>
  extra_cond_;  // optional

  OptimizeRule(OptimizerType opt_type,
               OptimizeRuleType type,
               const std::vector<std::string>& adj_info) {
    opt_type_ = opt_type;
    type_ = type;

    std::vector<std::vector<std::vector<std::string>>> adj_table;
    for (const std::string& row : adj_info) {
      std::vector<std::vector<std::string>> adj;
      std::vector<std::string> r = Split(row, ' ');
      for (const std::string& node : r) {
        std::vector<std::string> node_info = Split(node, ':');
        adj.push_back(node_info);
      }
      adj_table.push_back(adj);
    }

    std::unordered_set<int32_t> pre;
    std::unordered_set<int32_t> succ;
    for (size_t i = 0; i < adj_table.size(); ++i) {
      std::string node_type = adj_table[i][0][0];
      int32_t id = atoi(adj_table[i][0][1].c_str());
      sub_dag_.id_counter_ = id >= sub_dag_.id_counter_ ? id + 1 :
          sub_dag_.id_counter_;
      std::shared_ptr<NodeDef> node = std::make_shared<NodeDef>(node_type, id);
      sub_dag_.AddNodeDef(node, pre, succ);
    }
    for (size_t i = 0; i < adj_table.size(); ++i) {
      std::string node_type = adj_table[i][0][0];
      int32_t id = atoi(adj_table[i][0][1].c_str());
      std::shared_ptr<NodeDef> node = std::make_shared<NodeDef>(node_type, id);
      succ.clear();
      for (size_t j = 1; j < adj_table[i].size(); ++j) {
        succ.insert(atoi(adj_table[i][j][1].c_str()));
      }
      sub_dag_.AddNodeDef(node, pre, succ);
    }
  }

  virtual void ShowRule() {}
};

typedef void(*DynamicUniqueGather)(
    const NodeDef&, std::vector<std::vector<std::string>>*);

struct UniqueAndGatherRule: public OptimizeRule {
  bool dynamic_unique_;
  bool dynamic_gather_;
  /*
   unique op info
   [
    [
      unique_op_name,
      input_idx_list(input idx in pattern node input_edges, like "0,1")
    ]
   ]
   */
  std::vector<std::vector<std::string>> unique_op_info_;

  /*
   gather op info
   [
    [
      gather_op_name,
      unique_op(index in split_op_list_),
      input_idx_list(pattern node output idxs, like "0,1")
    ]
   ]
   */
  std::vector<std::vector<std::string>> gather_op_info_;

  /* optional */
  DynamicUniqueGather gen_unique_op_info_;

  /* optional */
  DynamicUniqueGather gen_gather_op_info_;

  void ShowRule() override {}

  UniqueAndGatherRule(
      const std::vector<std::string>& adj_info,
      const std::vector<std::vector<std::string>>& unique_op_info,
      const std::vector<std::vector<std::string>>& gather_op_info):
      OptimizeRule(local, unique_and_gather, adj_info) {
    if (adj_info.size() != 1) {
      EULER_LOG(FATAL) << "invalid unique gather rule!";
    }
    unique_op_info_ = unique_op_info;
    gather_op_info_ = gather_op_info;
    dynamic_unique_ = false;
    dynamic_gather_ = false;
  }
};

struct FusionAndShardRule: public OptimizeRule {
  bool dynamic_split_;
  bool dynamic_output_;
  std::string target_name_;

  /*
  fusion_output_map:
  [
    [
      inner_name,
      inner_id(pattern_node_id),
      inner_output_idx,
      fusion_output_idx,
    ]
  ]
  */
  std::vector<std::vector<std::string>> fusion_output_map_;

  /*
  split op info
  [
    [
      split_op_name,
      inputs_idx_list(input idx in fusion node input_edges_)
    ]
  ]
  */
  std::vector<std::vector<std::string>> split_op_info_;  // remote node attr

  /*
  merge op info: for each output in fusion_output_map
  [
    [
      merge_op_name,
      merge_idx_produce_op("split:0" or "merge:0",
          means merge idx is from split_op_info_[0] or merge_op_info_[0]),
      input_idx_list(fusion output idxs like "0,1")
    ]
  ]
  split_op could be -1, means this merge don't need merge idx
  */
  std::vector<std::vector<std::string>> merge_op_info_;  // remote node attr

  int32_t split_num_;  // remote node attr

  /* split info is produced by node's inputs */
  std::vector<DynamicSplit> dynamic_split_list_;  // optional

  /* output list is produced by node's attr */
  std::vector<DynamicOutput> dynamic_output_list_;  // optional

  /* fusion subset nodes in adj_info_ */
  std::vector<int32_t> fusion_nodes_;  // optional

  void ShowRule() override {
    std::cout << "------fusion_output_map------" << std::endl;
    for (const std::vector<std::string>& f : fusion_output_map_) {
      for (const std::string& ff : f) {
        std::cout << ff << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "--------split_op_info--------" << std::endl;
    for (const std::vector<std::string>& s : split_op_info_) {
      for (const std::string& ss : s) {
        std::cout << ss << " ";
      }
      std::cout << std::endl;
    }
    std::cout << "--------merge_op_info--------" << std::endl;
    for (const std::vector<std::string>& m : merge_op_info_) {
      for (const std::string& mm : m) {
        std::cout << mm << " ";
      }
      std::cout << std::endl;
    }
  }

  /* normal fusion rule */
  FusionAndShardRule(
      const std::vector<std::string>& adj_info,
      const std::string& target_name,
      const std::vector<std::vector<std::string>>& fusion_output_map):
      OptimizeRule(distribute, fusion_and_shard, adj_info) {
    target_name_ = target_name;
    fusion_output_map_ = fusion_output_map;
    dynamic_split_ = false;
    dynamic_output_ = false;
  }

  /* remote fusion shard rule */
  FusionAndShardRule(
      const std::vector<std::string>& adj_info,
      const std::string& target_name,
      const std::vector<std::vector<std::string>>& fusion_output_map,
      const std::vector<std::vector<std::string>>& split_op_info,
      const std::vector<std::vector<std::string>>& merge_op_info,
      int32_t split_num):
      FusionAndShardRule(adj_info, target_name, fusion_output_map) {
    split_op_info_ = split_op_info;
    merge_op_info_ = merge_op_info;
    split_num_ = split_num;
  }

  /* graph partition remote fusion shard rule */
  FusionAndShardRule(
      OptimizerType opt_type,
      const std::vector<std::string>& adj_info,
      const std::string& target_name,
      const std::vector<std::vector<std::string>>& fusion_output_map,
      const std::vector<std::vector<std::string>>& split_op_info,
      const std::vector<std::vector<std::string>>& merge_op_info,
      int32_t split_num):
      OptimizeRule(opt_type, fusion_and_shard, adj_info) {
    target_name_ = target_name;
    fusion_output_map_ = fusion_output_map;
    dynamic_split_ = false;
    dynamic_output_ = false;
    split_op_info_ = split_op_info;
    merge_op_info_ = merge_op_info;
    split_num_ = split_num;
  }
};
}  // namespace euler
#endif  // EULER_PARSER_OPTIMIZE_RULE_H_
