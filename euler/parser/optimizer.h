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

#ifndef EULER_PARSER_OPTIMIZER_H_
#define EULER_PARSER_OPTIMIZER_H_

#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>
#include <string>

#include "euler/parser/optimize_type.h"
#include "euler/parser/optimize_rule.h"
#include "euler/core/dag_def/dag_def.h"
namespace euler {

class Optimizer {
 public:
  explicit Optimizer(OptimizerType type, int32_t shard_num) {
    type_ = type;
    shard_num_ = shard_num;
    local_only_ops_ = {
        "AS", "REMOTE",
        "API_GET_NB_FILTER",
        "POST_PROCESS",
        "BROAD_CAST_SPLIT",
        "SAMPLE_NODE_SPLIT",
        "SAMPLE_EDGE_SPLIT",
        "GP_BROAD_CAST_SPLIT",
        "GP_APPEND_MERGE",
        "GP_UNIQUE_MERGE",
        "GP_IDX_MERGE",
        "GP_DATA_MERGE",
        "GP_REGULAR_DATA_MERGE"};

    /* key=op_name:input
     * value=split_op:total_inputs*/
    graph_part_mode_split_map_ = {
        {"API_SAMPLE_NODE:0", "BROAD_CAST_SPLIT:0"},
        {"API_SAMPLE_NODE:1", "SAMPLE_NODE_SPLIT:1,0"},
        {"API_SAMPLE_EDGE:0", "BROAD_CAST_SPLIT:0"},
        {"API_SAMPLE_EDGE:1", "SAMPLE_EDGE_SPLIT:1,0"},
        {"API_GET_NODE:0", "GP_BROAD_CAST_SPLIT:0"},
        {"API_GET_EDGE:0", "GP_BROAD_CAST_SPLIT:0"},
        {"API_SAMPLE_NB:0", "GP_BROAD_CAST_SPLIT:0"},
        {"API_GET_NB_NODE:0", "GP_BROAD_CAST_SPLIT:0"},
        {"API_GET_NODE_T:0", "GP_BROAD_CAST_SPLIT:0"},
        {"API_GET_P:0", "GP_BROAD_CAST_SPLIT:0"}};

    /* key=op_name:output_idx
     *
     * value=
     * merge_op_name:
     * merge_info_relate_input_idx(key_input):
     * input_idx_list(output idxs like "0,1")*/
    graph_part_mode_merge_map_ = {
        {"API_SAMPLE_NODE:0", "GP_APPEND_MERGE:1:0"},
        {"API_SAMPLE_EDGE:0", "GP_APPEND_MERGE:1:0"},
        {"API_GET_NODE:0", "GP_UNIQUE_MERGE:-1:0"},
        {"API_GET_EDGE:0", "GP_UNIQUE_MERGE:-1:0"},
        // GP_IDX_MERGE need to consider merge_idx duplicate problem
        {"API_SAMPLE_NB:0", "GP_IDX_MERGE:0:0"},
        // GP_DATA_MERGE need to consider merge_idx duplicate problem
        {"API_SAMPLE_NB:1", "GP_DATA_MERGE:0:1,0"},
        {"API_SAMPLE_NB:2", "GP_DATA_MERGE:0:2,0"},
        {"API_SAMPLE_NB:3", "GP_DATA_MERGE:0:3,0"},
        {"API_GET_NB_NODE:0", "GP_IDX_MERGE:0:0"},
        {"API_GET_NB_NODE:1", "GP_DATA_MERGE:0:1,0"},
        {"API_GET_NB_NODE:2", "GP_DATA_MERGE:0:2,0"},
        {"API_GET_NB_NODE:3", "GP_DATA_MERGE:0:3,0"},
        {"API_GET_NODE_T:0", "GP_REGULAR_DATA_MERGE:0:0"},
        {"API_GET_P:even", "GP_IDX_MERGE:0:even"},
        {"API_GET_P:odd", "GP_DATA_MERGE:0:odd,even"}};
  }

  void AddRule(std::shared_ptr<OptimizeRule> rule) {
    rules_.push_back(rule);
  }

  /* for graph partition mode */
  std::shared_ptr<OptimizeRule> ProduceRule(DAGDef* dag);

  bool Optimize(DAGDef* dag);

 private:
  std::vector<std::shared_ptr<OptimizeRule>> rules_;

  OptimizerType type_;

  int32_t shard_num_;

  std::unordered_set<std::string> local_only_ops_;

  std::unordered_map<std::string, std::string>
      graph_part_mode_split_map_;

  std::unordered_map<std::string, std::string>
      graph_part_mode_merge_map_;

  bool FusionAndShard(
      std::shared_ptr<FusionAndShardRule> rule, DAGDef* dag);

  bool UniqueAndGather(
      std::shared_ptr<UniqueAndGatherRule> rule, DAGDef* dag);

  void CommonSubexpressionElimination(DAGDef* dag);

  std::vector<int32_t> ProduceFusionAdj(
      DAGDef* dag, std::vector<std::string>* adj);

  std::vector<std::vector<std::string>> ProduceFusionOutputMap(
      const std::vector<int32_t>& subset, DAGDef* dag);

  std::vector<std::vector<std::string>> ProduceSplitOpInfo(
      const std::vector<int32_t>& subset, DAGDef* dag,
      std::unordered_map<std::string, int32_t>* op_key_input2split_info_idx);

  std::vector<std::string> GetMergeOpInfo(
      const std::string& op_name, int32_t output_idx,
      int32_t pre_op_output_num);

  std::vector<std::vector<std::string>> ProduceMergeOpInfo(
      const std::vector<int32_t>& subset,
      const std::unordered_map<std::string, int32_t>&
      op_key_input2split_info_idx, DAGDef* dag);
};
}  // namespace euler
#endif  // EULER_PARSER_OPTIMIZER_H_
