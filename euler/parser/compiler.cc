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

#include "euler/parser/compiler.h"

#include <string>
#include <vector>
#include <memory>

#include "euler/common/str_util.h"

namespace euler {

Compiler::Compiler(int32_t shard_num,
                   OptimizerType type, std::string index_info):
    shard_num_(shard_num), optimizer_(type, shard_num), translator_(type) {
  /* parse index info */
  std::vector<std::string> index_info_list = Split(index_info, ",");
  for (std::string name_type : index_info_list) {
    std::vector<std::string> name_type_info = Split(name_type, ":");
    index_info_[name_type_info[1]].push_back(name_type_info[0]);
    EULER_LOG(INFO) << name_type_info[1] << " : " << name_type_info[0];
  }

  /* unique -> API_GET_NB_NODE -> gather */
  {
    std::vector<std::string> adj_info = {"API_GET_NB_NODE:0"};
    std::vector<std::vector<std::string>> unique_op_info = {
        {"ID_UNIQUE", "0"}};
    std::vector<std::vector<std::string>> gather_op_info = {
        {"IDX_GATHER", "0", "0"},
        {"DATA_GATHER", "0", "1,0"},
        {"DATA_GATHER", "0", "2,0"},
        {"DATA_GATHER", "0", "3,0"}};
    std::shared_ptr<UniqueAndGatherRule> optmz_rule =
        std::make_shared<UniqueAndGatherRule>(
        adj_info, unique_op_info, gather_op_info);
    optimizer_.AddRule(optmz_rule);
  }

  /* unique -> API_GET_P -> gather */
  {
    std::vector<std::string> adj_info = {"API_GET_P:0"};
    std::vector<std::vector<std::string>> unique_op_info = {
        {"ID_UNIQUE", "0"}};
    std::vector<std::vector<std::string>> gather_op_info;
    std::shared_ptr<UniqueAndGatherRule> optmz_rule =
        std::make_shared<UniqueAndGatherRule>(
            adj_info, unique_op_info, gather_op_info);
    optmz_rule->dynamic_gather_ = true;
    optmz_rule->gen_gather_op_info_ =
        [](const NodeDef& node,
           std::vector<std::vector<std::string>>* gather_op_info) {
      for (size_t i = 0; i < node.attrs_.size(); ++i) {
        int32_t idx0 = i * 2, idx1 = i * 2 + 1;
        gather_op_info->push_back({"IDX_GATHER", "0", ToString(idx0)});
        gather_op_info->push_back(
            {"DATA_GATHER", "0", ToString(idx1, ",", idx0)});
      }
    };
    optimizer_.AddRule(optmz_rule);
  }

  /* unique -> API_SAMPLE_NB -> gather */
  {
    std::vector<std::string> adj_info = {"API_SAMPLE_NB:0"};
    std::vector<std::vector<std::string>> unique_op_info = {
        {"ID_UNIQUE", "0"}};
    std::vector<std::vector<std::string>> gather_op_info = {
        {"IDX_GATHER", "0", "0"},
        {"DATA_GATHER", "0", "1,0"},
        {"DATA_GATHER", "0", "2,0"},
        {"DATA_GATHER", "0", "3,0"}};
    std::shared_ptr<UniqueAndGatherRule> optmz_rule =
        std::make_shared<UniqueAndGatherRule>(
        adj_info, unique_op_info, gather_op_info);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_SAMPLE_NB, API_GET_P
   *                                          [ID_MERGE_0,
   *                                           DATA_MERGE_0,  // id
   * ID_SPLIT -> [REMOTE_0, REMOTE_1 ...] ->   DATA_MERGE_1,  // weight
   *                                           DATA_MERGE_2,  // type
   *                                           ID_MERGE, DATA_MERGE, ...]
   */
  {
    std::vector<std::string> adj_info =
        {"API_SAMPLE_NB:0 API_SAMPLE_NB:1 API_GET_P:2",
         "API_SAMPLE_NB:1", "API_GET_P:2"};
    std::vector<std::vector<std::string>>
        fusion_output_map;  // dynamically gen
    std::vector<std::vector<std::string>> split_op_info =
        {{"ID_SPLIT", "0"}, {"ID_SPLIT", "1"}};
    std::vector<std::vector<std::string>>
        merge_op_info;  // dynamically gen
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    std::static_pointer_cast<FusionAndShardRule>(optmz_rule)->
        fusion_nodes_ = {1, 2};
    std::static_pointer_cast<FusionAndShardRule>(optmz_rule)->
        dynamic_output_ = true;
    std::static_pointer_cast<FusionAndShardRule>(optmz_rule)->
        dynamic_output_list_.push_back(
        {
          1,  // API_SAMPLE_NB
          [](const NodeDef& node_m,
             const std::vector<std::vector<std::string>>& split_op_info,
             std::vector<std::vector<std::string>>* fusion_output_map,
             std::vector<std::vector<std::string>>* merge_op_info) {
            (void) node_m;
            (void) split_op_info;
            fusion_output_map->push_back({"API_SAMPLE_NB", "1", "0", "0"});
            fusion_output_map->push_back({"API_SAMPLE_NB", "1", "1", "1"});
            fusion_output_map->push_back({"API_SAMPLE_NB", "1", "2", "2"});
            fusion_output_map->push_back({"API_SAMPLE_NB", "1", "3", "3"});
            merge_op_info->push_back({"IDX_MERGE", "split:0", "0"});
            merge_op_info->push_back({"DATA_MERGE", "split:0", "1,0"});
            merge_op_info->push_back({"DATA_MERGE", "split:0", "2,0"});
            merge_op_info->push_back({"DATA_MERGE", "split:0", "3,0"});
          }
        });
    std::static_pointer_cast<FusionAndShardRule>(optmz_rule)->
        dynamic_output_list_.push_back(
        {
          2,  // API_GET_P
          [](const NodeDef& node_m,
             const std::vector<std::vector<std::string>>& split_op_info,
             std::vector<std::vector<std::string>>* fusion_output_map,
             std::vector<std::vector<std::string>>* merge_op_info) {
            (void) split_op_info;
            int32_t base_idx = fusion_output_map->size();
            for (size_t i = 0; i < node_m.attrs_.size(); ++i) {
              int32_t io0 = i * 2, io1 = i * 2 + 1;
              int32_t fo0 = base_idx + io0, fo1 = base_idx + io1;
              fusion_output_map->push_back(
                  {"API_GET_P", "2", ToString(io0), ToString(fo0)});
              fusion_output_map->push_back(
                  {"API_GET_P", "2", ToString(io1), ToString(fo1)});
              merge_op_info->push_back(
                  {"IDX_MERGE", "split:0", ToString(fo0)});
              merge_op_info->push_back(
                  {"DATA_MERGE", "split:0", ToString(fo1, ",", fo0)});
            }
          }
        });
    optimizer_.AddRule(optmz_rule);
  }

  /* API_SAMPLE_NODE:
   *
   * BROAD_CAST_SPLIT
   *                   -> [REMOTE_0, REMOTE_1 ...] -> APPEND_MERGE
   * SAMPLE_NODE_SPLIT
   *
   */
  {
    std::vector<std::string> adj_info = {"API_SAMPLE_NODE:0"};
    std::vector<std::vector<std::string>> fusion_output_map = {
        {"API_SAMPLE_NODE", "0", "0", "0"}};
    std::vector<std::vector<std::string>> split_op_info = {
        {"BROAD_CAST_SPLIT", "0"}, {"SAMPLE_NODE_SPLIT", "1,0"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"APPEND_MERGE", "split:1", "0"}};
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_SAMPLE_N_WITH_TYPES:
   *
   * BROAD_CAST_SPLIT                                   [MULTI_TYPE_IDX_MERGE,
   *                      -> [REMOTE_0, REMOTE_1 ...]->
   * SAMPLE_N_WITH_TYPES_SPLIT                           MULTI_TYPE_DATA_MERGE]
   *
   */
  {
    std::vector<std::string> adj_info = {"API_SAMPLE_N_WITH_TYPES:0"};
    std::vector<std::vector<std::string>> fusion_output_map = {
        {"API_SAMPLE_N_WITH_TYPES", "0", "0", "0"},
        {"API_SAMPLE_N_WITH_TYPES", "0", "1", "1"}};
    std::vector<std::vector<std::string>> split_op_info = {
        {"BROAD_CAST_SPLIT", "0"}, {"SAMPLE_N_WITH_TYPES_SPLIT", "1,0"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"MULTI_TYPE_IDX_MERGE", "split:-1", "0"},
        {"MULTI_TYPE_DATA_MERGE", "split:-1", "1,0"}};
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_SAMPLE_EDGE:
   *
   * BROAD_CAST_SPLIT
   *                   -> [REMOTE_0, REMOTE_1 ...] -> APPEND_MERGE
   * SAMPLE_EDGE_SPLIT
   *
   */
  {
    std::vector<std::string> adj_info = {"API_SAMPLE_EDGE:0"};
    std::vector<std::vector<std::string>> fusion_output_map = {
        {"API_SAMPLE_EDGE", "0", "0", "0"}};
    std::vector<std::vector<std::string>> split_op_info = {
        {"BROAD_CAST_SPLIT", "0"}, {"SAMPLE_EDGE_SPLIT", "1,0"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"APPEND_MERGE", "split:1", "0"}};
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_GET_NODE:
   *
   * ID_SPLIT -> [REMOTE_0, REMOTE_1 ...] -> APPEND_MERGE
   */
  {
    std::vector<std::string> adj_info = {"API_GET_NODE:0"};
    std::vector<std::vector<std::string>> fusion_output_map = {
        {"API_GET_NODE", "0", "0", "0"}};
    std::vector<std::vector<std::string>> split_op_info = {{"ID_SPLIT", "0"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"APPEND_MERGE", "split:-1", "0"}};
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    // extra cond
    std::static_pointer_cast<FusionAndShardRule>(optmz_rule)->
        extra_cond_["API_GET_NODE,0"] = [](const NodeDef& node_m){
      return node_m.input_edges_.size() > 0 &&
             node_m.attrs_.size() > 0;
    };
    optimizer_.AddRule(optmz_rule);
  }

  /* API_GET_NODE:
   *
   * [REMOTE_0, REMOTE_1 ...] -> APPEND_MERGE
   */
  {
    std::vector<std::string> adj_info = {"API_GET_NODE:0"};
    std::vector<std::vector<std::string>> fusion_output_map = {
        {"API_GET_NODE", "0", "0", "0"}};
    std::vector<std::vector<std::string>> split_op_info;
    std::vector<std::vector<std::string>> merge_op_info = {
        {"APPEND_MERGE", "split:-1", "0"}};
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    // extra cond
    std::static_pointer_cast<FusionAndShardRule>(optmz_rule)->
        extra_cond_["API_GET_NODE,0"] = [](const NodeDef& node_m){
      return node_m.input_edges_.size() == 0;
    };
    optimizer_.AddRule(optmz_rule);
  }

  /* API_GET_EDGE:
   *
   * ID_SPLIT -> [REMOTE_0, REMOTE_1 ...] -> APPEND_MERGE
   */
  {
    std::vector<std::string> adj_info = {"API_GET_EDGE:0"};
    std::vector<std::vector<std::string>> fusion_output_map = {
        {"API_GET_EDGE", "0", "0", "0"}};
    std::vector<std::vector<std::string>> split_op_info = {{"ID_SPLIT", "0"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"APPEND_MERGE", "split:-1", "0"}};
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    // extra cond
    std::static_pointer_cast<FusionAndShardRule>(optmz_rule)->
        extra_cond_["API_GET_EDGE,0"] = [](const NodeDef& node_m){
      return node_m.input_edges_.size() > 0 &&
             node_m.attrs_.size() > 0;
    };
    optimizer_.AddRule(optmz_rule);
  }

  /* API_GET_EDGE:
   *
   * [REMOTE_0, REMOTE_1 ...] -> APPEND_MERGE
   */
  {
    std::vector<std::string> adj_info = {"API_GET_EDGE:0"};
    std::vector<std::vector<std::string>> fusion_output_map = {
        {"API_GET_EDGE", "0", "0", "0"}};
    std::vector<std::vector<std::string>> split_op_info;
    std::vector<std::vector<std::string>> merge_op_info = {
        {"APPEND_MERGE", "split:-1", "0"}};
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    // extra cond
    std::static_pointer_cast<FusionAndShardRule>(optmz_rule)->
        extra_cond_["API_GET_EDGE,0"] = [](const NodeDef& node_m){
      return node_m.input_edges_.size() == 0;
    };
    optimizer_.AddRule(optmz_rule);
  }

  /* API_GET_NODE_T
   *
   * ID_SPLIT -> [REMOTE_0, REMOTE_1 ...] -> [REGULAR_DATA_MERGE]
   *
   */
  {
    std::vector<std::string> adj_info = {"API_GET_NODE_T:0"};
    std::vector<std::vector<std::string>> fusion_output_map {
      {"API_GET_NODE_T", "0", "0", "0"}};
    std::vector<std::vector<std::string>> split_op_info =
      {{"ID_SPLIT", "0"}};
    std::vector<std::vector<std::string>> merge_op_info =
      {{"REGULAR_DATA_MERGE", "split:0", "0"}};
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_GET_P:
   *
   * ID_SPLIT -> [REMOTE_0, REMOTE_1 ...] -> [ID_MERGE, DATA_MERGE, ...]
   *
   */
  {
    std::vector<std::string> adj_info = {"API_GET_P:0"};
    std::vector<std::vector<std::string>> fusion_output_map;  // dynamically gen
    std::vector<std::vector<std::string>> split_op_info = {{"ID_SPLIT", "0"}};
    std::vector<std::vector<std::string>> merge_op_info;  // dynamically gen
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    std::static_pointer_cast<FusionAndShardRule>(optmz_rule)->
        dynamic_output_ = true;
    std::static_pointer_cast<FusionAndShardRule>(optmz_rule)->
        dynamic_output_list_.push_back(
        {
          0,
          [](const NodeDef& node_m,
             const std::vector<std::vector<std::string>>& split_op_info,
             std::vector<std::vector<std::string>>* fusion_output_map,
             std::vector<std::vector<std::string>>* merge_op_info) {
            (void) split_op_info;
            for (size_t i = 0; i < node_m.attrs_.size(); ++i) {
              std::string idx = ToString(i * 2);
              fusion_output_map->push_back({"API_GET_P", "0", idx, idx});
              merge_op_info->push_back({"IDX_MERGE", "split:0",
                                       ToString(i * 2)});
              idx = ToString(i * 2 + 1);
              fusion_output_map->push_back({"API_GET_P", "0", idx, idx});
              merge_op_info->push_back(
                  {"DATA_MERGE", "split:0", ToString(i * 2 + 1, ",", i * 2)});
            }
          }
        });
    optimizer_.AddRule(optmz_rule);
  }

  /* API_SAMPLE_NB:
   *                                         [ID_MERGE_0,
   *                                          DATA_MERGE_0, // id
   * ID_SPLIT -> [REMOTE_0, REMOTE_1 ...] ->  DATA_MERGE_1, // weight
   *                                          DATA_MERGE_2] // type
   */
  {
    std::vector<std::string> adj_info = {"API_SAMPLE_NB:0"};
    std::vector<std::vector<std::string>> fusion_output_map = {
        {"API_SAMPLE_NB", "0", "0", "0"},
        {"API_SAMPLE_NB", "0", "1", "1"},  // id
        {"API_SAMPLE_NB", "0", "2", "2"},  // weight
        {"API_SAMPLE_NB", "0", "3", "3"}};  // type
    std::vector<std::vector<std::string>> split_op_info = {{"ID_SPLIT", "0"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"IDX_MERGE", "split:0", "0"},
        {"DATA_MERGE", "split:0", "1,0"},  // merge id
        {"DATA_MERGE", "split:0", "2,0"},  // merge weight
        {"DATA_MERGE", "split:0", "3,0"}};  // merge type
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_GET_NB_NODE:
   *                                         [ID_MERGE_0,
   *                                          DATA_MERGE_0, // id
   * ID_SPLIT -> [REMOTE_0, REMOTE_1 ...] ->  DATA_MERGE_1, // weight
   *                                          DATA_MERGE_2] // type
   */
  {
    std::vector<std::string> adj_info = {"API_GET_NB_NODE:0"};
    std::vector<std::vector<std::string>> fusion_output_map = {
        {"API_GET_NB_NODE", "0", "0", "0"},
        {"API_GET_NB_NODE", "0", "1", "1"},  // id
        {"API_GET_NB_NODE", "0", "2", "2"},  // weight
        {"API_GET_NB_NODE", "0", "3", "3"}};  // type
    std::vector<std::vector<std::string>> split_op_info = {{"ID_SPLIT", "0"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"IDX_MERGE", "split:0", "0"},
        {"DATA_MERGE", "split:0", "1,0"},  // merge id
        {"DATA_MERGE", "split:0", "2,0"},  // merge weight
        {"DATA_MERGE", "split:0", "3,0"}};  // merge type
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_GET_NB_EDGE:
   *                                         [ID_MERGE_0,
   * ID_SPLIT -> [REMOTE_0, REMOTE_1 ...] ->  DATA_MERGE_0, // id
   *                                          DATA_MERGE_1] // weight
   */
  {
    std::vector<std::string> adj_info = {"API_GET_NB_EDGE:0"};
    std::vector<std::vector<std::string>> fusion_output_map = {
        {"API_GET_NB_EDGE", "0", "0", "0"},
        {"API_GET_NB_EDGE", "0", "1", "1"},  // id
        {"API_GET_NB_EDGE", "0", "2", "2"}};  // weight
    std::vector<std::vector<std::string>> split_op_info = {{"ID_SPLIT", "0"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"IDX_MERGE", "split:0", "0"},
        {"DATA_MERGE", "split:0", "1,0"},  // merge id
        {"DATA_MERGE", "split:0", "2,0"}};  // merge weight
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_GET_EDGE_SUM_WEIGHT:
   *
   * ID_SPLIT -> [REMOTE_0, REMOTE_1 ...] -> [REGULAR_DATA_MERGE,
   *                                          REGULAR_DATA_MERGE]
   *
   */
  {
    std::vector<std::string> adj_info = {"API_GET_EDGE_SUM_WEIGHT:0"};
    std::vector<std::vector<std::string>> fusion_output_map {
        {"API_GET_EDGE_SUM_WEIGHT", "0", "0", "0"},
        {"API_GET_EDGE_SUM_WEIGHT", "0", "1", "1"}};
    std::vector<std::vector<std::string>> split_op_info = {{"ID_SPLIT", "0"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"REGULAR_DATA_MERGE", "split:0", "0"},
        {"REGULAR_DATA_MERGE", "split:0", "1"}};
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_SAMPLE_L:
   *                                         [REGULAR_DATA_MERGE,
   * ID_SPLIT -> [REMOTE_0, REMOTE_1 ...] ->  REGULAR_DATA_MERGE,
   *                                          REGULAR_DATA_MERGE]
   */
  {
    std::vector<std::string> adj_info = {"API_SAMPLE_L:0"};
    std::vector<std::vector<std::string>> fusion_output_map {
        {"API_SAMPLE_L", "0", "0", "0"},
        {"API_SAMPLE_L", "0", "1", "1"},
        {"API_SAMPLE_L", "0", "2", "2"}};
    std::vector<std::vector<std::string>> split_op_info = {{"ID_SPLIT", "0"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"REGULAR_DATA_MERGE", "split:0", "0"},
        {"REGULAR_DATA_MERGE", "split:0", "1"},
        {"REGULAR_DATA_MERGE", "split:0", "2"}};
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_GET_ADJ:
   *
   * ID_SPLIT -> [REMOTE_0, REMOTE_1 ...] -> [REGULAR_DATA_MERGE]
   */
  {
    std::vector<std::string> adj_info = {"API_GET_ADJ:0"};
    std::vector<std::vector<std::string>> fusion_output_map {
        {"API_GET_ADJ", "0", "0", "0"}};
    std::vector<std::vector<std::string>> split_op_info =
        {{"ID_SPLIT", "0"}};
    std::vector<std::vector<std::string>> merge_op_info =
        {{"REGULAR_DATA_MERGE", "split:0", "0"}};
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_SPARSE_GET_ADJ:
   *
   * ID_SPLIT                                        [ID_MERGE,
   *                  -> [REMOTE_0, REMOTE_1 ...] ->  DATA_MERGE
   * BROAD_CAST_SPLIT                                ]
   */
  {
    std::vector<std::string> adj_info = {"API_SPARSE_GET_ADJ:0"};
    std::vector<std::vector<std::string>> fusion_output_map {
        {"API_SPARSE_GET_ADJ", "0", "0", "0"},
        {"API_SPARSE_GET_ADJ", "0", "1", "1"}};
    std::vector<std::vector<std::string>> split_op_info =
        {{"ID_SPLIT", "0"}, {"BROAD_CAST_SPLIT", "1"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"IDX_MERGE", "split:0", "0"},
        {"DATA_MERGE", "split:0", "1,0"}};  // merge id
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }

  /* API_GET_GRAPH_BY_LABEL:
   *                                                 [IDX_ROW_APPEND_MERGE,
   * BROAD_CAST_SPLIT -> [REMOTE_0, REMOTE_1 ...] ->  DATA_ROW_APPEND_MERGE]
   *
   */
  {
    std::vector<std::string> adj_info = {"API_GET_GRAPH_BY_LABEL:0"};
    std::vector<std::vector<std::string>> fusion_output_map {
        {"API_GET_GRAPH_BY_LABEL", "0", "0", "0"},
        {"API_GET_GRAPH_BY_LABEL", "0", "1", "1"}};
    std::vector<std::vector<std::string>> split_op_info =
        {{"BROAD_CAST_SPLIT", "0"}};
    std::vector<std::vector<std::string>> merge_op_info = {
        {"IDX_ROW_APPEND_MERGE", "split:-1", "0"},
        {"DATA_ROW_APPEND_MERGE", "split:-1", "1,0"}};  // merge id
    std::shared_ptr<OptimizeRule> optmz_rule =
        std::make_shared<FusionAndShardRule>(
        adj_info, "REMOTE", fusion_output_map,
        split_op_info, merge_op_info, shard_num);
    optimizer_.AddRule(optmz_rule);
  }
}

Compiler* Compiler::instance_ = nullptr;

}  // namespace euler
