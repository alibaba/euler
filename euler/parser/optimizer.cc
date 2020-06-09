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

#include "euler/parser/optimizer.h"

#include <stdlib.h>
#include <unordered_set>
#include <string>
#include <vector>

#include "euler/core/dag_def/dag_def.h"
#include "euler/core/dag_def/sub_graph_iso.h"
#include "euler/common/str_util.h"

namespace euler {

namespace {
void PrepareFusionShardRule(
    bool dynamic_split,
    bool dynamic_output,
    const DAGDef& dag,
    const std::unordered_map<int32_t, int32_t>& pattern,
    std::shared_ptr<FusionAndShardRule> rule,
    std::unordered_set<int32_t>* fusion_set) {
  // prepare split_op_info_list
  if (dynamic_split) {
    rule->split_op_info_.clear();
    for (const DynamicSplit& d_s : rule->dynamic_split_list_) {
      int32_t node_id = pattern.at(d_s.pattern_node_id);
      std::shared_ptr<NodeDef> node_m = dag.GetNodeById(node_id);
      d_s.AddSplit(*node_m, &rule->split_op_info_);
    }
  }
  // prepare fusion_output_map and merge op info
  if (dynamic_output) {  // dynamic add output into fusion_output_map
    rule->fusion_output_map_.clear();
    rule->merge_op_info_.clear();
    for (const DynamicOutput& d_o : rule->dynamic_output_list_) {
      int32_t node_id = pattern.at(d_o.pattern_node_id);
      std::shared_ptr<NodeDef> node_m = dag.GetNodeById(node_id);
      d_o.AddOutput(*node_m, rule->split_op_info_,
                    &rule->fusion_output_map_, &rule->merge_op_info_);
    }
  }
  // prepare fusion set
  if (rule->fusion_nodes_.empty()) {
    for (std::unordered_map<int32_t, int32_t>::const_iterator it =
         pattern.begin(); it != pattern.end(); ++it) {
      fusion_set->insert(it->second);
    }
  } else {
    for (int32_t p_id : rule->fusion_nodes_) {
      fusion_set->insert(pattern.at(p_id));
    }
  }
}

void PrepareUniqueGatherRule(
    bool dynamic_unique,
    bool dynamic_gather,
    const DAGDef& dag,
    int32_t node_id,
    std::shared_ptr<UniqueAndGatherRule> rule) {
  if (dynamic_unique) {
    rule->unique_op_info_.clear();
    rule->gen_unique_op_info_(
        *(dag.GetNodeById(node_id)), &rule->unique_op_info_);
  }
  if (dynamic_gather) {
    rule->gather_op_info_.clear();
    rule->gen_gather_op_info_(
        *(dag.GetNodeById(node_id)), &rule->gather_op_info_);
  }
}

}  // namespace

bool Optimizer::FusionAndShard(
    std::shared_ptr<FusionAndShardRule> rule, DAGDef* dag) {
  std::string target_name = rule->target_name_;
  bool dynamic_split = rule->dynamic_split_;
  bool dynamic_output = rule->dynamic_output_;
  bool fusion_success = true;
  bool one_shot = true;
  while (fusion_success) {
    /* find pattern */
    std::vector<std::unordered_map<int32_t, int32_t>> patterns;
    if (rule->opt_type_ == graph_partition && one_shot) {
      // build pattern directly
      std::unordered_map<int32_t, std::shared_ptr<NodeDef>> sub_dag =
          rule->sub_dag_.GetNodeMap();
      std::unordered_map<int32_t, int32_t> pattern;
      for (auto it = sub_dag.begin(); it != sub_dag.end(); ++it) {
        pattern[it->first] = it->first;
      }
      patterns.push_back(pattern);
      one_shot = false;
    } else if (rule->opt_type_ != graph_partition) {
      patterns = SubGraphMatch(*dag, rule->sub_dag_, rule->extra_cond_);
    }

    fusion_success = false;
    /* try to fusion and shard */
    for (const std::unordered_map<int32_t, int32_t>& pattern : patterns) {
      std::unordered_set<int32_t> fusion_set;
      PrepareFusionShardRule(
          dynamic_split, dynamic_output, *dag, pattern, rule, &fusion_set);
      FusionRule fusion_rule(
          target_name, pattern, rule->fusion_output_map_, rule->fusion_nodes_);
      // fusion
      int32_t fusion_node_id = dag->FusionNodes(
          fusion_set, fusion_rule);
      fusion_success = fusion_success || fusion_node_id != -1;
      // shard
      if (fusion_success) {
        if (target_name == "REMOTE") {  // need to be shard
          ShardRule shard_rule(rule->split_op_info_,
                               rule->merge_op_info_,
                               rule->split_num_);
          if (!dag->ShardRemoteNodeDef(fusion_node_id, shard_rule)) {
            return false;
          }
        }
        break;
      }
    }
  }
  return true;
}

bool Optimizer::UniqueAndGather(
    std::shared_ptr<UniqueAndGatherRule> rule, DAGDef* dag) {
  std::vector<std::unordered_map<int32_t, int32_t>> patterns =
      SubGraphMatch(*dag, rule->sub_dag_, rule->extra_cond_);
  for (const std::unordered_map<int32_t, int32_t>& pattern : patterns) {
    int32_t node_id = pattern.begin()->second;
    PrepareUniqueGatherRule(rule->dynamic_unique_, rule->dynamic_gather_,
                            *dag, node_id, rule);
    UniqueGatherRule ugr(rule->unique_op_info_, rule->gather_op_info_);
    dag->UniqueAndGatherNode(node_id, ugr);
  }
  return true;
}

std::string BuildInputKey(const std::string& node_name,
                          const std::vector<EdgeDef>& input_edges) {
  std::string key = node_name;
  for (const EdgeDef& e : input_edges) {
    std::string e_str = ToString(e.src_id_, "_", e.src_slot_);
    key = ToString(key, ",", e_str);
  }
  return key;
}

void Optimizer::CommonSubexpressionElimination(DAGDef* dag) {
  std::unordered_set<std::string> op_name_set({"ID_SPLIT", "ID_UNIQUE"});
  std::unordered_map<std::string, int32_t> node_map;
  std::vector<int32_t> toposort_list = dag->TopologicSort();
  for (int32_t node_id : toposort_list) {
    std::shared_ptr<NodeDef> node = dag->GetNodeById(node_id);
    if (op_name_set.find(node->name_) != op_name_set.end()) {
      std::string input_key = BuildInputKey(node->name_, node->input_edges_);
      if (node_map.find(input_key) != node_map.end()) {
        // CommonSubexpression
        // cut relation
        std::unordered_set<int32_t> nodes({node_id});
        std::unordered_set<int32_t> in_nodes = node->pre_;
        std::unordered_set<int32_t> out_nodes = node->succ_;
        dag->CutRelation(nodes, in_nodes, out_nodes);
        // change out nodes inputs and pre dependencies
        int32_t keep_node_id = node_map.at(input_key);
        std::shared_ptr<NodeDef> keep_node = dag->GetNodeById(keep_node_id);
        for (int32_t succ_id : out_nodes) {
          std::shared_ptr<NodeDef> succ_node = dag->GetNodeById(succ_id);
          succ_node->pre_.insert(keep_node_id);
          keep_node->succ_.insert(succ_id);
          for (EdgeDef& input_edge : succ_node->input_edges_) {
            if (input_edge.src_id_ == node_id) {
              input_edge.src_id_ = keep_node_id;
            }
          }
        }
        dag->EraseNodeById(node_id);
      } else {
        node_map[input_key] = node_id;
      }
    }
  }
}

bool Optimizer::Optimize(DAGDef* dag) {
  if (type_ == graph_partition) {
    std::shared_ptr<OptimizeRule> rule = ProduceRule(dag);
    while (rule != nullptr) {
      if (!FusionAndShard(
              std::static_pointer_cast<FusionAndShardRule>(rule), dag)) {
        return false;
      }
      rule = ProduceRule(dag);
    }
  } else {
    for (std::shared_ptr<OptimizeRule> rule : rules_) {
      if (!compatible_matrix[type_][rule->opt_type_]) continue;
      if (rule->type_ == fusion_and_shard) {
        if (!FusionAndShard(
                std::static_pointer_cast<FusionAndShardRule>(rule), dag)) {
          return false;
        }
      } else if (rule->type_ == unique_and_gather) {
        if (!UniqueAndGather(
                std::static_pointer_cast<UniqueAndGatherRule>(rule), dag)) {
          return false;
        }
      } else {
        EULER_LOG(INFO) << "not support rule type";
      }
    }
    CommonSubexpressionElimination(dag);
  }
  return true;
}

std::vector<int32_t> Optimizer::ProduceFusionAdj(
    DAGDef* dag, std::vector<std::string>* adj) {
  std::unordered_set<int32_t> ids = dag->MacroFusionProcess(local_only_ops_);
  adj->reserve(ids.size());
  for (int32_t id : ids) {
    std::shared_ptr<NodeDef> node_def = dag->GetNodeById(id);
    std::vector<std::string> row = {ToString(node_def->name_, ":", id)};
    for (int32_t succ_id : node_def->succ_) {
      if (ids.find(succ_id) != ids.end()) {
        std::shared_ptr<NodeDef> succ_node_def = dag->GetNodeById(succ_id);
        row.push_back(ToString(succ_node_def->name_, ":", succ_id));
      }
    }
    adj->push_back(Join(row, " "));
  }
  return dag->TopologicSort(ids);
}

std::vector<std::vector<std::string>>
Optimizer::ProduceFusionOutputMap(
    const std::vector<int32_t>& subset, DAGDef* dag) {
  std::vector<std::vector<std::string>> results;
  int32_t counter = 0;
  for (int32_t id : subset) {
    std::shared_ptr<NodeDef> node_def = dag->GetNodeById(id);
    std::string op_name = node_def->name_;
    std::string op_id = ToString(node_def->id_);
    for (int32_t i = 0; i < node_def->output_num_; ++i) {
      std::vector<std::string> fusion_output_info =
          {op_name, op_id, ToString(i), ToString(counter++)};
      results.push_back(fusion_output_info);
    }
  }
  return results;
}

std::vector<std::vector<std::string>>
Optimizer::ProduceSplitOpInfo(
    const std::vector<int32_t>& subset, DAGDef* dag,
    std::unordered_map<std::string, int32_t>* op_key_input2split_info_idx) {
  std::unordered_map<std::string, int32_t> op_inputidx2fusion_input_idx;
  std::unordered_set<int32_t> set(subset.begin(), subset.end());
  std::vector<std::vector<std::string>> results;
  int32_t fusion_input_cnt = 0;
  for (int32_t id : subset) {
    std::shared_ptr<NodeDef> node_def = dag->GetNodeById(id);
    std::string op_name = node_def->name_;
    int32_t node_id = node_def->id_;
    int32_t input_idx = 0;
    for (EdgeDef& input : node_def->input_edges_) {
      if (set.find(input.src_id_) == set.end()) {  // outside input
        op_inputidx2fusion_input_idx[
            ToString(op_name, ",", node_id, ":", input_idx)] =
            fusion_input_cnt++;
      }
      ++input_idx;
    }
  }
  for (int32_t id : subset) {
    std::shared_ptr<NodeDef> node_def = dag->GetNodeById(id);
    std::string op_name = node_def->name_;
    int32_t node_id = node_def->id_;
    int32_t input_idx = 0;
    for (EdgeDef& input : node_def->input_edges_) {
      if (set.find(input.src_id_) == set.end()) {  // outside input
        std::string split_op_info =
            graph_part_mode_split_map_[ToString(op_name, ":", input_idx)];
        if (split_op_info.empty()) {
          EULER_LOG(FATAL) << op_name << ":" << input_idx << " split op error";
        }
        std::vector<std::string> split_op_and_inputs =
            Split(split_op_info, ':');
        std::string split_op_name = split_op_and_inputs[0];
        std::vector<std::string> inputs =
            Split(split_op_and_inputs[1], ',');
        std::vector<std::string> fusion_inputs;
        for (std::string& input : inputs) {
          fusion_inputs.push_back(
              ToString(op_inputidx2fusion_input_idx[
                       ToString(op_name, ",", node_id, ":", input)]));
        }
        results.push_back({split_op_name, Join(fusion_inputs, ",")});
        (*op_key_input2split_info_idx)[
            ToString(op_name, ",", node_id, ":", input_idx)] =
            results.size() - 1;
      }
      ++input_idx;
    }
  }
  return results;
}

std::vector<std::string> Optimizer::GetMergeOpInfo(
    const std::string& op_name, int32_t output_idx,
    int32_t pre_op_output_num) {
  std::string key;
  if (op_name == "API_GET_P") {
    if (output_idx % 2 == 0) {
      key = ToString(op_name, ":even");
    } else {
      key = ToString(op_name, ":odd");
    }
  } else {
    key = ToString(op_name, ":", output_idx);
  }
  std::string value = graph_part_mode_merge_map_[key];
  if (value.empty()) {
    EULER_LOG(FATAL) << op_name << ":" << output_idx << " merge op error";
  }
  std::vector<std::string> result = Split(value, ':');
  std::vector<std::string> input_list = Split(result[2], ',');
  std::vector<std::string> input_id_list;
  input_id_list.reserve(input_list.size());
  for (const std::string& input : input_list) {
    if (input == "even" || input == "odd") {
      input_id_list.push_back(ToString(pre_op_output_num + output_idx));
    } else {
      input_id_list.push_back(ToString(pre_op_output_num +
                                       atoi(input.c_str())));
    }
  }
  result[2] = Join(input_id_list, ",");
  return result;
}

std::vector<std::vector<std::string>>
Optimizer::ProduceMergeOpInfo(
    const std::vector<int32_t>& subset,
    const std::unordered_map<std::string, int32_t>&
    op_key_input2split_info_idx,  // <op_name,id:input_idx  split_info_idx>
    DAGDef* dag) {
  // Toposort subset
  //   每个op的每个输出，根据graph_part_mode_merge_map_生成一个merge op
  //   同时获得这个merge op对应的key_input
  //   if (key_input != -1 // 这个merge op需要merge index) {
  //     if (key_input是外部输入) {
  //       根据op_key_input2split_info_idx生成merge op info
  //       插入 merge_op_info
  //     } else {
  //       根据input的src_name,src_id:src_slot得到merge_info_idx
  //       插入 merge_op_info
  //     }
  //     并记录op_output2merge_info_idx
  //   }

  /*<op_name,id:outputidx merge_info_idx>*/
  std::unordered_map<std::string, int32_t> op_output2merge_info_idx;

  std::unordered_set<int32_t> set(subset.begin(), subset.end());
  std::vector<std::vector<std::string>> results;
  int32_t pre_op_output_num = 0;
  for (int32_t id : subset) {
    std::shared_ptr<NodeDef> node_def = dag->GetNodeById(id);
    std::string op_name = node_def->name_;
    int32_t node_id = node_def->id_;
    for (int32_t output = 0; output < node_def->output_num_;
         ++output) {
      std::vector<std::string> info =
          GetMergeOpInfo(op_name, output, pre_op_output_num);
      std::string merge_op_name = info[0];
      std::string key_input = info[1];
      std::string merge_op_input_list = info[2];
      if (key_input != "-1") {
        int32_t key_input_idx = atoi(key_input.c_str());
        if (key_input_idx > static_cast<int32_t>(
                node_def->input_edges_.size())) {
          EULER_LOG(FATAL) << "key input overflow: " << key_input_idx;
        }
        EdgeDef input_edge = node_def->input_edges_[key_input_idx];
        if (set.find(input_edge.src_id_) == set.end()) {
          // merge info come from split op
          int32_t split_info_idx = op_key_input2split_info_idx
              .at(ToString(op_name, ",", node_id, ":", key_input_idx));
          results.push_back(
              {merge_op_name, ToString("split:", split_info_idx), info[2]});
        } else {  // merge info come from pre merge op
          int32_t merge_info_idx = op_output2merge_info_idx
              .at(ToString(input_edge.src_name_, ",",
                           input_edge.src_id_, ":", input_edge.src_slot_));
          results.push_back(
              {merge_op_name, ToString("merge:", merge_info_idx), info[2]});
        }
      } else {
        results.push_back(
            {merge_op_name, "split:-1", info[2]});
      }
      op_output2merge_info_idx[
          ToString(op_name, ",", node_id, ":", output)] =
          results.size() - 1;
    }
    pre_op_output_num += node_def->output_num_;
  }
  return results;
}

std::shared_ptr<OptimizeRule> Optimizer::ProduceRule(
    DAGDef* dag) {
  std::vector<std::string> adj_info;
  std::vector<int32_t> subset = ProduceFusionAdj(dag, &adj_info);
  if (subset.empty()) return nullptr;

  std::vector<std::vector<std::string>> fusion_output_map =
      ProduceFusionOutputMap(subset, dag);

  std::unordered_map<std::string, int32_t>
      op_key_input2split_info_idx;
  std::vector<std::vector<std::string>> split_op_info =
      ProduceSplitOpInfo(subset, dag,
                         &op_key_input2split_info_idx);

  std::vector<std::vector<std::string>> merge_op_info =
      ProduceMergeOpInfo(
          subset, op_key_input2split_info_idx, dag);

  std::shared_ptr<OptimizeRule> optmz_rule =
      std::make_shared<FusionAndShardRule>(
      graph_partition, adj_info, "REMOTE", fusion_output_map,
      split_op_info, merge_op_info, shard_num_);
  return optmz_rule;
}

}  // namespace euler
