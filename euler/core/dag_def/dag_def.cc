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

#include "euler/core/dag_def/dag_def.h"

#include <stdlib.h>
#include <queue>
#include <utility>

#include "euler/common/str_util.h"

namespace euler {

FusionRule::FusionRule(
    const std::string& fusion_name,
    const std::unordered_map<int32_t, int32_t>& pattern,
    const std::vector<std::vector<std::string>>& fusion_infos,
    const std::vector<int32_t>& fusion_nodes) {
  fusion_name_ = fusion_name;
  for (const std::vector<std::string>& v : fusion_infos) {
    fusion_output_map_.push_back(FusionOutput(v, pattern));
  }
  for (int32_t nid : fusion_nodes) {
    fusion_nodes_.push_back(pattern.at(nid));
  }
}

ShardRule::ShardRule(const std::vector<std::vector<std::string>>& split_infos,
                     const std::vector<std::vector<std::string>>& merge_infos,
                     int32_t split_num) {
  for (const std::vector<std::string>& split_op_info : split_infos) {
    split_op_info_list_.push_back(SplitOpInfo(split_op_info));
  }
  for (const std::vector<std::string>& merge_op_info : merge_infos) {
    merge_op_info_list_.push_back(MergeOpInfo(merge_op_info));
  }
  split_num_ = split_num;
}

UniqueGatherRule::UniqueGatherRule(
    const std::vector<std::vector<std::string>>& unique_op_infos,
    const std::vector<std::vector<std::string>>& gather_op_infos) {
  for (const std::vector<std::string>& unique_op_info : unique_op_infos) {
    unique_op_info_list_.push_back(UniqueOpInfo(unique_op_info));
  }
  for (const std::vector<std::string>& gather_op_info : gather_op_infos) {
    gather_op_info_list_.push_back(GatherOpInfo(gather_op_info));
  }
}

bool DAGDef::AddNodeDef(std::shared_ptr<NodeDef> node,
                        const std::unordered_set<int32_t>& pre,
                        const std::unordered_set<int32_t>& succ) {
  // process pre
  for (int32_t id : pre) {
    if (node_map_.find(id) == node_map_.end()) {
      return false;
    }
    node_map_[id]->succ_.insert(node->id_);
    node->pre_.insert(id);
  }
  // process succ
  for (int32_t id : succ) {
    if (node_map_.find(id) == node_map_.end()) {
      return false;
    }
    node_map_[id]->pre_.insert(node->id_);
    node->succ_.insert(id);
  }
  // add node
  if (node_map_.find(node->id_) != node_map_.end()) {
    // merge node
    for (int32_t p : node_map_[node->id_]->pre_) {
      node->pre_.insert(p);
    }
    for (int32_t s : node_map_[node->id_]->succ_) {
      node->succ_.insert(s);
    }
  }
  node_map_[node->id_] = node;
  return true;
}

DAGProto DAGDef::ToProto() {
  DAGProto dag_proto;
  for (const std::pair<int32_t, std::shared_ptr<NodeDef>>& pair : node_map_) {
    DAGNodeProto* node_proto = dag_proto.add_nodes();
    pair.second->ToProto(node_proto);
  }
  return dag_proto;
}

bool DAGDef::FusionAvailable(
    const std::unordered_set<int32_t>& nodes,
    const std::unordered_set<int32_t>& in_nodes,
    const std::unordered_set<int32_t>& out_nodes,
    const std::unordered_map<std::string, int32_t>& fusion_outputs) {
  for (int32_t o_id : out_nodes) {
    if (in_nodes.find(o_id) != in_nodes.end()) {
      return false;
    }
    // fusioned nodes's outputs must in the fusion_output_map
    for (const EdgeDef& input_edge : node_map_[o_id]->input_edges_) {
      if (nodes.find(input_edge.src_id_) != nodes.end()) {
        std::string input_edge_str = ToString(input_edge.src_id_, ",",
                                              input_edge.src_slot_);
        if (fusion_outputs.find(input_edge_str) == fusion_outputs.end()) {
          return false;
        }
      }
    }
  }
  return true;
}

int32_t DAGDef::FusionNodes(
    const std::unordered_set<int32_t>& nodes,
    const FusionRule& fusion_rule) {

  const std::string& fusion_name = fusion_rule.fusion_name_;
  const std::vector<FusionOutput>&
  fusion_output_map = fusion_rule.fusion_output_map_;

  if (nodes.empty()) return -1;
  std::unordered_set<int32_t> in_nodes, out_nodes;
  GetInAndOutNode(nodes, &in_nodes, &out_nodes);

  std::unordered_map<std::string, int32_t> fusion_outputs;
  for (const FusionOutput& fusion_output : fusion_output_map) {
    std::string key = ToString(fusion_output.inner_id_, ",",
                               fusion_output.inner_output_idx_);
    fusion_outputs[key] = fusion_output.fusion_output_idx_;
  }

  /* check */
  if (!FusionAvailable(nodes, in_nodes, out_nodes, fusion_outputs)) return -1;

  /* cut edges */
  CutRelation(nodes, in_nodes, out_nodes);

  /* fusion nodes */
  int32_t new_id = GetIdCounter();
  // update out nodes's input_edges
  for (int32_t o_id : out_nodes) {
    for (size_t i = 0; i < node_map_[o_id]->input_edges_.size(); ++i) {
      int32_t src_id = node_map_[o_id]->input_edges_[i].src_id_;
      int32_t src_slot = node_map_[o_id]->input_edges_[i].src_slot_;
      if (nodes.find(src_id) != nodes.end()) {
        std::string key = ToString(src_id, ",", src_slot);
        node_map_[o_id]->input_edges_[i].src_name_ = fusion_name;
        node_map_[o_id]->input_edges_[i].src_id_ = new_id;
        node_map_[o_id]->input_edges_[i].src_slot_ = fusion_outputs.at(key);
      }
    }
  }
  std::shared_ptr<NodeDef> new_node;
  if (fusion_name == "REMOTE") {  // fusion to remote node
    std::vector<std::shared_ptr<NodeDef>> node_set;
    for (int32_t id : nodes) {
      node_set.push_back(node_map_[id]);
    }
    new_node = std::make_shared<RemoteNodeDef>(
        fusion_name, new_id, 0, node_set,
        fusion_output_map, fusion_output_map.size());
  } else {  // fusion to normal node
    new_node = std::make_shared<NodeDef>(
        fusion_name, new_id, fusion_output_map.size());
  }
  // update fusion node's input_edges_
  const std::vector<int32_t> sorted_nodes = fusion_rule.fusion_nodes_.empty() ?
      TopologicSort(nodes) : fusion_rule.fusion_nodes_;
  for (int32_t id : sorted_nodes) {
    std::shared_ptr<NodeDef> node = node_map_[id];
    for (EdgeDef& e : node->input_edges_) {
      // only keep outside inputs
      if (nodes.find(e.src_id_) == nodes.end()) {
        new_node->input_edges_.push_back(e);
        // build inner node inputs to remote node inputs map
        if (fusion_name == "REMOTE") {
          e.src_name_ = "REMOTE";
          e.src_id_ = new_node->id_;  // useless field
          e.src_slot_ = new_node->input_edges_.size() - 1;
        }
      }
    }
    // delete fusioned nodes
    node_map_.erase(id);
  }
  AddNodeDef(new_node, in_nodes, out_nodes);
  return new_id;
}

void DAGDef::InsertSplitNode(
    const std::vector<SplitOpInfo>& split_op_info_list,
    int32_t shard_num, const RemoteNodeDef& remote_node_prototype,
    std::vector<std::shared_ptr<NodeDef>>* split_nodes) {
  /* create */
  for (const SplitOpInfo& split_op_info : split_op_info_list) {
    split_nodes->push_back(
        ProduceNodeDef(split_op_info.split_op_name_, shard_num * 2));
  }
  /* add inputs */
  for (size_t i = 0; i < split_op_info_list.size(); ++i) {
    for (int32_t j : split_op_info_list[i].inputs_idx_) {
      (*split_nodes)[i]->input_edges_.push_back(
          remote_node_prototype.input_edges_[j]);
    }
  }
  /* insert into dag */
  std::unordered_set<int32_t> pre, succ;
  for (size_t i = 0; i < split_op_info_list.size(); ++i) {
    pre.clear();
    for (EdgeDef input_edge : (*split_nodes)[i]->input_edges_) {
      if (input_edge.src_id_ != -1) pre.insert(input_edge.src_id_);
    }
    AddNodeDef((*split_nodes)[i], pre, succ);
  }
}

void DAGDef::InsertRemoteNode(
    int32_t shard_num, const RemoteNodeDef& remote_node_prototype,
    const std::vector<std::shared_ptr<NodeDef>>& split_nodes,
    std::vector<std::shared_ptr<RemoteNodeDef>>* remote_nodes) {
  /* create */
  for (int32_t i = 0; i < shard_num; ++i) {
    std::shared_ptr<RemoteNodeDef> remote_node =
        std::make_shared<RemoteNodeDef>(remote_node_prototype);
    // keep fusion_output_map_ and inner_node input_edges_
    remote_node->id_ = GetIdCounter();
    remote_node->pre_.clear();
    remote_node->succ_.clear();
    remote_node->input_edges_.clear();
    remote_node->shard_idx_ = i;

    remote_nodes->push_back(remote_node);
  }
  /* add inputs */
  for (int32_t i = 0; i < static_cast<int32_t>(remote_nodes->size()); ++i) {
    for (size_t j = 0; j < remote_node_prototype.input_edges_.size(); ++j) {
      (*remote_nodes)[i]->input_edges_.push_back(
          {split_nodes[j]->name_, split_nodes[j]->id_, 2 * i});
    }
  }
  /* insert into dag */
  std::unordered_set<int32_t> pre, succ;
  for (size_t i = 0; i < split_nodes.size(); ++i) {
    pre.insert(split_nodes[i]->id_);
  }
  for (size_t i = 0; i < remote_nodes->size(); ++i) {
    AddNodeDef((*remote_nodes)[i], pre, succ);
  }
}

void DAGDef::InsertMergeNode(
    const std::vector<MergeOpInfo>& merge_op_info_list,
    const std::vector<std::shared_ptr<NodeDef>>& split_nodes,
    const std::vector<std::shared_ptr<RemoteNodeDef>>& remote_nodes,
    const std::unordered_set<int32_t>& out_nodes) {
  std::vector<std::shared_ptr<NodeDef>> merge_nodes;
  /* create */
  for (const MergeOpInfo& merge_op_info : merge_op_info_list) {
    merge_nodes.push_back(ProduceNodeDef(merge_op_info.merge_op_name_,
                                         1 + remote_nodes.size()));
  }
  /* add inputs */
  for (size_t i = 0; i < merge_op_info_list.size(); ++i) {
    std::string merge_idx_node_name = "";
    int32_t merge_idx_node_id = -1;
    if (merge_op_info_list[i].merge_idx_op_idx_ != -1) {
      if (merge_op_info_list[i].merge_idx_op_type_ == "split") {
        std::shared_ptr<NodeDef> split_node =
            split_nodes[merge_op_info_list[i].merge_idx_op_idx_];
        merge_idx_node_name = split_node->name_;
        merge_idx_node_id = split_node->id_;
      } else {
        std::shared_ptr<NodeDef> merge_node =
            merge_nodes[merge_op_info_list[i].merge_idx_op_idx_];
        merge_idx_node_name = merge_node->name_;
        merge_idx_node_id = merge_node->id_;
      }
    }
    // inputs
    for (size_t j = 0; j < remote_nodes.size(); ++j) {
      // data input
      for (int32_t input_idx : merge_op_info_list[i].inputs_idx_) {
        merge_nodes[i]->input_edges_.push_back({
            remote_nodes[j]->name_, remote_nodes[j]->id_, input_idx});
      }
      int32_t merge_idx_slot = 0;
      if (merge_op_info_list[i].merge_idx_op_type_ == "split") {
        merge_idx_slot = merge_idx_node_id == -1 ? -1 : j * 2 + 1;
      } else {
        merge_idx_slot = merge_idx_node_id == -1 ? -1 : j + 1;
      }
      merge_nodes[i]->input_edges_.push_back(
          {merge_idx_node_name, merge_idx_node_id, merge_idx_slot});
    }
  }
  for (int32_t merge_succ_id : out_nodes) {
    for (EdgeDef& input_edge : node_map_[merge_succ_id]->input_edges_) {
      if (input_edge.src_name_ == "REMOTE") {
        int32_t origin_remote_output_idx = input_edge.src_slot_;
        std::shared_ptr<NodeDef> input_node =
            merge_nodes[origin_remote_output_idx];
        input_edge.src_name_ = input_node->name_;
        input_edge.src_id_ = input_node->id_;
        input_edge.src_slot_ = 0;
      }
    }
  }
  /* insert into dag */
  std::unordered_set<int32_t> pre, succ;
  for (size_t i = 0; i < merge_nodes.size(); ++i) {
    pre.clear();
    succ.clear();
    for (const EdgeDef& merge_input : merge_nodes[i]->input_edges_) {
      if (merge_input.src_id_ != -1) pre.insert(merge_input.src_id_);
    }
    int32_t merge_node_id = merge_nodes[i]->id_;
    for (int32_t o_id : out_nodes) {
      for (const EdgeDef& input_edge : node_map_[o_id]->input_edges_) {
        if (input_edge.src_id_ == merge_node_id) {
          succ.insert(o_id);
        }
      }
    }
    AddNodeDef(merge_nodes[i], pre, succ);
  }
}

bool DAGDef::ShardRemoteNodeDef(int32_t remote_node_id,
                                const ShardRule& shard_rule) {
  const std::vector<SplitOpInfo>&
  split_op_info_list = shard_rule.split_op_info_list_;
  const std::vector<MergeOpInfo>&
  merge_op_info_list = shard_rule.merge_op_info_list_;
  int32_t split_num = shard_rule.split_num_;

  if (node_map_.find(remote_node_id) == node_map_.end()) {
    return false;
  }
  std::shared_ptr<RemoteNodeDef> origin_remote_node =
      std::static_pointer_cast<RemoteNodeDef>(node_map_[remote_node_id]);
  if (origin_remote_node->fusion_output_map_.size() !=
      merge_op_info_list.size()) {
    EULER_LOG(ERROR) << "fusion_output_map size != merge_op_info_list size";
    return false;
  }
  if (origin_remote_node->input_edges_.size() != split_op_info_list.size()) {
    EULER_LOG(ERROR) << "input_edges size != split_op_info_list size";
    EULER_LOG(INFO) << "input edges-------";
    for (const EdgeDef& e : origin_remote_node->input_edges_) {
      EULER_LOG(INFO) << e.src_name_ << "," << e.src_id_ << "," << e.src_slot_;
    }
    EULER_LOG(INFO) << "split op info-------";
    for (const SplitOpInfo& s : split_op_info_list) {
      EULER_LOG(INFO) << s.split_op_name_;
    }
    EULER_LOG(INFO) << "inner node--------";
    for (std::shared_ptr<NodeDef> node : origin_remote_node->nodes_) {
      EULER_LOG(INFO) << node->name_ << "," << node->id_;
    }
    return false;
  }

  /* remove origin remote node */
  std::unordered_set<int32_t> in_nodes = origin_remote_node->pre_;
  std::unordered_set<int32_t> out_nodes = origin_remote_node->succ_;
  std::unordered_set<int32_t> remote_node_set({remote_node_id});
  CutRelation(remote_node_set, in_nodes, out_nodes);
  node_map_.erase(remote_node_id);

  /* build data dependencies
   * split nodes:  input = origin remote node's input
   *               outputs = (split_input, merge_idx) * split_num
   * remote nodes: inputs = split nodes's split_input
   *               outputs = origin remote node's output
   *               remote_outputs: an output_2_remote_output map, is built in
   *               ToProto phase
   * merge nodes:  inputs: from remote nodes's output and split nodes's
   *               merge_idx
   *               outputs: merged data
   */

  std::vector<std::shared_ptr<NodeDef>> split_nodes;
  std::vector<std::shared_ptr<RemoteNodeDef>> remote_nodes;
  std::vector<std::shared_ptr<NodeDef>> merge_nodes;

  InsertSplitNode(split_op_info_list, split_num,
                  *origin_remote_node, &split_nodes);
  InsertRemoteNode(split_num, *origin_remote_node, split_nodes, &remote_nodes);
  InsertMergeNode(merge_op_info_list, split_nodes, remote_nodes, out_nodes);

  return true;
}

void DAGDef::InsertUniqueNode(
    const std::vector<UniqueOpInfo>& unique_op_info_list,
    const NodeDef& pattern_node,
    std::vector<std::shared_ptr<NodeDef>>* unique_nodes) {
  /* create */
  for (const UniqueOpInfo& unique_op_info : unique_op_info_list) {
    unique_nodes->push_back(
        ProduceNodeDef(unique_op_info.unique_op_name_, 2));
  }
  /* add input */
  for (size_t i = 0; i < unique_op_info_list.size(); ++i) {
    for (int32_t input_idx : unique_op_info_list[i].inputs_idx_) {
      (*unique_nodes)[i]->input_edges_.push_back(
          pattern_node.input_edges_[input_idx]);
    }
  }
  /* insert into dag */
  std::unordered_set<int32_t> pre, succ;
  for (size_t i = 0; i < unique_op_info_list.size(); ++i) {
    pre.clear();
    for (EdgeDef input_edge : (*unique_nodes)[i]->input_edges_) {
      if (input_edge.src_id_ != -1) pre.insert(input_edge.src_id_);
    }
    AddNodeDef((*unique_nodes)[i], pre, succ);
  }
}

void DAGDef::InsertPatternNode(
    const std::vector<UniqueOpInfo>& unique_op_info_list,
    const std::vector<std::shared_ptr<NodeDef>>& unique_nodes,
    std::shared_ptr<NodeDef> pattern_node) {
  /* update input */
  std::unordered_map<int32_t, int32_t> input2unique;
  for (size_t i = 0; i < unique_op_info_list.size(); ++i) {
    int32_t input_idx = unique_op_info_list[i].inputs_idx_[0];
    input2unique[input_idx] = i;
  }
  for (size_t i = 0; i < pattern_node->input_edges_.size(); ++i) {
    if (input2unique.find(i) != input2unique.end()) {
      std::shared_ptr<NodeDef> unique_node = unique_nodes[input2unique.at(i)];
      pattern_node->input_edges_[i].src_name_ = unique_node->name_;
      pattern_node->input_edges_[i].src_id_ = unique_node->id_;
      pattern_node->input_edges_[i].src_slot_ = 0;
    }
  }
  /* insert into dag */
  std::unordered_set<int32_t> pre, succ;
  for (EdgeDef input_edge : pattern_node->input_edges_) {
    if (input_edge.src_id_ != -1) pre.insert(input_edge.src_id_);
  }
  AddNodeDef(pattern_node, pre, succ);
}

void DAGDef::InsertGatherNode(
    const std::vector<GatherOpInfo>& gather_op_info_list,
    const std::vector<std::shared_ptr<NodeDef>>& unique_nodes,
    const NodeDef& pattern_node,
    const std::unordered_set<int32_t>& out_nodes) {
  std::vector<std::shared_ptr<NodeDef>> gather_nodes;
  /* create */
  std::unordered_map<int32_t, int32_t> po2gi;  // pattern output 2 gather index
  int32_t cnt = 0;
  for (const GatherOpInfo& gather_op_info : gather_op_info_list) {
    gather_nodes.push_back(
        ProduceNodeDef(gather_op_info.gather_op_name_, 1));
    po2gi[gather_op_info.inputs_idx_[0]] = cnt++;
  }
  /* add input */
  for (size_t i = 0; i < gather_op_info_list.size(); ++i) {
    for (int32_t input_idx : gather_op_info_list[i].inputs_idx_) {
      gather_nodes[i]->input_edges_.push_back(
          {pattern_node.name_, pattern_node.id_, input_idx});  // raw results
    }
    int32_t unique_op_idx = gather_op_info_list[i].unique_op_idx_;
    std::shared_ptr<NodeDef> unique_node = unique_nodes[unique_op_idx];
    gather_nodes[i]->input_edges_.push_back(
        {unique_node->name_, unique_node->id_, 1});  // gather idx
  }
  for (int32_t i : out_nodes) {
    for (EdgeDef& input_edge : node_map_[i]->input_edges_) {
      if (input_edge.src_name_ == pattern_node.name_ &&
          po2gi.find(input_edge.src_slot_) != po2gi.end()) {
        int32_t gather_idx = po2gi.at(input_edge.src_slot_);
        std::shared_ptr<NodeDef> gather_node = gather_nodes[gather_idx];
        input_edge.src_name_ = gather_node->name_;
        input_edge.src_id_ = gather_node->id_;
        input_edge.src_slot_ = 0;  // gather result
      }
    }
  }
  /* insert into dag */
  std::unordered_set<int32_t> pre, succ;
  for (size_t i = 0; i < gather_nodes.size(); ++i) {
    pre.clear();
    succ.clear();
    for (const EdgeDef& gather_input : gather_nodes[i]->input_edges_) {
      if (gather_input.src_id_ != -1) pre.insert(gather_input.src_id_);
    }
    int32_t gather_node_id = gather_nodes[i]->id_;
    for (int32_t o_id : out_nodes) {
      for (const EdgeDef& input_edge : node_map_[o_id]->input_edges_) {
        if (input_edge.src_id_ == gather_node_id) {
          succ.insert(o_id);
        }
      }
    }
    AddNodeDef(gather_nodes[i], pre, succ);
  }
}

bool DAGDef::UniqueAndGatherNode(int32_t pattern_node_id,
                                 const UniqueGatherRule& unique_gather_rule) {
  const std::vector<UniqueOpInfo>&
  unique_op_info_list = unique_gather_rule.unique_op_info_list_;
  const std::vector<GatherOpInfo>&
  gather_op_info_list = unique_gather_rule.gather_op_info_list_;

  /* remove pattern node */
  std::shared_ptr<NodeDef> pattern_node = node_map_.at(pattern_node_id);
  std::unordered_set<int32_t> in_nodes = pattern_node->pre_;
  std::unordered_set<int32_t> out_nodes = pattern_node->succ_;
  std::unordered_set<int32_t> pattern_node_set({pattern_node_id});
  CutRelation(pattern_node_set, in_nodes, out_nodes);
  node_map_.erase(pattern_node_id);

  /* insert */
  std::vector<std::shared_ptr<NodeDef>> unique_nodes;
  InsertUniqueNode(unique_op_info_list, *pattern_node, &unique_nodes);
  InsertPatternNode(unique_op_info_list, unique_nodes, pattern_node);
  InsertGatherNode(gather_op_info_list, unique_nodes, *pattern_node, out_nodes);
  return true;
}

void DAGDef::CutRelation(const std::unordered_set<int32_t>& nodes,
                         const std::unordered_set<int32_t>& in_nodes,
                         const std::unordered_set<int32_t>& out_nodes) {
  for (int32_t i_id : in_nodes) {
    std::shared_ptr<NodeDef> in_node = node_map_[i_id];
    for (auto succ_it = in_node->succ_.begin();
         succ_it != in_node->succ_.end();) {
      if (nodes.find(*succ_it) != nodes.end()) {
        node_map_[*succ_it]->pre_.erase(i_id);
        succ_it = in_node->succ_.erase(succ_it);
      } else {
        ++succ_it;
      }
    }
  }
  for (int32_t o_id : out_nodes) {
    std::shared_ptr<NodeDef> out_node = node_map_[o_id];
    for (auto pre_it = out_node->pre_.begin();
         pre_it != out_node->pre_.end();) {
      if (nodes.find(*pre_it) != nodes.end()) {
        node_map_[*pre_it]->succ_.erase(o_id);
        pre_it = out_node->pre_.erase(pre_it);
      } else {
        ++pre_it;
      }
    }
  }
}

std::vector<int32_t> DAGDef::TopologicSort(
    const std::unordered_set<int32_t>& nodes) {
  std::vector<int32_t> result;
  // build adj table and degree map
  std::unordered_map<int32_t, int32_t> degree_map;
  std::unordered_map<int32_t, std::unordered_set<int32_t>> adj_table;
  for (int32_t node_id : nodes) {
    degree_map[node_id] = 0;
  }
  for (int32_t node_id : nodes) {
    std::shared_ptr<NodeDef> node = node_map_[node_id];
    std::unordered_set<int32_t> out;
    for (int32_t succ_id : node->succ_) {
      if (nodes.find(succ_id) != nodes.end()) {
        out.insert(succ_id);
        degree_map[succ_id] += 1;
      }
    }
    adj_table[node_id] = out;
  }
  // topo sort
  std::queue<int32_t> q;
  for (const std::pair<int32_t, int32_t>& p : degree_map) {
    if (p.second == 0) {
      q.push(p.first);
    }
  }
  while (!q.empty()) {
    int32_t node_id = q.front();
    result.push_back(node_id);
    q.pop();
    for (int32_t succ_id : adj_table[node_id]) {
      if (--degree_map[succ_id] == 0) {
        q.push(succ_id);
      }
    }
  }
  return result;
}

std::vector<int32_t> DAGDef::TopologicSort() {
  std::unordered_set<int32_t> nodes;
  for (auto it = node_map_.begin(); it != node_map_.end(); ++it) {
    nodes.insert(it->first);
  }
  return TopologicSort(nodes);
}

void DAGDef::GetInAndOutNode(const std::unordered_set<int32_t>& sub_nodes,
                     std::unordered_set<int32_t>* in_nodes,
                     std::unordered_set<int32_t>* out_nodes) {
  for (int32_t n : sub_nodes) {
    for (int32_t succ : node_map_[n]->succ_) {
      if (sub_nodes.find(succ) == sub_nodes.end()) {
        out_nodes->insert(succ);
      }
    }
    for (int32_t pre : node_map_[n]->pre_) {
      if (sub_nodes.find(pre) == sub_nodes.end()) {
        in_nodes->insert(pre);
      }
    }
  }
}

std::unordered_set<int32_t> DAGDef::MacroFusionProcess(
    const std::unordered_set<std::string>& local_only_ops) {
  std::vector<int32_t> sort_nodes = TopologicSort();
  std::unordered_set<int32_t> results(sort_nodes.size());
  for (int32_t node_id : sort_nodes) {
    if (local_only_ops.find(node_map_[node_id]->name_) ==
        local_only_ops.end()) {
      results.insert(node_id);
      std::unordered_set<int32_t> in_nodes, out_nodes;
      GetInAndOutNode(results, &in_nodes, &out_nodes);
      bool dag_preserve = true;
      for (int32_t o_id : out_nodes) {
        if (in_nodes.find(o_id) != in_nodes.end()) {
          dag_preserve = false;
        }
      }
      if (!dag_preserve) {
        results.erase(node_id);
      }
    }
  }
  return results;
}

}  // namespace euler
