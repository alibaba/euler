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

#include <algorithm>
#include <utility>

#include "gtest/gtest.h"

#include "euler/core/dag_def/dag_def.h"
#include "euler/core/dag_def/sub_graph_iso.h"
namespace euler {

bool CheckStrct(std::shared_ptr<NodeDef> node,
                const std::vector<int32_t>& pre,
                const std::vector<int32_t>& succ) {
  if (node->pre_.size() != pre.size()) {
    EULER_LOG(ERROR) << "pre size not match " << node->pre_.size();
    return false;
  }
  for (int32_t p : pre) {
    if (node->pre_.find(p) == node->pre_.end()) {
      return false;
    }
  }
  if (node->succ_.size() != succ.size()) {
    EULER_LOG(ERROR) << "succ size not match " << node->succ_.size();
    return false;
  }
  for (int32_t s : succ) {
    if (node->succ_.find(s) == node->succ_.end()) {
      return false;
    }
  }
  return true;
}

bool CheckNode(std::shared_ptr<NodeDef> node,
               const std::vector<int32_t>& pre,
               const std::vector<int32_t>& succ,
               const std::vector<EdgeDef>& input_edges) {
  bool result = CheckStrct(node, pre, succ);
  if (!result) return false;
  result = input_edges.size() == node->input_edges_.size();
  for (size_t i = 0; i < input_edges.size(); ++i) {
    /*
    EULER_LOG(INFO)
        << node->input_edges_[i].src_name_ << ","
        << node->input_edges_[i].src_id_ << ","
        << node->input_edges_[i].src_slot_;
    */
    result = result &&
        input_edges[i].src_name_ == node->input_edges_[i].src_name_ &&
        input_edges[i].src_id_ == node->input_edges_[i].src_id_ &&
        input_edges[i].src_slot_ == node->input_edges_[i].src_slot_;
  }
  return result;
}

bool CheckRemoteNode(std::shared_ptr<NodeDef> node,
                     const std::vector<int32_t>& pre,
                     const std::vector<int32_t>& succ,
                     const std::vector<EdgeDef>& input_edges,
                     const std::vector<std::shared_ptr<NodeDef>>& nodes) {
  bool result = CheckNode(node, pre, succ, input_edges);
  std::shared_ptr<RemoteNodeDef> remote_node =
      std::dynamic_pointer_cast<RemoteNodeDef>(node);
  result = result && nodes.size() == remote_node->nodes_.size();
  for (std::shared_ptr<NodeDef> a : nodes) {
    bool temp = false;
    for (std::shared_ptr<NodeDef> b : remote_node->nodes_) {
      if (a->id_ == b->id_ && a.get() == b.get()) {
        temp = true;
      }
    }
    result = result && temp;
  }
  return result;
}

DAGDef BuildDAG() {
  DAGDef dag;
  /*
   0 1
   |/
   2
   |\
   3 4
   |/
   5
   * */
  std::shared_ptr<NodeDef> node0 = dag.ProduceNodeDef("0", 1);
  std::shared_ptr<NodeDef> node1 = dag.ProduceNodeDef("1", 1);
  std::shared_ptr<NodeDef> node2 = dag.ProduceNodeDef("2", 1);
  node2->input_edges_.push_back({"0", 0, 0});
  node2->input_edges_.push_back({"1", 1, 0});
  std::shared_ptr<NodeDef> node3 = dag.ProduceNodeDef("3", 1);
  node3->input_edges_.push_back({"2", 2, 0});
  std::shared_ptr<NodeDef> node4 = dag.ProduceNodeDef("4", 1);
  node4->input_edges_.push_back({"2", 2, 0});
  std::shared_ptr<NodeDef> node5 = dag.ProduceNodeDef("5", 1);
  node5->input_edges_.push_back({"3", 3, 0});
  node5->input_edges_.push_back({"4", 4, 0});
  std::unordered_set<int32_t> pre0;
  std::unordered_set<int32_t> succ;
  dag.AddNodeDef(node0, pre0, succ);
  std::unordered_set<int32_t> pre1;
  dag.AddNodeDef(node1, pre1, succ);
  std::unordered_set<int32_t> pre2; pre2.insert(0); pre2.insert(1);
  dag.AddNodeDef(node2, pre2, succ);
  std::unordered_set<int32_t> pre3; pre3.insert(2);
  dag.AddNodeDef(node3, pre3, succ);
  std::unordered_set<int32_t> pre4; pre4.insert(2);
  dag.AddNodeDef(node4, pre4, succ);
  std::unordered_set<int32_t> pre5; pre5.insert(3); pre5.insert(4);
  dag.AddNodeDef(node5, pre5, succ);
  return dag;
}

TEST(DAGDef, MacroFusion) {
  DAGDef dag = BuildDAG();
  std::unordered_set<std::string> local_only_ops = {"3"};
  std::unordered_set<int32_t> results = dag.MacroFusionProcess(local_only_ops);
  std::unordered_set<int32_t> expect = {0, 1, 2, 4};
  ASSERT_EQ(results.size(), expect.size());
  for (int32_t id : results) {
    ASSERT_TRUE(expect.find(id) != expect.end());
  }
}

TEST(DAGDef, Init) {
  DAGDef dag = BuildDAG();
  bool result = true;
  result = result && CheckStrct(dag.GetNodeById(0), {}, {2});
  result = result && CheckStrct(dag.GetNodeById(1), {}, {2});
  result = result && CheckStrct(dag.GetNodeById(2), {0, 1}, {3, 4});
  result = result && CheckStrct(dag.GetNodeById(3), {2}, {5});
  result = result && CheckStrct(dag.GetNodeById(4), {2}, {5});
  result = result && CheckStrct(dag.GetNodeById(5), {3, 4}, {});
  ASSERT_EQ(result, true);
}

TEST(DAGDef, ISO) {
  DAGDef gm, gp;
  /*
   gm:
     0 1
     |/
     2
   / | \
   3 4 3
   \ | /
     6

   gp:
     2
     |\
     3 4
     |/
     6
  * */
  std::shared_ptr<NodeDef> node0 = gm.ProduceNodeDef("0", 1);  // id = 0
  std::shared_ptr<NodeDef> node1 = gm.ProduceNodeDef("1", 1);  // id = 1
  std::shared_ptr<NodeDef> node2 = gm.ProduceNodeDef("2", 1);  // id = 2
  std::shared_ptr<NodeDef> node3 = gm.ProduceNodeDef("3", 1);  // id = 3
  std::shared_ptr<NodeDef> node4 = gm.ProduceNodeDef("4", 1);  // id = 4
  std::shared_ptr<NodeDef> node5 = gm.ProduceNodeDef("3", 1);  // id = 5
  std::shared_ptr<NodeDef> node6 = gm.ProduceNodeDef("6", 1);  // id = 6

  std::unordered_set<int32_t> pre0;
  std::unordered_set<int32_t> succ;
  gm.AddNodeDef(node0, pre0, succ);
  std::unordered_set<int32_t> pre1;
  gm.AddNodeDef(node1, pre1, succ);
  std::unordered_set<int32_t> pre2; pre2.insert(0); pre2.insert(1);
  gm.AddNodeDef(node2, pre2, succ);
  std::unordered_set<int32_t> pre3; pre3.insert(2);
  gm.AddNodeDef(node3, pre3, succ);
  std::unordered_set<int32_t> pre4; pre4.insert(2);
  gm.AddNodeDef(node4, pre4, succ);
  std::unordered_set<int32_t> pre5; pre5.insert(2);
  gm.AddNodeDef(node5, pre5, succ);
  std::unordered_set<int32_t> pre6;
  pre6.insert(3); pre6.insert(4); pre6.insert(5);
  gm.AddNodeDef(node6, pre6, succ);

  std::shared_ptr<NodeDef> node12 = gp.ProduceNodeDef("2", 1);  // id = 0
  std::shared_ptr<NodeDef> node13 = gp.ProduceNodeDef("3", 1);  // id = 1
  std::shared_ptr<NodeDef> node14 = gp.ProduceNodeDef("4", 1);  // id = 2
  std::shared_ptr<NodeDef> node16 = gp.ProduceNodeDef("6", 1);  // id = 3

  std::unordered_set<int32_t> pre12;
  std::unordered_set<int32_t> succ0;
  gp.AddNodeDef(node12, pre12, succ0);
  std::unordered_set<int32_t> pre13;
  gp.AddNodeDef(node13, pre13, succ0); pre13.insert(2);
  std::unordered_set<int32_t> pre14;
  gp.AddNodeDef(node14, pre14, succ0); pre14.insert(2);
  std::unordered_set<int32_t> pre16;
  gp.AddNodeDef(node16, pre16, succ0); pre16.insert(3); pre16.insert(4);

  std::unordered_map<std::string, bool(*)(const NodeDef& node_m)> extra_cond;
  std::vector<std::unordered_map<int32_t, int32_t>>
  matches = SubGraphMatch(gm, gp, extra_cond);
  for (std::unordered_map<int32_t, int32_t> match : matches) {
    for (std::pair<int32_t, int32_t> e : match) {
      if (e.first == 0) {
        ASSERT_EQ(2, e.second);
      } else if (e.first == 1) {
        ASSERT_TRUE(3 == e.second || 5 == e.second);
      } else if (e.first == 2) {
        ASSERT_EQ(4, e.second);
      } else if (e.first == 3) {
        ASSERT_EQ(6, e.second);
      }
    }
  }
}

TEST(DAGDef, TopologicalSort) {
  DAGDef dag = BuildDAG();
  std::unordered_set<int32_t> sub_nodes;
  sub_nodes.insert(2);
  sub_nodes.insert(3);
  sub_nodes.insert(4);
  sub_nodes.insert(5);
  std::vector<int32_t> result = dag.TopologicSort(sub_nodes);
  std::vector<int32_t> true_result = {2, 3, 4, 5};
  for (size_t i = 0; i < result.size(); ++i) {
    ASSERT_EQ(result[i], true_result[i]);
  }
}

TEST(DAGDef, FusionRemote) {
  DAGDef dag = BuildDAG();
  std::unordered_set<int32_t> nodes;
  nodes.insert(2);
  nodes.insert(3);
  nodes.insert(4);
  std::vector<FusionOutput> fusion_output_map = {
      {"3", 3, 0, 0}, {"4", 4, 0, 1}};
  std::shared_ptr<NodeDef> node2 = dag.GetNodeById(2);
  std::shared_ptr<NodeDef> node3 = dag.GetNodeById(3);
  std::shared_ptr<NodeDef> node4 = dag.GetNodeById(4);
  FusionRule fusion_rule;
  fusion_rule.fusion_name_ = "REMOTE";
  fusion_rule.fusion_output_map_ = fusion_output_map;
  int32_t fusion_id = dag.FusionNodes(nodes, fusion_rule);
  ASSERT_EQ(6, fusion_id);
  std::vector<EdgeDef> ie0 = {};
  std::vector<EdgeDef> ie1 = {};
  std::vector<EdgeDef> ie2 = {{"REMOTE", 6, 0}, {"REMOTE", 6, 1}};
  std::vector<EdgeDef> ie3 = {{"2", 2, 0}};
  std::vector<EdgeDef> ie4 = {{"2", 2, 0}};
  std::vector<EdgeDef> ie6 = {{"0", 0, 0}, {"1", 1, 0}};
  std::vector<EdgeDef> ie5 = {{"REMOTE", 6, 0}, {"REMOTE", 6, 1}};
  ASSERT_EQ(CheckNode(dag.GetNodeById(0), {}, {6}, ie0), true);
  ASSERT_EQ(CheckNode(dag.GetNodeById(1), {}, {6}, ie1), true);
  ASSERT_EQ(CheckNode(node2, {}, {3, 4}, ie2), true);
  ASSERT_EQ(CheckNode(node3, {2}, {}, ie3), true);
  ASSERT_EQ(CheckNode(node4, {2}, {}, ie4), true);
  std::vector<std::shared_ptr<NodeDef>> inner_nodes = {node2, node3, node4};
  ASSERT_EQ(CheckRemoteNode(
            dag.GetNodeById(fusion_id), {0, 1}, {5}, ie6, inner_nodes), true);
  ASSERT_EQ(CheckNode(dag.GetNodeById(5), {6}, {}, ie5), true);
}

TEST(DAGDef, Fusion) {
  DAGDef dag = BuildDAG();
  std::unordered_set<int32_t> nodes;
  nodes.insert(2);
  nodes.insert(3);
  nodes.insert(4);
  std::vector<FusionOutput> fusion_output_map = {
      {"3", 3, 0, 0}, {"4", 4, 0, 1}};
  FusionRule fusion_rule;
  fusion_rule.fusion_name_ = "6";
  fusion_rule.fusion_output_map_ = fusion_output_map;
  int32_t fusion_id = dag.FusionNodes(nodes, fusion_rule);
  ASSERT_EQ(6, fusion_id);
  std::vector<EdgeDef> ie0 = {};
  std::vector<EdgeDef> ie1 = {};
  std::vector<EdgeDef> ie6 = {{"0", 0, 0}, {"1", 1, 0}};
  std::vector<EdgeDef> ie5 = {{"6", 6, 0}, {"6", 6, 1}};
  ASSERT_EQ(CheckNode(dag.GetNodeById(0), {}, {6}, ie0), true);
  ASSERT_EQ(CheckNode(dag.GetNodeById(1), {}, {6}, ie1), true);
  ASSERT_EQ(CheckNode(dag.GetNodeById(fusion_id), {0, 1}, {5}, ie6), true);
  ASSERT_EQ(CheckNode(dag.GetNodeById(5), {6}, {}, ie5), true);
}

TEST(DAGDef, Shard) {
  DAGDef dag = BuildDAG();
  std::unordered_set<int32_t> nodes;
  nodes.insert(2);
  std::shared_ptr<NodeDef> node2 = dag.GetNodeById(2);
  std::vector<FusionOutput> fusion_output_map = {{"2", 2, 0, 0}};
  FusionRule fusion_rule;
  fusion_rule.fusion_name_ = "REMOTE";
  fusion_rule.fusion_output_map_ = fusion_output_map;
  int32_t fusion_id = dag.FusionNodes(nodes, fusion_rule);
  std::vector<SplitOpInfo>
  split_op_info_list = {{"split0", {0}}, {"split1", {1}}};
  std::vector<MergeOpInfo> merge_op_info_list =
      {{"merge0", "split", 0, {0}}};
  ShardRule shard_rule;
  shard_rule.split_op_info_list_ = split_op_info_list;
  shard_rule.merge_op_info_list_ = merge_op_info_list;
  shard_rule.split_num_ = 2;
  bool result = dag.ShardRemoteNodeDef(fusion_id, shard_rule);
  ASSERT_EQ(true, result);
  std::vector<EdgeDef> ie0 = {};
  std::vector<EdgeDef> ie1 = {};
  std::vector<EdgeDef> ie2 = {{"REMOTE", 6, 0}, {"REMOTE", 6, 1}};
  std::vector<EdgeDef> ie3 = {{"merge0", 11, 0}};
  std::vector<EdgeDef> ie4 = {{"merge0", 11, 0}};
  std::vector<EdgeDef> ie7 = {{"0", 0, 0}};  // split
  std::vector<EdgeDef> ie8 = {{"1", 1, 0}};  // split
  std::vector<EdgeDef> ie9 = {{"split0", 7, 0}, {"split1", 8, 0}};  // remote
  std::vector<EdgeDef> ie10 = {{"split0", 7, 2}, {"split1", 8, 2}};  // remote
  std::vector<EdgeDef>ie11 = {{"REMOTE", 9, 0}, {"split0", 7, 1},
                              {"REMOTE", 10, 0}, {"split0", 7, 3}};  // merge
  ASSERT_EQ(CheckNode(dag.GetNodeById(0), {}, {7}, ie0), true);
  ASSERT_EQ(CheckNode(dag.GetNodeById(1), {}, {8}, ie1), true);
  ASSERT_EQ(CheckNode(node2, {}, {}, ie2), true);
  ASSERT_EQ(CheckNode(dag.GetNodeById(3), {11}, {5}, ie3), true);
  ASSERT_EQ(CheckNode(dag.GetNodeById(4), {11}, {5}, ie4), true);
  ASSERT_EQ(CheckNode(dag.GetNodeById(7), {0}, {9, 10, 11}, ie7), true);
  ASSERT_EQ(CheckNode(dag.GetNodeById(8), {1}, {9, 10}, ie8), true);
  std::vector<std::shared_ptr<NodeDef>> inner_nodes = {node2};
  ASSERT_EQ(CheckRemoteNode(
            dag.GetNodeById(9), {7, 8}, {11}, ie9, inner_nodes), true);
  ASSERT_EQ(CheckRemoteNode(
            dag.GetNodeById(10), {7, 8}, {11}, ie10, inner_nodes), true);

  ASSERT_EQ(CheckNode(dag.GetNodeById(11), {9, 10, 7}, {3, 4}, ie11), true);
}

}  // namespace euler
