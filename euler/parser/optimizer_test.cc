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

#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "euler/parser/optimize_rule.h"
#include "euler/parser/optimizer.h"
#include "euler/core/dag_def/dag_def.h"

namespace euler {

bool CheckStrct(std::shared_ptr<NodeDef> node,
                std::vector<int32_t> pre,
                std::vector<int32_t> succ) {
  if (node->pre_.size() != pre.size()) {
    return false;
  }
  for (int32_t p : pre) {
    if (node->pre_.find(p) == node->pre_.end()) {
      return false;
    }
  }
  if (node->succ_.size() != succ.size()) {
    return false;
  }
  for (int32_t s : succ) {
    if (node->succ_.find(s) == node->succ_.end()) {
      return false;
    }
  }
  return true;
}

TEST(OptimizerTest, Optimizer) {
  std::vector<std::vector<std::string>> fusion_output_map;
  // 0 inner_node's 1 ouput -> fusion_node's 0 output
  fusion_output_map.push_back({"2", "0", "1", "0"});
  // 1 inner_node's 0 ouput -> fusion_node's 1 output
  fusion_output_map.push_back({"3", "1", "0", "1"});
  OptimizeRule optmz_rule1({"2:0 3:1", "3:1"}, "NEW", fusion_output_map);
  Optimizer optimizer(local);
  optimizer.AddRule(optmz_rule1);

  /*
   0 1
   |/
   2
   |\
   3 4
   |/
   5
   * */
  OptimizeRule optmz_rule(
      {"0:0 2:2", "1:1 2:2", "2:2 3:3 4:4", "3:3 5:5", "4:4 5:5", "5:5"},
       "", {});
  DAGDef dag(optmz_rule.adj_info_);
  std::shared_ptr<NodeDef> node0 = dag.GetNodeById(0);
  std::shared_ptr<NodeDef> node1 = dag.GetNodeById(1);
  std::shared_ptr<NodeDef> node2 = dag.GetNodeById(2);
  std::shared_ptr<NodeDef> node3 = dag.GetNodeById(3);
  std::shared_ptr<NodeDef> node4 = dag.GetNodeById(4);
  std::shared_ptr<NodeDef> node5 = dag.GetNodeById(5);
  node2->input_edges_.push_back({"0", 0, 0});
  node2->input_edges_.push_back({"1", 1, 0});
  node3->input_edges_.push_back({"2", 2, 0});
  node4->input_edges_.push_back({"2", 2, 1});
  node5->input_edges_.push_back({"3", 3, 0});
  node5->input_edges_.push_back({"4", 4, 0});
  optimizer.Optimize(&dag);
  int32_t fusion_id = 6;
  bool result = true;
  result = result && CheckStrct(node0, {}, {fusion_id});
  result = result && CheckStrct(node1, {}, {fusion_id});
  result = result && CheckStrct(dag.GetNodeById(fusion_id), {0, 1}, {4, 5});
  result = result && CheckStrct(node4, {fusion_id}, {5});
  result = result && CheckStrct(node5, {fusion_id, 4}, {});
  ASSERT_EQ(result, true);

  std::shared_ptr<NodeDef> fusion_node = dag.GetNodeById(fusion_id);
  for (EdgeDef ed : fusion_node->input_edges_) {
    std::cout << ed.src_name_
        << "," << ed.src_id_
        << "," << ed.src_slot_ << std::endl;
  }

  for (EdgeDef ed : node4->input_edges_) {
    std::cout << ed.src_name_
        << "," << ed.src_id_
        << "," << ed.src_slot_ << std::endl;
  }

  for (EdgeDef ed : node5->input_edges_) {
    std::cout << ed.src_name_
        << "," << ed.src_id_
        << "," << ed.src_slot_ << std::endl;
  }
}

void PrintNodeDef(const NodeDef& node) {
  std::cout << "===============" << std::endl;
  std::cout << "name:id " << node.name_ << ":" <<
      node.id_ << std::endl;
  std::cout << "pre: ";
  for (int32_t pre : node.pre_) {
    std::cout << pre << " ";
  }
  std::cout << std::endl;
  std::cout << "succ: ";
  for (int32_t succ : node.succ_) {
    std::cout << succ << " ";
  }
  std::cout << std::endl;
  std::cout << "input edges:" << std::endl;
  for (EdgeDef input : node.input_edges_) {
    std::cout << input.src_name_ << "," <<
        input.src_id_ << "," <<
        input.src_slot_ << std::endl;
  }
  if (node.name_ == "REMOTE") {
    std::cout << "[" << std::endl;
    for (NodeDef inner_node :
         static_cast<const RemoteNodeDef*>(&node)->nodes_) {
      PrintNodeDef(inner_node);
    }
    std::cout << "]" << std::endl;
  }
}

TEST(OptimizerTest, OptimizerShard) {
  std::vector<std::vector<std::string>> fusion_output_map;
  // 1 inner_node's 0 ouput -> fusion_node's 0 output
  fusion_output_map.push_back({"2", "1", "0", "0"});
  OptimizeRule optmz_rule1({"1:0 2:1", "2:1"}, "REMOTE", fusion_output_map,
                           "SPLIT", "MERGE", 2);
  Optimizer optimizer(distribute);
  optimizer.AddRule(optmz_rule1);

  // 0-1-2-3-4-5  id
  // 0-1-2-1-2-3  name
  OptimizeRule optmz_rule(
      {"0:0 1:1", "1:1 2:2", "2:2 1:3", "1:3 2:4", "2:4 3:5", "3:5"},
       "", {});  // fake
  DAGDef dag(optmz_rule.adj_info_);
  dag.GetNodeById(1)->input_edges_.push_back({"0", 0, 0});
  dag.GetNodeById(2)->input_edges_.push_back({"1", 1, 0});
  dag.GetNodeById(3)->input_edges_.push_back({"2", 2, 0});
  dag.GetNodeById(4)->input_edges_.push_back({"1", 3, 0});
  dag.GetNodeById(5)->input_edges_.push_back({"2", 4, 0});

  /*    / 17(21-20) \    / 8(12-11) \
   * 0-16            19-7            10-5  // id
   *    \ 18(23-22) /    \ 9(14-13) /
   * */
  optimizer.Optimize(&dag);

  PrintNodeDef(*dag.GetNodeById(0));
  PrintNodeDef(*dag.GetNodeById(16));
  PrintNodeDef(*dag.GetNodeById(17));
  PrintNodeDef(*dag.GetNodeById(18));
  PrintNodeDef(*dag.GetNodeById(19));
  PrintNodeDef(*dag.GetNodeById(7));
  PrintNodeDef(*dag.GetNodeById(8));
  PrintNodeDef(*dag.GetNodeById(9));
  PrintNodeDef(*dag.GetNodeById(10));
  PrintNodeDef(*dag.GetNodeById(5));
}

}  // namespace euler
