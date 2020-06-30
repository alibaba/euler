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

#ifndef EULER_PARSER_TRANSLATOR_H_
#define EULER_PARSER_TRANSLATOR_H_
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

#include "euler/parser/tree.h"
#include "euler/parser/attribute_calculator.h"
#include "euler/parser/gen_node_def_input_output.h"
#include "euler/parser/optimizer.h"
#include "euler/core/dag_def/dag_def.h"
namespace euler {

class Translator {
 public:
  explicit Translator(OptimizerType type) : env_type_(type) {
    func_map_["SIMPLE_CONDITION"] = SimpleCondition;
    func_map_["HAS_KEY"] = HasKey;
    func_map_["HAS_LABEL"] = HasLabel;
    func_map_["HAS"] = Has;
    func_map_["TERM"] = Term;
    func_map_["CONJ"] = CONJ;
    func_map_["DNF"] = DNF;
    func_map_["LIMIT"] = Limit;
    func_map_["ORDER_BY"] = OrderBy;
    func_map_["AS"] = As;
    func_map_["POST_PROCESS"] = PostProcess;
    func_map_["CONDITION"] = Condtition;
    func_map_["PARAMS"] = Params;
    func_map_["VA"] = Va;
    func_map_["SAMPLE_NB"] = SampleNB;
    func_map_["SAMPLE_LNB"] = SampleLNB;
    func_map_["SAMPLE_EDGE"] = SampleEdge;
    func_map_["SAMPLE_NODE"] = SampleNode;
    func_map_["SAMPLE_N_WITH_TYPES"] = SampleNWithTypes;
    func_map_["E"] = E;
    func_map_["V"] = V;
    func_map_["API_SAMPLE_NB"] = APISampleNB;
    func_map_["API_SAMPLE_LNB"] = APISampleLNB;
    func_map_["API_GET_NB_EDGE"] = APIGetNBEdge;
    func_map_["API_GET_RNB_NODE"] = APIGetRNBNode;
    func_map_["API_GET_NB_NODE"] = APIGetNBNode;
    func_map_["API_GET_NODE_T"] = APIGetNode;
    func_map_["API_GET_P"] = APIGetP;
    func_map_["API_SAMPLE_EDGE"] = APISampleEdge;
    func_map_["API_GET_EDGE"] = APIGetEdge;
    func_map_["API_SAMPLE_NODE"] = APISampleNode;
    func_map_["API_SAMPLE_N_WITH_TYPES"] = APISampleNWithTypes;
    func_map_["API_GET_NODE"] = APIGetNode;
    func_map_["SELECT"] = Select;

    // gen_input
    node_inputs_map_["API_SAMPLE_NB"] = SampleNBInputs;
    node_inputs_map_["API_GET_NB_EDGE"] = GetNBEdgeInputs;
    node_inputs_map_["API_GET_RNB_NODE"] = GetRNBNodeInputs;
    node_inputs_map_["API_GET_NB_NODE"] = GetNBNodeInputs;
    node_inputs_map_["API_GET_NODE_T"] = GetNodeTInputs;
    node_inputs_map_["API_GET_P"] = GetPInputs;
    node_inputs_map_["API_SAMPLE_EDGE"] = SampleEdgeInputs;
    node_inputs_map_["API_GET_EDGE"] = GetEdgeInputs;
    node_inputs_map_["API_SAMPLE_NODE"] = SampleNodeInputs;
    node_inputs_map_["API_SAMPLE_N_WITH_TYPES"] = SampleNWithTypesInputs;
    node_inputs_map_["API_GET_NODE"] = GetNodeInputs;

    // gen_output
    node_output_num_map_["API_SAMPLE_NB"] = SampleNBOutputNum;
    node_output_num_map_["API_GET_NB_EDGE"] = GetNBEdgeOutputNum;
    node_output_num_map_["API_GET_RNB_NODE"] = GetRNBNodeOutputNum;
    node_output_num_map_["API_GET_NB_NODE"] = GetNBNodeOutputNum;
    node_output_num_map_["API_GET_NODE_T"] = GetNodeTOutputNum;
    node_output_num_map_["API_GET_P"] = GetPOutputNum;
    node_output_num_map_["API_SAMPLE_EDGE"] = SampleEdgeOutputNum;
    node_output_num_map_["API_GET_EDGE"] = GetEdgeOutputNum;
    node_output_num_map_["API_SAMPLE_NODE"] = SampleNodeOutputNum;
    node_output_num_map_["API_SAMPLE_N_WITH_TYPES"] = SampleNWithTypesOutputNum;
    node_output_num_map_["API_GET_NODE"] = GetNodeOutputNum;

    // build_node
    build_node_map_["API_SAMPLE_NB"] = &Translator::SampleNBNodeBuilder;
    build_node_map_["API_GET_NB_EDGE"] = &Translator::SingleNodeBuilder;
    build_node_map_["API_GET_RNB_NODE"] = &Translator::SingleNodeBuilder;
    build_node_map_["API_GET_NB_NODE"] = &Translator::GetNBNodeBuilder;
    build_node_map_["API_GET_NODE_T"] = &Translator::SingleNodeBuilder;
    build_node_map_["API_GET_P"] = &Translator::SingleNodeBuilder;
    build_node_map_["API_SAMPLE_EDGE"] = &Translator::SingleNodeBuilder;
    build_node_map_["API_GET_EDGE"] = &Translator::SingleNodeBuilder;
    build_node_map_["API_SAMPLE_NODE"] = &Translator::SingleNodeBuilder;
    build_node_map_["API_SAMPLE_N_WITH_TYPES"] = &Translator::SingleNodeBuilder;
    build_node_map_["API_GET_NODE"] = &Translator::SingleNodeBuilder;
    build_node_map_["API_SAMPLE_LNB"] = &Translator::LayerSamplerNodeBuilder;
  }

  void Translate(const Tree& tree, DAGDef* dag_def);

 private:
  OptimizerType env_type_;
  std::unordered_map<std::string, bool(*)(TreeNode*)> func_map_;
  std::unordered_map<std::string, void(*)(const NodeDef&, NodeDef*)>
      node_inputs_map_;
  std::unordered_map<std::string, int32_t(*)(const NodeDef&)>
      node_output_num_map_;
  typedef int32_t (Translator::*BuildNode)(
      const TreeNode&, int32_t, DAGDef*,
      std::unordered_map<std::string, int32_t>*);
  std::unordered_map<std::string, BuildNode> build_node_map_;

  void FillNodeDef(const TreeNode& tree_node, NodeDef* node_def);

  std::shared_ptr<NodeDef> GetPreNode(
      const DAGDef& dag_def, const TreeNode& api_node,
      const std::unordered_map<std::string, int32_t>& as_table,
      int32_t default_pre_node_id);

  void AddAsNode(
      NodeDef* node_def, DAGDef* dag_def,
      std::unordered_map<std::string, int32_t>* as_table);

  std::shared_ptr<NodeDef> AddPostProcessNode(
      const NodeDef& node_def, DAGDef* dag_def);

  int32_t SingleNodeBuilder(
      const TreeNode& tree_node,
      int32_t default_pre_node_id, DAGDef* dag_def,
      std::unordered_map<std::string, int32_t>* as_table);

  int32_t GetNBNodeBuilder(
      const TreeNode& tree_node,
      int32_t default_pre_node_id, DAGDef* dag_def,
      std::unordered_map<std::string, int32_t>* as_table);

  int32_t SampleNBNodeBuilder(
      const TreeNode& tree_node,
      int32_t default_pre_node_id, DAGDef* dag_def,
      std::unordered_map<std::string, int32_t>* as_table);

  std::vector<std::shared_ptr<NodeDef>>
  TrivialSampleLayer(const std::string& edge_types, const std::string& n,
                     const std::string& m, const std::string& default_node,
                     const TreeNode& tree_node,
                     int32_t default_pre_node_id,
                     const std::unordered_map<std::string, int32_t>& as_table,
                     DAGDef* dag_def);

  std::vector<std::shared_ptr<NodeDef>>
  GeneralSampleLayer(const std::string& edge_types, const std::string& n,
                     const std::string& m, const std::string& weight_func,
                     const std::string& default_node,
                     const TreeNode& tree_node,
                     int32_t default_pre_node_id,
                     const std::unordered_map<std::string, int32_t>& as_table,
                     DAGDef* dag_def);

  int32_t LayerSamplerNodeBuilder(
      const TreeNode& tree_node,
      int32_t default_pre_node_id, DAGDef* dag_def,
      std::unordered_map<std::string, int32_t>* as_table);
};
}  // namespace euler
#endif  // EULER_PARSER_TRANSLATOR_H_
