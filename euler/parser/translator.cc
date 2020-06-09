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

#include <unordered_set>
#include <vector>
#include <string>
#include <memory>

#include "euler/parser/translator.h"

#include "euler/parser/compiler.h"
#include "euler/common/logging.h"

namespace euler {

void FillDNF(const Prop& dnf, AttrDef* cond_attr_def) {
  std::vector<Prop*> conjs = dnf.GetNestingValues();
  static_cast<CondAttrDef*>(cond_attr_def)->
      dnf_attr_.resize(conjs.size());
  for (size_t i = 0; i < conjs.size(); ++i) {
    Prop* conj = conjs[i];
    std::vector<Prop*> terms = conj->GetNestingValues();
    for (Prop* term : terms) {
      std::vector<std::string> params = term->GetValues();
      std::string params_str = params[0];
      for (size_t j = 1; j < params.size(); ++j) {
        params_str += " " + params[j];
      }
      static_cast<CondAttrDef*>(cond_attr_def)->
          dnf_attr_[i].push_back(params_str);
    }
  }
}

void FillPostProcess(const Prop& post_process, AttrDef* cond_attr_def) {
  std::vector<Prop*> cmd_list = post_process.GetNestingValues();
  for (Prop* cmd : cmd_list) {
    std::vector<std::string> params = cmd->GetValues();
    std::string params_str = params[0];
    for (size_t j = 1; j < params.size(); ++j) {
      params_str += " " + params[j];
    }
    static_cast<CondAttrDef*>(cond_attr_def)->
        post_process_.push_back(params_str);
  }
}

void CheckNBIndex(const Prop& prop, int* nb_index_cnt, int* not_nb_index_cnt) {
  *nb_index_cnt = 0;
  *not_nb_index_cnt = 0;
  if (prop.GetNestingValues().empty()) return;
  std::unordered_map<std::string, std::vector<std::string>> index_info =
      Compiler::GetInstance()->GetIndexInfo();
  std::unordered_set<std::string> nb_index_set;
  for (const std::string& name : index_info["hash_range_index"]) {
    nb_index_set.insert(name);
  }
  Prop* dnf = prop.GetNestingValues()[0];
  if (dnf != nullptr) {
    std::vector<Prop*> conjs = dnf->GetNestingValues();
    for (Prop* conj : conjs) {
      std::vector<Prop*> terms = conj->GetNestingValues();
      for (Prop* term : terms) {
        std::vector<std::string> params = term->GetValues();
        if (nb_index_set.find(params[0]) == nb_index_set.end()) {
          ++(*not_nb_index_cnt);
        } else {
          ++(*nb_index_cnt);
        }
      }
    }
  }
}

void Translator::FillNodeDef(const TreeNode& tree_node, NodeDef* node_def) {
  Prop* prop = tree_node.GetProp();
  // fill op_alias
  node_def->op_alias_ = tree_node.GetOpAlias();
  // fill norm prop and udf
  bool udf = false, udf_with_num_p = false;
  for (const std::string& prop_key : prop->GetValues()) {
    if (prop_key.substr(0, 4) == "udf_") {
      udf = true;
      node_def->udf_name_ = prop_key;
    } else {
      if (udf == false) {
        node_def->attrs_.push_back(std::make_shared<NormAttrDef>(prop_key));
      } else {
        if (prop_key == "[") {
          udf_with_num_p = true;
        } else if (prop_key == "]") {
          udf_with_num_p = false;
        } else {
          if (!udf_with_num_p) {
            node_def->udf_str_params_.push_back(prop_key);
          } else {
            node_def->udf_num_params_.push_back(prop_key);
          }
        }
      }
    }
  }
  // fill condition
  std::vector<Prop*> condition = prop->GetNestingValues();
  if (!condition.empty()) {
    std::shared_ptr<AttrDef> cond_attr_def = std::make_shared<CondAttrDef>();
    Prop* dnf = condition[0];
    if (dnf != nullptr) {
      FillDNF(*dnf, cond_attr_def.get());
    }
    Prop* post_process = condition[1];
    if (post_process != nullptr) {
      FillPostProcess(*post_process, cond_attr_def.get());
    }
    node_def->attrs_.push_back(cond_attr_def);
  }
}

bool IsSelectPreNode(const TreeNode& api_node,
                     TreeNode** search_with_select_node) {
  TreeNode* search_node = api_node.GetParent();
  if (search_node != nullptr) {
    *(search_with_select_node) = search_node->GetParent();
    if (*(search_with_select_node) == nullptr) {
      EULER_LOG(FATAL) << "node: " << api_node.GetType() << " in wrong place";
    }
    return (*search_with_select_node)->GetChildren()[0]->GetType() == "SELECT";
  } else {
    EULER_LOG(FATAL) << "node: " << api_node.GetType() << " in wrong place";
  }
  return false;
}

std::shared_ptr<NodeDef> Translator::GetPreNode(
    const DAGDef& dag_def,
    const TreeNode& api_node,
    const std::unordered_map<std::string, int32_t>& as_table,
    int32_t default_pre_node_id) {
  TreeNode* search_with_select_node = nullptr;
  if (IsSelectPreNode(api_node, &search_with_select_node)) {
    std::string key = search_with_select_node->GetChildren()[0]->
        GetProp()->GetValues()[0];
    if (as_table.find(key) == as_table.end()) {
      EULER_LOG(FATAL) << "node: " << api_node.GetType() << " select error";
    }
    return dag_def.GetNodeById(as_table.at(key));
  } else {
    return dag_def.GetNodeById(default_pre_node_id);
  }
}

void Translator::AddAsNode(
    NodeDef* node_def, DAGDef* dag_def,
    std::unordered_map<std::string, int32_t>* as_table) {
  std::shared_ptr<NodeDef> as_node =
      dag_def->ProduceNodeDef("AS", node_def->output_num_);
  as_node->op_alias_ = node_def->op_alias_; node_def->op_alias_ = "";
  for (int32_t i = 0; i < node_def->output_num_; ++i) {
    as_node->input_edges_.push_back({node_def->name_, node_def->id_, i});
  }
  std::unordered_set<int32_t> as_node_pre, succ;
  as_node_pre.insert(node_def->id_);
  dag_def->AddNodeDef(as_node, as_node_pre, succ);
  (*as_table)[as_node->op_alias_] = node_def->id_;
}

std::shared_ptr<NodeDef> Translator::AddPostProcessNode(
    const NodeDef& node_def, DAGDef* dag_def) {
  for (std::shared_ptr<AttrDef> attr : node_def.attrs_) {
    if (attr->attr_type_ == AttrDef::kCond) {
      std::shared_ptr<CondAttrDef> cond =
          std::static_pointer_cast<CondAttrDef>(attr);
      if (!cond->post_process_.empty()) {
        std::shared_ptr<NodeDef> pp_node =
            dag_def->ProduceNodeDef("POST_PROCESS", node_def.output_num_);
        pp_node->op_alias_ = node_def.name_;
        std::shared_ptr<CondAttrDef> cond_attr_def =
            std::make_shared<CondAttrDef>();
        for (const std::string& pp : cond->post_process_) {
          std::vector<std::string> vec = Split(pp, " ");
          if (vec[0] == "order_by" && vec[1] == "weight" &&
              node_def.name_ != "API_GET_NODE_WITH_WEIGHT" &&
              node_def.name_ != "API_GET_EDGE_WITH_WEIGHT") {
            EULER_LOG(FATAL) << "order by weight need weight output";
          }
          cond_attr_def->post_process_.push_back(pp);
        }
        for (int32_t i = 0; i < node_def.output_num_; ++i) {
          pp_node->input_edges_.push_back({node_def.name_, node_def.id_, i});
        }
        pp_node->attrs_.push_back(cond_attr_def);
        std::unordered_set<int32_t> pp_node_pre, succ;
        pp_node_pre.insert(node_def.id_);
        dag_def->AddNodeDef(pp_node, pp_node_pre, succ);
        return pp_node;  // assume only one cond attr
      }
    }
  }
  return nullptr;
}

int32_t Translator::SingleNodeBuilder(
    const TreeNode& tree_node, int32_t default_pre_node_id, DAGDef* dag_def,
    std::unordered_map<std::string, int32_t>* as_table) {
  NodeDef empty;
  // produce NodeDef
  std::shared_ptr<NodeDef> node_def =
      dag_def->ProduceNodeDef(tree_node.GetType(), 0);
  // fill node
  FillNodeDef(tree_node, node_def.get());
  // get pre node
  std::shared_ptr<NodeDef> pre_node = GetPreNode(
      *dag_def, tree_node, *as_table, default_pre_node_id);
  if (pre_node == nullptr)
    node_inputs_map_[node_def->name_](empty, node_def.get());
  else
    node_inputs_map_[node_def->name_](*pre_node, node_def.get());  // gen input
  node_def->output_num_ =
      node_output_num_map_[node_def->name_](*node_def);  // gen output
  // add into DAGDef
  std::unordered_set<int32_t> pre; std::unordered_set<int32_t> succ;
  if (pre_node != nullptr) pre.insert(pre_node->id_);
  dag_def->AddNodeDef(node_def, pre, succ);
  // add post process
  std::shared_ptr<NodeDef> pp_node = nullptr;
  if (env_type_ != local) {
    if (node_def->name_ == "API_GET_NODE" ||
        node_def->name_ == "API_GET_EDGE") {
      pp_node = AddPostProcessNode(*node_def, dag_def);
    }
  }
  // add as node
  if (node_def->op_alias_ != "") {
    pp_node == nullptr ?
      AddAsNode(node_def.get(), dag_def, as_table) :
      AddAsNode(pp_node.get(), dag_def, as_table);
  }
  return pp_node == nullptr ? node_def->id_ : pp_node->id_;
}

int32_t Translator::GetNBNodeBuilder(
    const TreeNode& tree_node, int32_t default_pre_node_id, DAGDef* dag_def,
    std::unordered_map<std::string, int32_t>* as_table) {
  Prop* prop = tree_node.GetProp();
  int32_t not_nb_index_cnt = 0;
  int32_t nb_index_cnt = 0;
  CheckNBIndex(*prop, &nb_index_cnt, &not_nb_index_cnt);
  if (nb_index_cnt != 0 && not_nb_index_cnt != 0) {
    EULER_LOG(FATAL)
        << "index type should be all neighbor index or global index!";
    return -1;
  } else if (env_type_ != local && nb_index_cnt == 0 &&
             not_nb_index_cnt != 0) {
    EULER_LOG(INFO)
        << "using global index to filter nbs will cause performance issue!";
    std::unordered_set<int32_t> pre; std::unordered_set<int32_t> succ;
    /* build API_GET_NB_NODE without dnf_cond and post_process cmd */
    std::shared_ptr<NodeDef> node_def0 =
        dag_def->ProduceNodeDef("API_GET_NB_NODE", 4);
    for (const std::string& prop_key : prop->GetValues()) {
      node_def0->attrs_.push_back(std::make_shared<NormAttrDef>(prop_key));
    }
    std::shared_ptr<NodeDef> pre_node = GetPreNode(
        *dag_def, tree_node, *as_table, default_pre_node_id);
    node_inputs_map_[node_def0->name_](*pre_node, node_def0.get());
    pre.insert(pre_node->id_);
    dag_def->AddNodeDef(node_def0, pre, succ);

    /* build API_GET_NODE with dnf_cond */
    std::shared_ptr<NodeDef> node_def1 =
        dag_def->ProduceNodeDef("API_GET_NODE", 1);
    std::vector<Prop*> condition = prop->GetNestingValues();
    std::shared_ptr<AttrDef> dnf_attr_def = std::make_shared<CondAttrDef>();
    Prop* dnf = condition[0];
    if (dnf != nullptr) {
      FillDNF(*dnf, dnf_attr_def.get());
    }
    node_def1->attrs_.push_back(dnf_attr_def);
    node_def1->input_edges_.push_back({node_def0->name_, node_def0->id_, 1});
    pre.clear(); pre.insert(node_def0->id_);
    dag_def->AddNodeDef(node_def1, pre, succ);

    /* build API_GET_NB_FILTER with post_process */
    std::shared_ptr<NodeDef> node_def2 =
        dag_def->ProduceNodeDef("API_GET_NB_FILTER", 4);
    std::shared_ptr<AttrDef> pp_attr_def = std::make_shared<CondAttrDef>();
    Prop* post_process = condition[1];
    if (post_process != nullptr) {
      FillPostProcess(*post_process, pp_attr_def.get());
    }
    node_def2->attrs_.push_back(pp_attr_def);
    node_def2->input_edges_.push_back({node_def0->name_, node_def0->id_, 0});
    node_def2->input_edges_.push_back({node_def0->name_, node_def0->id_, 1});
    node_def2->input_edges_.push_back({node_def0->name_, node_def0->id_, 2});
    node_def2->input_edges_.push_back({node_def0->name_, node_def0->id_, 3});
    node_def2->input_edges_.push_back({node_def1->name_, node_def1->id_, 0});
    pre.clear(); pre.insert(node_def0->id_); pre.insert(node_def1->id_);
    dag_def->AddNodeDef(node_def2, pre, succ);

    /* add as node */
    node_def2->op_alias_ = tree_node.GetOpAlias();
    if (node_def2->op_alias_ != "") {
      AddAsNode(node_def2.get(), dag_def, as_table);
    }
    return node_def2->id_;
  } else {
    return SingleNodeBuilder(tree_node, default_pre_node_id,
                             dag_def, as_table);
  }
}

int32_t Translator::SampleNBNodeBuilder(
    const TreeNode& tree_node, int32_t default_pre_node_id, DAGDef* dag_def,
    std::unordered_map<std::string, int32_t>* as_table) {
  Prop* prop = tree_node.GetProp();
  int32_t not_nb_index_cnt = 0;
  int32_t nb_index_cnt = 0;
  CheckNBIndex(*prop, &nb_index_cnt, &not_nb_index_cnt);
  if (not_nb_index_cnt != 0) {
    EULER_LOG(FATAL) << "sample neighbor support neighbor index only";
  }
  return SingleNodeBuilder(tree_node, default_pre_node_id,
                           dag_def, as_table);
}

std::vector<std::shared_ptr<NodeDef>>
Translator::TrivialSampleLayer(
    const std::string& edge_types, const std::string& n,
    const std::string& m, const std::string& default_node,
    const TreeNode& tree_node,
    int32_t default_pre_node_id,
    const std::unordered_map<std::string, int32_t>& as_table,
    DAGDef* dag_def) {
  std::unordered_set<int32_t> pre; std::unordered_set<int32_t> succ;
  /* API_GET_EDGE_SUM_WEIGHT */
  std::shared_ptr<NodeDef> node_def0 =
      dag_def->ProduceNodeDef("API_GET_EDGE_SUM_WEIGHT", 2);
  node_def0->attrs_.push_back(std::make_shared<NormAttrDef>(edge_types));
  std::shared_ptr<NodeDef> pre_node = GetPreNode(
      *dag_def, tree_node, as_table, default_pre_node_id);
  if (pre_node->name_ == "API_GATHER_RESULT") {
    node_def0->input_edges_.push_back({pre_node->name_, pre_node->id_, 4});
  } else if (pre_node->name_ == "API_GET_NODE" ||
             pre_node->name_ == "API_SAMPLE_NODE") {
    node_def0->input_edges_.push_back({pre_node->name_, pre_node->id_, 0});
  } else {
    node_def0->input_edges_.push_back({pre_node->name_, pre_node->id_, 1});
  }
  pre.insert(pre_node->id_);
  dag_def->AddNodeDef(node_def0, pre, succ);

  /* API_SAMPLE_ROOT */
  std::shared_ptr<NodeDef> node_def1 =
      dag_def->ProduceNodeDef("API_SAMPLE_ROOT", 1);
  node_def1->attrs_.push_back(std::make_shared<NormAttrDef>(n));
  node_def1->attrs_.push_back(std::make_shared<NormAttrDef>(m));
  node_def1->attrs_.push_back(std::make_shared<NormAttrDef>(default_node));
  node_def1->input_edges_.push_back({node_def0->name_, node_def0->id_, 0});
  node_def1->input_edges_.push_back({node_def0->name_, node_def0->id_, 1});
  pre.clear(); pre.insert(node_def0->id_);
  dag_def->AddNodeDef(node_def1, pre, succ);

  /* API_SAMPLE_L */
  std::shared_ptr<NodeDef> node_def2 =
      dag_def->ProduceNodeDef("API_SAMPLE_L", 3);
  node_def2->attrs_.push_back(std::make_shared<NormAttrDef>(edge_types));
  node_def2->attrs_.push_back(std::make_shared<NormAttrDef>(default_node));
  node_def2->input_edges_.push_back({node_def1->name_, node_def1->id_, 0});
  pre.clear(); pre.insert(node_def1->id_);
  dag_def->AddNodeDef(node_def2, pre, succ);

  std::vector<std::shared_ptr<NodeDef>> results =
      {node_def0, node_def1, node_def2};
  return results;
}

std::vector<std::shared_ptr<NodeDef>>
Translator::GeneralSampleLayer(
    const std::string& edge_types, const std::string& n,
    const std::string& m, const std::string& weight_func,
    const std::string& default_node,
    const TreeNode& tree_node,
    int32_t default_pre_node_id,
    const std::unordered_map<std::string, int32_t>& as_table,
    DAGDef* dag_def) {
  std::unordered_set<int32_t> pre; std::unordered_set<int32_t> succ;
  /* API_RESHAPE */
  std::shared_ptr<NodeDef> node_def0 =
      dag_def->ProduceNodeDef("API_RESHAPE", 1);
  node_def0->attrs_.push_back(std::make_shared<NormAttrDef>("?,1"));
  std::shared_ptr<NodeDef> pre_node = GetPreNode(
      *dag_def, tree_node, as_table, default_pre_node_id);
  if (pre_node->name_ == "API_GATHER_RESULT") {
    node_def0->input_edges_.push_back({pre_node->name_, pre_node->id_, 4});
  } else if (pre_node->name_ == "API_GET_NODE" ||
             pre_node->name_ == "API_SAMPLE_NODE") {
    node_def0->input_edges_.push_back({pre_node->name_, pre_node->id_, 0});
  } else {
    node_def0->input_edges_.push_back({pre_node->name_, pre_node->id_, 1});
  }
  pre.insert(pre_node->id_);
  dag_def->AddNodeDef(node_def0, pre, succ);

  /* API_GET_NB_NODE */
  std::shared_ptr<NodeDef> node_def1 =
      dag_def->ProduceNodeDef("API_GET_NB_NODE", 4);
  node_def1->attrs_.push_back(std::make_shared<NormAttrDef>(edge_types));
  if (pre_node->name_ == "API_GATHER_RESULT") {
    node_def1->input_edges_.push_back({pre_node->name_, pre_node->id_, 4});
  } else if (pre_node->name_ == "API_GET_NODE" ||
             pre_node->name_ == "API_SAMPLE_NODE") {
    node_def1->input_edges_.push_back({pre_node->name_, pre_node->id_, 0});
  } else {
    node_def1->input_edges_.push_back({pre_node->name_, pre_node->id_, 1});
  }
  pre.clear(); pre.insert(pre_node->id_);
  dag_def->AddNodeDef(node_def1, pre, succ);

  /* API_LOCAL_SAMPLE_L */
  std::shared_ptr<NodeDef> node_def2 =
      dag_def->ProduceNodeDef("API_LOCAL_SAMPLE_L", 3);
  node_def2->attrs_.push_back(std::make_shared<NormAttrDef>(n));
  node_def2->attrs_.push_back(std::make_shared<NormAttrDef>(m));
  node_def2->attrs_.push_back(std::make_shared<NormAttrDef>(weight_func));
  node_def2->attrs_.push_back(std::make_shared<NormAttrDef>(default_node));
  node_def2->input_edges_.push_back({node_def1->name_, node_def1->id_, 0});
  node_def2->input_edges_.push_back({node_def1->name_, node_def1->id_, 1});
  node_def2->input_edges_.push_back({node_def1->name_, node_def1->id_, 2});
  node_def2->input_edges_.push_back({node_def1->name_, node_def1->id_, 3});
  pre.clear(); pre.insert(node_def1->id_);
  dag_def->AddNodeDef(node_def2, pre, succ);

  std::vector<std::shared_ptr<NodeDef>> results =
      {node_def0, node_def1, node_def2};
  return results;
}

int32_t Translator::LayerSamplerNodeBuilder(
    const TreeNode& tree_node, int32_t default_pre_node_id, DAGDef* dag_def,
    std::unordered_map<std::string, int32_t>* as_table) {
  Prop* prop = tree_node.GetProp();
  const std::vector<std::string>& p_values = prop->GetValues();

  std::string edge_types, n, m, weight_func, default_node;
  std::shared_ptr<NodeDef> node_def0, node_def1, node_def2;

  if (p_values.size() == 4) {
    edge_types = p_values[0];
    n = p_values[1];
    m = p_values[2];
    default_node = p_values[3];

    std::vector<std::shared_ptr<NodeDef>> tmp =
        TrivialSampleLayer(edge_types, n, m, default_node,
                           tree_node, default_pre_node_id, *as_table, dag_def);
    node_def0 = tmp[0];
    node_def1 = tmp[1];
    node_def2 = tmp[2];
  } else if (p_values.size() == 5) {
    edge_types = p_values[0];
    n = p_values[1];
    m = p_values[2];
    weight_func = p_values[3];
    default_node = p_values[4];

    std::vector<std::shared_ptr<NodeDef>> tmp =
        GeneralSampleLayer(edge_types, n, m, weight_func, default_node,
                           tree_node, default_pre_node_id, *as_table, dag_def);
    node_def0 = tmp[0];
    node_def1 = tmp[1];
    node_def2 = tmp[2];
  } else {
    EULER_LOG(FATAL) <<
        "layer sampler params error, " <<
        "should be edge_types, n, m, [weight_func], default_node!";
  }

  std::unordered_set<int32_t> pre; std::unordered_set<int32_t> succ;
  /* API_SPARSE_GEN_ADJ */
  std::shared_ptr<NodeDef> node_def3 =
      dag_def->ProduceNodeDef("API_SPARSE_GEN_ADJ", 2);
  node_def3->attrs_.push_back(std::make_shared<NormAttrDef>(n));
  node_def3->input_edges_.push_back(
      {node_def0->name_, node_def0->id_, 0});  // root
  node_def3->input_edges_.push_back(
      {node_def2->name_, node_def2->id_, 0});  // l_nb
  pre.clear(); pre.insert(node_def0->id_); pre.insert(node_def2->id_);
  dag_def->AddNodeDef(node_def3, pre, succ);

  /* API_SPARSE_GET_ADJ */
  std::shared_ptr<NodeDef> node_def4 =
      dag_def->ProduceNodeDef("API_SPARSE_GET_ADJ", 2);
  node_def4->attrs_.push_back(std::make_shared<NormAttrDef>(edge_types));
  node_def4->attrs_.push_back(std::make_shared<NormAttrDef>(m));
  node_def4->input_edges_.push_back(
      {node_def3->name_, node_def3->id_, 0});  // root_batch
  node_def4->input_edges_.push_back(
      {node_def3->name_, node_def3->id_, 1});  // l_nb
  pre.clear(); pre.insert(node_def3->id_);
  dag_def->AddNodeDef(node_def4, pre, succ);

  /* API_GATHER_RESULT */
  std::shared_ptr<NodeDef> node_def5 =
      dag_def->ProduceNodeDef("API_GATHER_RESULT", 3);
  node_def5->input_edges_.push_back(
      {node_def4->name_, node_def4->id_, 0});  // adj_idx
  node_def5->input_edges_.push_back(
      {node_def4->name_, node_def4->id_, 1});  // adj nb id
  node_def5->input_edges_.push_back(
      {node_def2->name_, node_def2->id_, 0});  // l_nb
  pre.clear(); pre.insert(node_def2->id_); pre.insert(node_def4->id_);
  dag_def->AddNodeDef(node_def5, pre, succ);

  // add as node
  node_def5->op_alias_ = tree_node.GetOpAlias();
  if (node_def5->op_alias_ != "") {
    AddAsNode(node_def5.get(), dag_def, as_table);
  }
  return node_def5->id_;
}

void Translator::Translate(const Tree& tree, DAGDef* dag_def) {
  std::vector<TreeNode*> post_sequence;
  tree.PostTraversal(nullptr, &post_sequence);
  // calculate Comprehensive prop
  for (TreeNode* node : post_sequence) {
    if (func_map_.find(node->GetType()) != func_map_.end()) {
      func_map_[node->GetType()](node);
    }
  }
  int32_t default_pre_node_id = -1;
  std::unordered_map<std::string, int32_t> as_table;  // as name, node idx
  for (TreeNode* node : post_sequence) {
    if (build_node_map_.find(node->GetType()) != build_node_map_.end()) {
      BuildNode bn = build_node_map_.at(node->GetType());
      default_pre_node_id = (this->*bn)
          (*node, default_pre_node_id, dag_def, &as_table);
    }
  }
}

}  // namespace euler
