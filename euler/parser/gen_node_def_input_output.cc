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

#include "euler/parser/gen_node_def_input_output.h"

namespace euler {

/* input */
// each output in pre nodes corresponds an input
#define NORMAL_INPUT_GEN() {                                          \
  for (int32_t i = 0; i < pre_node.output_num_; ++i) {                \
    node->input_edges_.push_back({pre_node.name_, pre_node.id_, i});  \
  }                                                                   \
}

// some attr need be treated as input tensors
#define BEGIN_INPUT_GEN() {                                           \
  for (auto it = node->attrs_.begin();                                \
       it != node->attrs_.end();) {                                   \
    if ((*it)->attr_type_ == AttrDef::kNorm) {                        \
      node->input_edges_.push_back(                                   \
          {std::static_pointer_cast<NormAttrDef>(*it)->attr_key_,     \
          -1, -1});                                                   \
      it = node->attrs_.erase(it);                                    \
    } else {                                                          \
      ++it;                                                           \
    }                                                                 \
  }                                                                   \
}

void SampleNBInputs(const NodeDef& pre_node, NodeDef* node) {
  if (pre_node.name_ == "API_SAMPLE_NB" ||
      pre_node.name_ == "API_GATHER_RESULT" ||
      pre_node.name_ == "API_GET_RNB_NODE" ||
      pre_node.name_ == "API_GET_NB_NODE" ||
      pre_node.name_ == "API_GET_NB_FILTER" ||
      pre_node.name_ == "API_SAMPLE_N_WITH_TYPES") {
    node->input_edges_.push_back({pre_node.name_, pre_node.id_, 1});
  } else {
    NORMAL_INPUT_GEN();
  }
}

void GetNBEdgeInputs(const NodeDef& pre_node, NodeDef* node) {
  if (pre_node.name_ == "API_SAMPLE_NB" ||
      pre_node.name_ == "API_GATHER_RESULT" ||
      pre_node.name_ == "API_GET_RNB_NODE" ||
      pre_node.name_ == "API_GET_NB_NODE" ||
      pre_node.name_ == "API_GET_NB_FILTER" ||
      pre_node.name_ == "API_SAMPLE_N_WITH_TYPES") {
    node->input_edges_.push_back({pre_node.name_, pre_node.id_, 1});
  } else {
    NORMAL_INPUT_GEN();
  }
}

void GetRNBNodeInputs(const NodeDef& pre_node, NodeDef* node) {
  if (pre_node.name_ == "API_SAMPLE_NB" ||
      pre_node.name_ == "API_GATHER_RESULT" ||
      pre_node.name_ == "API_GET_RNB_NODE" ||
      pre_node.name_ == "API_GET_NB_NODE" ||
      pre_node.name_ == "API_GET_NB_FILTER" ||
      pre_node.name_ == "API_SAMPLE_N_WITH_TYPES") {
    node->input_edges_.push_back({pre_node.name_, pre_node.id_, 1});
  } else {
    NORMAL_INPUT_GEN();
  }
}

void GetNBNodeInputs(const NodeDef& pre_node, NodeDef* node) {
  if (pre_node.name_ == "API_SAMPLE_NB" ||
      pre_node.name_ == "API_GATHER_RESULT" ||
      pre_node.name_ == "API_GET_RNB_NODE" ||
      pre_node.name_ == "API_GET_NB_NODE" ||
      pre_node.name_ == "API_GET_NB_FILTER" ||
      pre_node.name_ == "API_SAMPLE_N_WITH_TYPES") {
    node->input_edges_.push_back({pre_node.name_, pre_node.id_, 1});
  } else {
    NORMAL_INPUT_GEN();
  }
}

void GetNodeTInputs(const NodeDef& pre_node, NodeDef* node) {
  if (pre_node.name_ == "API_SAMPLE_NB" ||
      pre_node.name_ == "API_GATHER_RESULT" ||
      pre_node.name_ == "API_GET_RNB_NODE" ||
      pre_node.name_ == "API_GET_NB_NODE" ||
      pre_node.name_ == "API_GET_NB_FILTER" ||
      pre_node.name_ == "API_SAMPLE_N_WITH_TYPES") {
    node->input_edges_.push_back({pre_node.name_, pre_node.id_, 1});
  } else {
    NORMAL_INPUT_GEN();
  }
}

void GetPInputs(const NodeDef& pre_node, NodeDef* node) {
  if (pre_node.name_ == "API_SAMPLE_NB" ||
      pre_node.name_ == "API_GATHER_RESULT" ||
      pre_node.name_ == "API_GET_RNB_NODE" ||
      pre_node.name_ == "API_GET_NB_NODE" ||
      pre_node.name_ == "API_GET_NB_EDGE" ||
      pre_node.name_ == "API_GET_NB_FILTER" ||
      pre_node.name_ == "API_SAMPLE_N_WITH_TYPES") {
    node->input_edges_.push_back({pre_node.name_, pre_node.id_, 1});
  } else {
    NORMAL_INPUT_GEN();
  }
}

void SampleEdgeInputs(const NodeDef& pre_node, NodeDef* node) {
  (void) pre_node;
  BEGIN_INPUT_GEN();
}

void GetEdgeInputs(const NodeDef& pre_node, NodeDef* node) {
  (void) pre_node;
  BEGIN_INPUT_GEN();
}

void SampleNodeInputs(const NodeDef& pre_node, NodeDef* node) {
  (void) pre_node;
  BEGIN_INPUT_GEN();
}

void SampleNWithTypesInputs(const NodeDef& pre_node, NodeDef* node) {
  (void) pre_node;
  BEGIN_INPUT_GEN();
}

void GetNodeInputs(const NodeDef& pre_node, NodeDef* node) {
  (void) pre_node;
  BEGIN_INPUT_GEN();
}

/* output */
int32_t SampleNBOutputNum(const NodeDef& node_def) {
  (void) node_def;
  return 4;
}

int32_t GetNBEdgeOutputNum(const NodeDef& node_def) {
  (void) node_def;
  return 3;
}

int32_t GetRNBNodeOutputNum(const NodeDef& node_def) {
  (void) node_def;
  return 4;
}

int32_t GetNBNodeOutputNum(const NodeDef& node_def) {
  (void) node_def;
  return 4;
}

int32_t GetNodeTOutputNum(const NodeDef& node_def) {
  (void) node_def;
  return 1;
}

int32_t GetPOutputNum(const NodeDef& node_def) {
  int32_t cnt = 0;
  for (std::shared_ptr<AttrDef> attr : node_def.attrs_) {
    if (attr->attr_type_ == AttrDef::kNorm) {
      ++cnt;
    }
  }
  return cnt * 2;
}

int32_t SampleEdgeOutputNum(const NodeDef& node_def) {
  (void) node_def;
  return 1;
}

int32_t GetEdgeOutputNum(const NodeDef& node_def) {
  (void) node_def;
  return 1;
}

int32_t SampleNodeOutputNum(const NodeDef& node_def) {
  (void) node_def;
  return 1;
}

int32_t SampleNWithTypesOutputNum(const NodeDef& node_def) {
  (void) node_def;
  return 2;
}

int32_t GetNodeOutputNum(const NodeDef& node_def) {
  (void) node_def;
  return 1;
}

}  // namespace euler
