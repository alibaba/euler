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

#ifndef EULER_PARSER_GEN_NODE_DEF_INPUT_OUTPUT_H_
#define EULER_PARSER_GEN_NODE_DEF_INPUT_OUTPUT_H_

#include "euler/core/dag_def/dag_def.h"

namespace euler {

void SampleNBInputs(const NodeDef& pre_node, NodeDef* node);
void GetNBEdgeInputs(const NodeDef& pre_node, NodeDef* node);
void GetRNBNodeInputs(const NodeDef& pre_node, NodeDef* node);
void GetNBNodeInputs(const NodeDef& pre_node, NodeDef* node);
void GetNodeTInputs(const NodeDef& pre_node, NodeDef* node);
void GetPInputs(const NodeDef& pre_node, NodeDef* node);
void SampleEdgeInputs(const NodeDef& pre_node, NodeDef* node);
void GetEdgeInputs(const NodeDef& pre_node, NodeDef* node);
void SampleNodeInputs(const NodeDef& pre_node, NodeDef* node);
void SampleNWithTypesInputs(const NodeDef& pre_node, NodeDef* node);
void GetNodeInputs(const NodeDef& pre_node, NodeDef* node);

int32_t SampleNBOutputNum(const NodeDef& node_def);
int32_t GetNBEdgeOutputNum(const NodeDef& node_def);
int32_t GetRNBNodeOutputNum(const NodeDef& node_def);
int32_t GetNBNodeOutputNum(const NodeDef& node_def);
int32_t GetNodeTOutputNum(const NodeDef& node_def);
int32_t GetPOutputNum(const NodeDef& node_def);
int32_t SampleEdgeOutputNum(const NodeDef& node_def);
int32_t GetEdgeOutputNum(const NodeDef& node_def);
int32_t SampleNodeOutputNum(const NodeDef& node_def);
int32_t SampleNWithTypesOutputNum(const NodeDef& node_def);
int32_t GetNodeOutputNum(const NodeDef& node_def);

}  // namespace euler
#endif  // EULER_PARSER_GEN_NODE_DEF_INPUT_OUTPUT_H_
