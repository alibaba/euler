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

#ifndef EULER_PARSER_ATTRIBUTE_CALCULATOR_H_
#define EULER_PARSER_ATTRIBUTE_CALCULATOR_H_

#include "euler/parser/tree.h"

namespace euler {

bool SimpleCondition(TreeNode* t);
bool HasKey(TreeNode* t);
bool HasLabel(TreeNode* t);
bool Has(TreeNode* t);
bool Term(TreeNode* t);
bool CONJ(TreeNode* t);
bool DNF(TreeNode* t);
bool Limit(TreeNode* t);
bool OrderBy(TreeNode* t);
bool As(TreeNode* t);
bool PostProcess(TreeNode* t);
bool Condtition(TreeNode* t);
bool Params(TreeNode* t);
bool Va(TreeNode* t);
bool SampleNB(TreeNode* t);
bool SampleLNB(TreeNode* t);
bool SampleEdge(TreeNode* t);
bool SampleNode(TreeNode* t);
bool SampleNWithTypes(TreeNode* t);
bool E(TreeNode* t);
bool V(TreeNode* t);
bool APISampleNB(TreeNode* t);
bool APISampleLNB(TreeNode* t);
bool APIGetNBEdge(TreeNode* t);
bool APIGetRNBNode(TreeNode* t);
bool APIGetNBNode(TreeNode* t);
bool APIGetNodeT(TreeNode* t);
bool APIGetP(TreeNode* t);
bool APISampleEdge(TreeNode* t);
bool APIGetEdge(TreeNode* t);
bool APISampleNode(TreeNode* t);
bool APISampleNWithTypes(TreeNode* t);
bool APIGetNode(TreeNode* t);
bool Select(TreeNode* t);

}  // namespace euler

#endif  // EULER_PARSER_ATTRIBUTE_CALCULATOR_H_
