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

#include "gtest/gtest.h"

#include "euler/parser/translator.h"
#include "euler/parser/tree.h"
#include "euler/core/dag_def/dag_def.h"

namespace euler {

TEST(TranslatorTest, Translator) {
  Translator translator(distribute);
  std::string gremlin = "sampleN(node_type, cnt).has(p0 eq 23)";
  Tree tree = BuildGrammarTree(gremlin);

  DAGDef dag_def;
  translator.Translate(tree, &dag_def);
  std::vector<int32_t> result = dag_def.TopologicSort();
  for (int32_t i : result) {
    std::shared_ptr<NodeDef> node_def = dag_def.GetNodeById(i);
    std::cout << node_def->name_ << "," << node_def->id_ << std::endl;
    // cout attr
    for (std::shared_ptr<AttrDef> attr : node_def->attrs_) {
      if (attr->attr_type_ == AttrDef::kNorm) {
        std::cout <<
            std::static_pointer_cast<NormAttrDef>(attr)->attr_key_ <<
            std::endl;
      } else {
        std::cout << "[" << std::endl;
        for (size_t i = 0;
             i < std::static_pointer_cast<CondAttrDef>(attr)->
             dnf_attr_.size();
             ++i) {
          for (size_t j = 0;
               j < std::static_pointer_cast<CondAttrDef>(attr)->
               dnf_attr_[i].size();
               ++j) {
            std::cout <<
                std::static_pointer_cast<CondAttrDef>(attr)->
                dnf_attr_[i][j] << ", ";
          }
          std::cout << std::endl;
        }
        std::cout << "]" << std::endl;
        std::cout << "(" << std::endl;
        for (size_t i = 0;
             i < std::static_pointer_cast<CondAttrDef>(attr)->
             post_process_.size();
             ++i) {
          std::cout << std::static_pointer_cast<CondAttrDef>(attr)->
              post_process_[i] << std::endl;
        }
        std::cout << ")" << std::endl;
      }
    }
    // cout topo
    std::cout << "topology:" << std::endl;
    std::cout << "pre:";
    for (int32_t p : node_def->pre_) {
      std::cout << p << " ";
    }
    std::cout << " succ:";
    for (int32_t s : node_def->succ_) {
      std::cout << s << " ";
    }
    std::cout << "\ninput_edges: " << std::endl;
    for (EdgeDef ed : node_def->input_edges_) {
      std::cout
          << ed.src_name_ << ","
          << ed.src_id_ << ","
          << ed.src_slot_ << std::endl;
    }
    std::cout << "\nalias: " << node_def->op_alias_ << std::endl;
    std::cout << "-----------------" << std::endl;
  }
}

}  // namespace euler
