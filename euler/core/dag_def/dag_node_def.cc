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

#include "euler/core/dag_def/dag_node_def.h"

namespace euler {

void NodeDef::ToProto(DAGNodeProto* proto) {
  proto->set_name(ToString(name_, ",", id_));
  proto->set_op(name_);
  proto->set_op_alias(op_alias_);
  for (const EdgeDef& input : input_edges_) {
    if (input.src_id_ != -1) {
      proto->add_inputs(ToString(input.src_name_,
                                 ",",
                                 input.src_id_,
                                 ":",
                                 input.src_slot_));
    } else {
      proto->add_inputs(input.src_name_);
    }
  }
  for (std::shared_ptr<AttrDef> attr_def : attrs_) {
    if (attr_def->attr_type_ == AttrDef::kNorm) {  // put into inputs
      proto->add_inputs(std::static_pointer_cast<NormAttrDef>(attr_def)->
                        attr_key_);
    } else {  // put into cond
      std::shared_ptr<CondAttrDef> cond_attr_def =
          std::static_pointer_cast<CondAttrDef>(attr_def);
      for (auto& and_form_cond : cond_attr_def->dnf_attr_) {
        std::string and_cond;
        for (size_t i = 0; i < and_form_cond.size(); ++i) {
          if (i == 0) {
            and_cond += and_form_cond[i];
          } else {
            and_cond += ("," + and_form_cond[i]);
          }
        }
        proto->add_dnf(and_cond);
      }
      // put into post_process
      for (const std::string& pp : cond_attr_def->post_process_) {
        proto->add_post_process(pp);
      }
    }
  }
  proto->set_output_num(output_num_);
  proto->set_udf_name(udf_name_);
  for (const std::string& str : udf_str_params_) {
    proto->add_udf_str_params(str);
  }
  for (const std::string& num : udf_num_params_) {
    proto->add_udf_num_params(num);
  }
}

void RemoteNodeDef::ToProto(DAGNodeProto* proto) {
  this->NodeDef::ToProto(proto);
  proto->set_shard_idx(shard_idx_);
  // inner node
  for (std::shared_ptr<NodeDef> inner_node : nodes_) {
    DAGNodeProto* inner_proto = proto->add_inner_nodes();
    inner_node->ToProto(inner_proto);
  }
  // output list
  proto->clear_output_list();
  for (const FusionOutput& output : fusion_output_map_) {
    proto->add_output_list(ToString(output.inner_name_,
                                    ",",
                                    output.inner_id_,
                                    ":",
                                    output.inner_output_idx_));
  }
  // remote output list
  for (const FusionOutput& output : fusion_output_map_) {
    proto->add_remote_output_list(ToString(
            name_, ",", id_, ":", output.fusion_output_idx_));
  }
}

}  // namespace euler
