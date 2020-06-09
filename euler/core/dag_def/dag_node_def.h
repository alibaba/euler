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

#ifndef EULER_CORE_DAG_DEF_DAG_NODE_DEF_H_
#define EULER_CORE_DAG_DEF_DAG_NODE_DEF_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <vector>

#include "euler/core/framework/dag.pb.h"
#include "euler/core/framework/attr_value.h"
#include "euler/common/str_util.h"

namespace euler {

struct AttrDef {
  enum Type {
    kNorm,
    kCond
  };

  Type attr_type_;
};

struct NormAttrDef: public AttrDef {
  std::string attr_key_;

  explicit NormAttrDef(const std::string& attr_key) {
    attr_key_ = attr_key;
    attr_type_ = kNorm;
  }
};

struct CondAttrDef: public AttrDef {
  std::vector<std::vector<std::string>> dnf_attr_;
  std::vector<std::string> post_process_;

  CondAttrDef() {
    attr_type_ = kCond;
  }
};

/*this should be bijection*/
struct FusionOutput {
  std::string inner_name_;
  int32_t inner_id_;
  int32_t inner_output_idx_;
  int32_t fusion_output_idx_;

  FusionOutput(const std::string& inner_name, int32_t inner_id,
               int32_t inner_output_idx, int32_t fusion_output_idx) {
    inner_name_ = inner_name;
    inner_id_ = inner_id;
    inner_output_idx_ = inner_output_idx;
    fusion_output_idx_ = fusion_output_idx;
  }

  FusionOutput(const std::vector<std::string>& info,
               const std::unordered_map<int32_t, int32_t>& pattern) {
    inner_name_ = info[0];
    inner_id_ = pattern.at(atoi(info[1].c_str()));
    inner_output_idx_ = atoi(info[2].c_str());
    fusion_output_idx_ = atoi(info[3].c_str());
  }
};

struct SplitOpInfo {
  std::string split_op_name_;
  std::vector<int32_t> inputs_idx_;

  SplitOpInfo(const std::string& split_op_name,
              const std::vector<int32_t>& inputs_idx) {
    split_op_name_ = split_op_name;
    inputs_idx_ = inputs_idx;
  }

  explicit SplitOpInfo(const std::vector<std::string>& info) {
    split_op_name_ = info[0];
    std::vector<std::string> inputs_info = Split(info[1], ',');
    for (const std::string& input_info : inputs_info) {
      inputs_idx_.push_back(atoi(input_info.c_str()));
    }
  }
};

struct MergeOpInfo {
  std::string merge_op_name_;
  std::string merge_idx_op_type_;
  int32_t merge_idx_op_idx_;
  std::vector<int32_t> inputs_idx_;

  MergeOpInfo(const std::string& merge_op_name,
              std::string merge_idx_op_type,
              int32_t merge_idx_op_idx,
              const std::vector<int32_t>& inputs_idx) {
    merge_op_name_ = merge_op_name;
    merge_idx_op_type_ = merge_idx_op_type;
    merge_idx_op_idx_ = merge_idx_op_idx;
    inputs_idx_ = inputs_idx;
  }

  explicit MergeOpInfo(const std::vector<std::string>& info) {
    merge_op_name_ = info[0];
    std::vector<std::string> merge_idx_info = Split(info[1], ':');
    merge_idx_op_type_ = merge_idx_info[0];
    merge_idx_op_idx_ = atoi(merge_idx_info[1].c_str());
    std::vector<std::string> inputs_info = Split(info[2], ',');
    for (const std::string& input_info : inputs_info) {
      inputs_idx_.push_back(atoi(input_info.c_str()));
    }
  }
};

struct UniqueOpInfo {
  std::string unique_op_name_;
  std::vector<int32_t> inputs_idx_;

  explicit UniqueOpInfo(const std::vector<std::string>& info) {
    unique_op_name_ = info[0];
    std::vector<std::string> input_idxs_info = Split(info[1], ',');
    for (const std::string& s : input_idxs_info) {
      inputs_idx_.push_back(atoi(s.c_str()));
    }
  }
};

struct GatherOpInfo {
  std::string gather_op_name_;
  int32_t unique_op_idx_;
  std::vector<int32_t> inputs_idx_;

  explicit GatherOpInfo(const std::vector<std::string>& info) {
    gather_op_name_ = info[0];
    unique_op_idx_ = atoi(info[1].c_str());
    std::vector<std::string> input_idxs_info = Split(info[2], ',');
    for (const std::string& s : input_idxs_info) {
      inputs_idx_.push_back(atoi(s.c_str()));
    }
  }
};

struct EdgeDef {
  std::string src_name_;
  int32_t src_id_;
  int32_t src_slot_;
};

struct NodeDef {
  std::string name_;
  std::string op_alias_;
  int32_t id_;
  std::vector<std::shared_ptr<AttrDef>> attrs_;
  std::vector<EdgeDef> input_edges_;

  std::unordered_set<int32_t> pre_;
  std::unordered_set<int32_t> succ_;
  int32_t output_num_;

  std::string udf_name_;
  std::vector<std::string> udf_str_params_;
  std::vector<std::string> udf_num_params_;

  NodeDef() {
  }

  NodeDef(const std::string& name,
          int32_t id) {
    name_ = name;
    id_ = id;
    output_num_ = 1;
    op_alias_ = "";
  }

  NodeDef(const std::string& name,
          int32_t id,
          int32_t output_num) {
    name_ = name;
    id_ = id;
    output_num_ = output_num;
    op_alias_ = "";
  }

  virtual void ToProto(DAGNodeProto* proto);
};

struct RemoteNodeDef: public NodeDef {
  int32_t shard_idx_;
  std::vector<std::shared_ptr<NodeDef>> nodes_;  // inner sub_nodes
  std::vector<FusionOutput> fusion_output_map_;

  RemoteNodeDef(const std::string& name,
                int32_t id,
                int32_t shard_idx,
                const std::vector<std::shared_ptr<NodeDef>>& nodes,
                const std::vector<FusionOutput>& fusion_output_map,
                int32_t output_num) {
    name_ = name;
    id_ = id;
    shard_idx_ = shard_idx;
    nodes_ = nodes;
    fusion_output_map_ = fusion_output_map;
    output_num_ = output_num;
  }

  virtual void ToProto(DAGNodeProto* proto);
};

}  // namespace euler
#endif  // EULER_CORE_DAG_DEF_DAG_NODE_DEF_H_
