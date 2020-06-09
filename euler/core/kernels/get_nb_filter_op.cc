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

#include <vector>
#include <string>
#include <unordered_set>

#include "euler/common/logging.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/common/str_util.h"
#include "euler/core/kernels/common.h"

namespace euler {

class GetNBFilter: public OpKernel {
 public:
  explicit GetNBFilter(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void GetNBFilter::Compute(const DAGNodeProto& node_def,
                          OpKernelContext* ctx) {
  Tensor* nb_idx_t = nullptr;  // int32
  Tensor* nb_id_t = nullptr;  // uint64
  Tensor* nb_w_t = nullptr;  // float
  Tensor* nb_t_t = nullptr;  // int32
  Tensor* filtered_node_t = nullptr;  // uint64
  ctx->tensor(node_def.inputs(0), &nb_idx_t);
  ctx->tensor(node_def.inputs(1), &nb_id_t);
  ctx->tensor(node_def.inputs(2), &nb_w_t);
  ctx->tensor(node_def.inputs(3), &nb_t_t);
  ctx->tensor(node_def.inputs(4), &filtered_node_t);

  // post process cmds
  std::vector<std::vector<std::string>> pp_cmds;
  for (const std::string& pp : node_def.post_process()) {
    pp_cmds.push_back(Split(pp, " "));
  }

  std::unordered_set<uint64_t> filter;
  for (int32_t i = 0; i < filtered_node_t->NumElements(); ++i) {
    filter.insert(filtered_node_t->Raw<uint64_t>()[i]);
  }

  size_t root_num = nb_idx_t->Shape().Dims()[0];
  std::vector<std::vector<IdWeightType>> tmp_results(root_num);

  for (size_t i = 0; i < root_num; ++i) {
    int32_t old_b = nb_idx_t->Raw<int32_t>()[i * 2];
    int32_t old_e = nb_idx_t->Raw<int32_t>()[i * 2 + 1];
    tmp_results[i].reserve(old_e - old_b);
    for (int32_t j = old_b; j < old_e; ++j) {
      if (filter.find(nb_id_t->Raw<uint64_t>()[j]) != filter.end()) {
        tmp_results[i].push_back({nb_id_t->Raw<uint64_t>()[j],
                                  nb_w_t->Raw<float>()[j],
                                  nb_t_t->Raw<int32_t>()[j]});
      }
    }
  }

  size_t data_num = 0;
  for (size_t i = 0; i < root_num; ++i) {
    for (const std::vector<std::string>& pp : pp_cmds) {
      if (pp[0] == "order_by" && pp[1] == "id" && pp[2] == "asc") {
        std::sort(tmp_results[i].begin(), tmp_results[i].end(),
                  [](IdWeightType a, IdWeightType b) {return a.id_ < b.id_;});
      } else if (pp[0] == "order_by" && pp[1] == "id" && pp[2] == "desc") {
        std::sort(tmp_results[i].begin(), tmp_results[i].end(),
                  [](IdWeightType a, IdWeightType b) {return a.id_ > b.id_;});
      } else if (pp[0] == "order_by" && pp[1] == "weight" && pp[2] == "asc") {
        std::sort(tmp_results[i].begin(), tmp_results[i].end(),
                  [](IdWeightType a, IdWeightType b) {return a.w_ < b.w_;});
      } else if (pp[0] == "order_by" && pp[1] == "weight" && pp[2] == "desc") {
        std::sort(tmp_results[i].begin(), tmp_results[i].end(),
                  [](IdWeightType a, IdWeightType b) {return a.w_ > b.w_;});
      } else {  // limit
        size_t new_size = static_cast<size_t>(atoi(pp[1].c_str()));
        if (new_size < tmp_results[i].size()) {
          tmp_results[i].resize(new_size);
        }
      }
    }
    data_num += tmp_results[i].size();
  }

  // output
  Tensor* o_idx_t = nullptr;
  Tensor* o_id_t = nullptr;
  Tensor* o_w_t = nullptr;
  Tensor* o_t_t = nullptr;
  TensorShape idx_shape = nb_idx_t->Shape();
  TensorShape data_shape({data_num});
  ctx->Allocate(OutputName(node_def, 0), idx_shape, DataType::kInt32, &o_idx_t);
  ctx->Allocate(OutputName(node_def, 1), data_shape,
                DataType::kUInt64, &o_id_t);
  ctx->Allocate(OutputName(node_def, 2), data_shape, DataType::kFloat, &o_w_t);
  ctx->Allocate(OutputName(node_def, 3), data_shape, DataType::kInt32, &o_t_t);

  int32_t addr = 0;
  for (size_t i = 0; i < root_num; ++i) {
    o_idx_t->Raw<int32_t>()[i * 2] = addr;
    for (size_t j = 0; j < tmp_results[i].size(); ++j) {
      o_id_t->Raw<uint64_t>()[addr] = tmp_results[i][j].id_;
      o_w_t->Raw<float>()[addr] = tmp_results[i][j].w_;
      o_t_t->Raw<int32_t>()[addr] = tmp_results[i][j].t_;
      ++addr;
    }
    o_idx_t->Raw<int32_t>()[i * 2 + 1] = addr;
  }
}

REGISTER_OP_KERNEL("API_GET_NB_FILTER", GetNBFilter);

}  // namespace euler
