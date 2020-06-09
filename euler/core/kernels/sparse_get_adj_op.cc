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

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/core/api/api.h"
#include "euler/common/str_util.h"

namespace euler {

class SparseGetAdj: public OpKernel {
 public:
  explicit SparseGetAdj(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void SparseGetAdj::Compute(const DAGNodeProto& node_def,
                           OpKernelContext* ctx) {
  Tensor* root_batch_t = nullptr;
  Tensor* l_nb_t = nullptr;
  Tensor* edge_types_t = nullptr;
  Tensor* m_t = nullptr;
  ctx->tensor(node_def.inputs(0), &root_batch_t);
  ctx->tensor(node_def.inputs(1), &l_nb_t);
  ctx->tensor(node_def.inputs(2), &edge_types_t);
  ctx->tensor(node_def.inputs(3), &m_t);

  std::vector<int32_t> edge_types(edge_types_t->NumElements());
  std::copy(edge_types_t->Raw<int32_t>(),
            edge_types_t->Raw<int32_t>() + edge_types.size(),
            edge_types.begin());
  int32_t m = m_t->Raw<int32_t>()[0];
  size_t root_num = root_batch_t->NumElements() / 2;

  size_t total_cnt = 0;
  std::vector<std::vector<uint64_t>> adj(root_num);
  for (size_t i = 0; i < root_num; ++i) {
    uint64_t root_id = root_batch_t->Raw<uint64_t>()[i * 2];
    int32_t batch_num = static_cast<int32_t>(
        root_batch_t->Raw<uint64_t>()[i * 2 + 1]);
    int32_t l_nb_batch_begin = batch_num * m;
    adj[i].reserve(10);
    for (int32_t j = 0; j < m; ++j) {
      uint64_t nb_id = l_nb_t->Raw<uint64_t>()[l_nb_batch_begin + j];
      bool exist = false;
      for (int32_t e_type : edge_types) {
        EdgeId eid(root_id, nb_id, e_type);
        exist = exist || EdgeExist(eid);
      }
      if (exist) {
        adj[i].push_back(nb_id);
        ++total_cnt;
      }
    }
  }

  Tensor* sp_adj_idx = nullptr;
  Tensor* sp_adj_data = nullptr;
  TensorShape idx_shape({root_num, 2});
  TensorShape data_shape({total_cnt});
  ctx->Allocate(OutputName(node_def, 0), idx_shape,
                DataType::kInt32, &sp_adj_idx);
  ctx->Allocate(OutputName(node_def, 1), data_shape,
                DataType::kUInt64, &sp_adj_data);
  int32_t offset = 0;
  for (size_t i = 0; i < root_num; ++i) {
    sp_adj_idx->Raw<int32_t>()[i * 2] = offset;
    sp_adj_idx->Raw<int32_t>()[i * 2 + 1] = offset + adj[i].size();
    std::copy(adj[i].begin(), adj[i].end(),
              sp_adj_data->Raw<uint64_t>() + offset);
    offset += adj[i].size();
  }
}

REGISTER_OP_KERNEL("API_SPARSE_GET_ADJ", SparseGetAdj);

}  // namespace euler
