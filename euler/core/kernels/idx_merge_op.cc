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
#include "euler/common/str_util.h"

namespace euler {
class IdxMerge: public OpKernel {
 public:
  explicit IdxMerge(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void IdxMerge::Compute(const DAGNodeProto& node_def,
                       OpKernelContext* ctx) {
  /* get input tensor and merge index */
  std::vector<Tensor*> idxs;
  std::vector<Tensor*> merge_idxs;
  size_t ids_num = 0;
  int32_t offset = 1;
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    Tensor* t = nullptr;
    ctx->tensor(node_def.inputs(i), &t);
    if (i % 2 == 0) {  // id
      idxs.push_back(t);
      ids_num += t->Shape().Dims()[0];
    } else {  // merge_idx
      merge_idxs.push_back(t);
    }
  }
  for (size_t i = 1; i < idxs[0]->Shape().Dims().size(); ++i) {
    offset *= idxs[0]->Shape().Dims()[i];
  }

  /* merge output result */
  std::string output_name = OutputName(node_def, 0);
  Tensor* idx_result = nullptr;
  std::vector<size_t> idx_shape_v = idxs[0]->Shape().Dims();
  idx_shape_v[0] = ids_num;
  TensorShape idx_shape(idx_shape_v);
  ctx->Allocate(output_name, idx_shape, DataType::kInt32, &idx_result);

  for (size_t i = 0; i < idxs.size(); ++i) {
    Tensor* merge_t = merge_idxs[i];
    for (size_t j = 0; j < idxs[i]->Shape().Dims()[0]; ++j) {
      int32_t result_addr = merge_t->Raw<int32_t>()[j] * offset;
      std::copy(idxs[i]->Raw<int32_t>() + j * offset,
                idxs[i]->Raw<int32_t>() + (j + 1) * offset,
                idx_result->Raw<int32_t>() + result_addr);
    }
  }
  int32_t base_addr = 0;
  for (int32_t i = 0; i < idx_result->NumElements(); i += 2) {
    int32_t offset = idx_result->Raw<int32_t>()[i + 1] -
                     idx_result->Raw<int32_t>()[i];
    idx_result->Raw<int32_t>()[i] = base_addr;
    idx_result->Raw<int32_t>()[i + 1] = offset + base_addr;
    base_addr = idx_result->Raw<int32_t>()[i + 1];
  }
}

class GPIdxMerge: public OpKernel {
 public:
  explicit GPIdxMerge(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void GPIdxMerge::Compute(const DAGNodeProto& node_def,
                         OpKernelContext* ctx) {
  /* get input tensor and merge index */
  std::vector<Tensor*> idxs;
  std::vector<Tensor*> merge_idxs;
  size_t ids_num = 0;
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    Tensor* t = nullptr;
    ctx->tensor(node_def.inputs(i), &t);
    if (i % 2 == 0) {  // id
      idxs.push_back(t);
      ids_num += t->Shape().Dims()[0];
    } else {  // merge_idx
      merge_idxs.push_back(t);
    }
  }

  size_t shard_num = idxs.size();
  std::unordered_map<int32_t, int32_t> merge_idx2size(ids_num);
  for (size_t i = 0; i < shard_num; ++i) {
    for (int32_t j = 0; j < merge_idxs[i]->NumElements(); ++j) {
      int32_t merge_idx = merge_idxs[i]->Raw<int32_t>()[j];
      int32_t size = idxs[i]->Raw<int32_t>()[j * 2 + 1] -
          idxs[i]->Raw<int32_t>()[j * 2];
      if (merge_idx2size.find(merge_idx) == merge_idx2size.end() ||
          merge_idx2size.at(merge_idx) < size) {
        merge_idx2size[merge_idx] = size;
      }
    }
  }
  ids_num = merge_idx2size.size();

  /* merge output result */
  std::string output_name = OutputName(node_def, 0);
  Tensor* idx_result = nullptr;
  TensorShape idx_shape({ids_num, 2});
  ctx->Allocate(output_name, idx_shape, DataType::kInt32, &idx_result);
  for (size_t i = 0; i < idxs.size(); ++i) {
    Tensor* merge_t = merge_idxs[i];
    for (size_t j = 0; j < idxs[i]->Shape().Dims()[0]; ++j) {
      int32_t result_addr = merge_t->Raw<int32_t>()[j] * 2;
      int32_t result_size = merge_idx2size.at(merge_t->Raw<int32_t>()[j]);
      idx_result->Raw<int32_t>()[result_addr] = 0;
      idx_result->Raw<int32_t>()[result_addr + 1] = result_size;
    }
  }
  int32_t base_addr = 0;
  for (int32_t i = 0; i < idx_result->NumElements(); i += 2) {
    int32_t offset = idx_result->Raw<int32_t>()[i + 1] -
                     idx_result->Raw<int32_t>()[i];
    idx_result->Raw<int32_t>()[i] = base_addr;
    idx_result->Raw<int32_t>()[i + 1] = offset + base_addr;
    base_addr = idx_result->Raw<int32_t>()[i + 1];
  }
}

REGISTER_OP_KERNEL("IDX_MERGE", IdxMerge);
REGISTER_OP_KERNEL("GP_IDX_MERGE", GPIdxMerge);

}  // namespace euler
