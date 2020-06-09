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

#include <string>

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/common/logging.h"
#include "euler/core/api/api.h"
#include "euler/core/kernels/common.h"
#include "euler/common/compact_weighted_collection.h"

namespace euler {

class SampleNWithTypesOp: public OpKernel {
 public:
  explicit SampleNWithTypesOp(const std::string& name): OpKernel(name) { }
  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
};

void SampleNWithTypesOp::Compute(const DAGNodeProto& node_def,
                                 OpKernelContext* ctx) {
  /* get inputs */
  Tensor* node_types_t = nullptr;
  Tensor* counts_t = nullptr;
  ctx->tensor(node_def.inputs(0), &node_types_t);
  ctx->tensor(node_def.inputs(1), &counts_t);
  size_t types_num = node_types_t->NumElements();

  /* sample */
  size_t total_cnt = 0;
  std::vector<NodeIdVec> node_id_vec_list(types_num);
  for (size_t i = 0; i < types_num; ++i) {
    int32_t type = node_types_t->Raw<int32_t>()[i];
    int32_t count = counts_t->Raw<int32_t>()[i];
    node_id_vec_list[i] = SampleNode({type}, count);
    total_cnt += count;
  }

  /* output */
  TensorShape idx_shape({types_num, 2});
  TensorShape data_shape({total_cnt});
  std::string output_idx_name = OutputName(node_def, 0);
  std::string output_data_name = OutputName(node_def, 1);
  Tensor* idx_t = nullptr;
  Tensor* data_t = nullptr;
  ctx->Allocate(output_idx_name, idx_shape, DataType::kInt32, &idx_t);
  ctx->Allocate(output_data_name, data_shape, DataType::kUInt64, &data_t);
  int32_t base_addr = 0;
  for (size_t i = 0; i < types_num; ++i) {
    int32_t seg_begin = base_addr;
    int32_t seg_end = base_addr + node_id_vec_list[i].size();
    idx_t->Raw<int32_t>()[i * 2] = seg_begin;
    idx_t->Raw<int32_t>()[i * 2 + 1] = seg_end;
    std::copy(node_id_vec_list[i].begin(), node_id_vec_list[i].end(),
              data_t->Raw<uint64_t>() + base_addr);
    base_addr = seg_end;
  }
}

REGISTER_OP_KERNEL("API_SAMPLE_N_WITH_TYPES", SampleNWithTypesOp);

}  // namespace euler
