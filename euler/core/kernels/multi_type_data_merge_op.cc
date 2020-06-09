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

class MultiTypeDataMerge: public OpKernel {
 public:
  explicit MultiTypeDataMerge(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void MultiTypeDataMerge::Compute(const DAGNodeProto& node_def,
                                 OpKernelContext* ctx) {
  /* get input tensor and merge index */
  std::vector<Tensor*> datas;
  std::vector<Tensor*> idxs;
  size_t types_num = 0;
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    Tensor* t = nullptr;
    if (node_def.inputs(i) != "") ctx->tensor(node_def.inputs(i), &t);
    if (i % 3 == 0) {  // data
      datas.push_back(t);
    } else if (i % 3 == 1) {  // idx
      idxs.push_back(t);
      types_num = t->Shape().Dims()[0];
    }
  }
  /* output */
  size_t node_cnt = 0;
  for (size_t i = 0; i < datas.size(); ++i) {
    node_cnt += datas[i]->NumElements();
  }
  std::string output_name = OutputName(node_def, 0);
  Tensor* data_result = nullptr;
  TensorShape data_shape({node_cnt});
  ctx->Allocate(output_name, data_shape, DataType::kUInt64, &data_result);
  int32_t base_addr = 0;
  for (size_t i = 0; i < types_num; ++i) {  // each type
    for (size_t j = 0; j < datas.size(); ++j) {  // each shard
      int32_t seg_begin = idxs[j]->Raw<int32_t>()[i * 2];
      int32_t seg_end = idxs[j]->Raw<int32_t>()[i * 2 + 1];
      std::copy(datas[j]->Raw<uint64_t>() + seg_begin,
                datas[j]->Raw<uint64_t>() + seg_end,
                data_result->Raw<uint64_t>() + base_addr);
      base_addr += seg_end - seg_begin;
    }
  }
}

REGISTER_OP_KERNEL("MULTI_TYPE_DATA_MERGE", MultiTypeDataMerge);

}  // namespace euler
