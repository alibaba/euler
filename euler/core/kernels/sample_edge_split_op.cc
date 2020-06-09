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

#include <math.h>

#include <vector>

#include "euler/common/logging.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/client/client_manager.h"
#include "euler/client/query_proxy.h"
#include "euler/common/str_util.h"
#include "euler/common/random.h"

namespace euler {

class SampleEdgeSplit: public OpKernel {
 public:
  explicit SampleEdgeSplit(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void SampleEdgeSplit::Compute(const DAGNodeProto& node_def,
                              OpKernelContext* ctx) {
  // get split num
  int32_t split_num = QueryProxy::GetInstance()->GetShardNum();
  // get count tensor
  Tensor* count_tensor = nullptr;
  ctx->tensor(node_def.inputs(0), &count_tensor);
  int32_t count = count_tensor->Raw<int32_t>()[0];
  // get edge_type tensor
  Tensor* edge_type_tensor = nullptr;
  ctx->tensor(node_def.inputs(1), &edge_type_tensor);
  std::vector<int32_t> edge_types(edge_type_tensor->NumElements());
  std::copy(edge_type_tensor->Raw<int32_t>(),
            edge_type_tensor->Raw<int32_t>() + edge_types.size(),
            edge_types.begin());

  const std::vector<std::vector<float>>& shard_edge_weight =
      QueryProxy::GetInstance()->GetShardEdgeWeight();

  std::vector<int32_t> split_cnt(split_num);
  int32_t remain_cnt = count;
  std::vector<size_t> no_zero_idxs; no_zero_idxs.reserve(split_num);
  for (int32_t i = 0; i < split_num; ++i) {
    float sum_weight0 = 0, sum_weight1 = 0;
    for (int32_t edge_type : edge_types) {
      if (edge_type == -1) {
        edge_type = shard_edge_weight.size() - 1;
        if (edge_types.size() > 1) {
          EULER_LOG(FATAL) << "ERROR edge types!";
        }
      }
      sum_weight0 += shard_edge_weight[edge_type][i];
      sum_weight1 += shard_edge_weight[edge_type][split_num];
    }
    if (fabs(sum_weight1 - 0.0) < 0.0000001) {
      EULER_LOG(FATAL) << "edge type sum weight is zero!";
    }
    split_cnt[i] = floor(count * sum_weight0 / sum_weight1);
    if (sum_weight0 > 0) {
      no_zero_idxs.push_back(i);
    }
    remain_cnt -= split_cnt[i];
  }

  for (size_t i = 0; remain_cnt > 0; ++i, --remain_cnt) {
    size_t no_zero_idx = no_zero_idxs[
        floor(common::ThreadLocalRandom() * no_zero_idxs.size())];
    split_cnt[no_zero_idx] += 1;
  }

  /* output */
  int32_t output_idx = 0;
  int32_t begin_id = 0;
  for (int32_t i = 0; i < split_num; ++i) {
    // output split intput
    std::string split_count_out_name =
        OutputName(node_def, output_idx);
    ++output_idx;
    Tensor* split_count_t = nullptr;
    DataType type = kInt32;
    TensorShape shape({1});
    ctx->Allocate(split_count_out_name,
                  shape, type, &split_count_t);
    *(split_count_t->Raw<int32_t>()) = split_cnt[i];

    // output merge idx
    std::string merge_idx_name =
        OutputName(node_def, output_idx);
    ++output_idx;
    Tensor* merge_idx_t = nullptr;
    ctx->Allocate(merge_idx_name,
                  shape, type, &merge_idx_t);
    *(merge_idx_t->Raw<int32_t>()) = begin_id;
    begin_id += split_cnt[i] * 3;
  }
}

REGISTER_OP_KERNEL("SAMPLE_EDGE_SPLIT", SampleEdgeSplit);

}  // namespace euler
