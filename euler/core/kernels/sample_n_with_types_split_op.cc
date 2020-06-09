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

class SampleNWithTypesSplit: public OpKernel {
 public:
  explicit SampleNWithTypesSplit(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void SampleNWithTypesSplit::Compute(const DAGNodeProto& node_def,
                                    OpKernelContext* ctx) {
  // get split num
  int32_t split_num = QueryProxy::GetInstance()->GetShardNum();
  // get counts tensor
  Tensor* counts_t = nullptr;
  ctx->tensor(node_def.inputs(0), &counts_t);
  // get node_types tensor
  Tensor* node_types_t = nullptr;
  ctx->tensor(node_def.inputs(1), &node_types_t);

  int32_t types_num = node_types_t->NumElements();
  if (types_num != counts_t->NumElements()) {
    EULER_LOG(FATAL) << "types num != counts num";
  }

  const std::vector<std::vector<float>>& shard_node_weight =
      QueryProxy::GetInstance()->GetShardNodeWeight();

  // split_num * types_num
  std::vector<std::vector<int32_t>> split_cnts(split_num);
  for (int32_t i = 0; i < split_num; ++i) {
    split_cnts[i].resize(types_num);
  }

  std::vector<size_t> no_zero_idxs; no_zero_idxs.reserve(split_num);
  for (int32_t i = 0; i < types_num; ++i) {
    int32_t count = counts_t->Raw<int32_t>()[i];
    int32_t remain_cnt = count;
    int32_t node_type = node_types_t->Raw<int32_t>()[i];
    for (int32_t j = 0; j < split_num; ++j) {
      if (fabs(shard_node_weight[node_type][split_num] - 0.0) < 0.0000001) {
        EULER_LOG(FATAL) << "node type " << node_type << " sum weight is zero!";
      }
      split_cnts[j][i] = floor(count * shard_node_weight[node_type][j] /
        shard_node_weight[node_type][split_num]);
      remain_cnt -= split_cnts[j][i];
      if (shard_node_weight[node_type][j] > 0) {
        no_zero_idxs.push_back(j);
      }
    }
    for (size_t j = 0; remain_cnt > 0; ++j, --remain_cnt) {
      size_t no_zero_idx = no_zero_idxs[
          floor(common::ThreadLocalRandom() * no_zero_idxs.size())];
      split_cnts[no_zero_idx][i] += 1;
    }
    no_zero_idxs.clear();
  }

  /* output */
  int32_t output_idx = 0;
  for (int32_t i = 0; i < split_num; ++i) {
    std::string output_name =
        OutputName(node_def, output_idx);
    Tensor* split_counts_t = nullptr;
    DataType type = kInt32;
    TensorShape shape({static_cast<size_t>(types_num)});
    ctx->Allocate(output_name,
                  shape, type, &split_counts_t);
    std::copy(split_cnts[i].begin(), split_cnts[i].end(),
              split_counts_t->Raw<int32_t>());
    output_idx += 2;
  }
}

REGISTER_OP_KERNEL("SAMPLE_N_WITH_TYPES_SPLIT", SampleNWithTypesSplit);

}  // namespace euler
