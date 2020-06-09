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
#include <string>

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/core/api/api.h"
#include "euler/common/str_util.h"
#include "euler/common/compact_weighted_collection.h"

namespace euler {

struct DstTypeWeight {
  uint64_t dst_id_;
  float edge_weight_;
  int32_t edge_type_;
};

class LocalSampleLayer: public OpKernel {
 public:
  explicit LocalSampleLayer(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void LocalSampleLayer::Compute(const DAGNodeProto& node_def,
                               OpKernelContext* ctx) {
  Tensor* batch_nb_idx_t = nullptr;  // size = batch * n * 2
  Tensor* batch_nb_id_t = nullptr;
  Tensor* batch_nb_w_t = nullptr;
  Tensor* batch_nb_type_t = nullptr;
  Tensor* n_t = nullptr;
  Tensor* m_t = nullptr;
  std::string weight_func = "";
  int64_t default_node = 0;
  ctx->tensor(node_def.inputs(0), &batch_nb_idx_t);
  ctx->tensor(node_def.inputs(1), &batch_nb_id_t);
  ctx->tensor(node_def.inputs(2), &batch_nb_w_t);
  ctx->tensor(node_def.inputs(3), &batch_nb_type_t);
  ctx->tensor(node_def.inputs(4), &n_t);
  ctx->tensor(node_def.inputs(5), &m_t);
  weight_func = node_def.inputs(6);
  default_node = atol(node_def.inputs(7).c_str());
  int32_t n = n_t->Raw<int32_t>()[0];
  int32_t m = m_t->Raw<int32_t>()[0];
  int32_t batch = batch_nb_idx_t->NumElements() / (n * 2);

  std::vector<int32_t> batch_nb_offset;
  batch_nb_offset.reserve(batch);
  for (int32_t i = 0; i < batch_nb_idx_t->NumElements(); i += n * 2) {
    batch_nb_offset.push_back(batch_nb_idx_t->Raw<int32_t>()[i]);
  }

  std::vector<std::unordered_map<std::string, DstTypeWeight>>
   batch_nb_unique_map(batch);
  for (int32_t i = 0; i < batch; ++i) {
    int32_t batch_begin = batch_nb_offset[i];
    int32_t batch_end = 0;
    if (i < batch - 1) {
      batch_end = batch_nb_offset[i + 1];
    } else {
      batch_end =
        batch_nb_idx_t->Raw<int32_t>()[batch_nb_idx_t->NumElements() - 1];
    }
    for (int32_t j = batch_begin; j < batch_end; ++j) {
      uint64_t dst_id = batch_nb_id_t->Raw<uint64_t>()[j];
      float weight = batch_nb_w_t->Raw<float>()[j];
      int32_t type = batch_nb_type_t->Raw<int32_t>()[j];
      std::string key = std::to_string(dst_id) + std::to_string(type);
      if (batch_nb_unique_map[i].find(key) == batch_nb_unique_map[i].end()) {
        batch_nb_unique_map[i][key] = {dst_id, weight, type};
      } else {
        batch_nb_unique_map[i][key].edge_weight_ += weight;
      }
    }
  }
  /* use weight func */
  if (weight_func == "sqrt") {
    for (int32_t i = 0; i < batch; ++i) {
      for (auto it = batch_nb_unique_map[i].begin();
          it != batch_nb_unique_map[i].end(); ++it) {
        it->second.edge_weight_ = sqrt(it->second.edge_weight_);
      }
    }
  } else {
    EULER_LOG(ERROR) << "weight function: " << weight_func << " not support";
  }
  /* build sampler */
  std::vector<euler::common::CompactWeightedCollection<DstTypeWeight>>
      batch_sampler(batch);
  for (int32_t i = 0; i < batch; ++i) {
    if (!batch_nb_unique_map[i].empty()) {
      std::vector<DstTypeWeight> values;
      values.reserve(batch_nb_unique_map[i].size());
      std::vector<float> weights;
      weights.reserve(batch_nb_unique_map[i].size());
      for (auto it = batch_nb_unique_map[i].begin();
           it != batch_nb_unique_map[i].end(); ++it) {
        values.push_back(it->second);
        weights.push_back(it->second.edge_weight_);
      }
      batch_sampler[i].Init(values, weights);
    }
  }

  /* sample layerwise nb */
  Tensor* o_nb_t = nullptr;
  Tensor* o_w_t = nullptr;
  Tensor* o_t_t = nullptr;
  TensorShape shape({static_cast<size_t>(batch * m), 1});
  ctx->Allocate(OutputName(node_def, 0), shape, DataType::kUInt64, &o_nb_t);
  ctx->Allocate(OutputName(node_def, 1), shape, DataType::kFloat, &o_w_t);
  ctx->Allocate(OutputName(node_def, 2), shape, DataType::kInt32, &o_t_t);
  for (int32_t i= 0; i < batch; ++i) {
    if (batch_sampler[i].GetSize() == 0 ||
      batch_sampler[i].GetSumWeight() == 0) {
      memset(o_nb_t->Raw<uint64_t>() + i * m, default_node,
             sizeof(uint64_t) * m);
      memset(o_w_t->Raw<float>() + i * m, 0, sizeof(float) * m);
      memset(o_t_t->Raw<int32_t>() + i * m, 0, sizeof(int32_t) * m);
    } else {
      for (int32_t j = 0; j < m; ++j) {
        DstTypeWeight e = batch_sampler[i].Sample().first;
        o_nb_t->Raw<uint64_t>()[i * m + j] = e.dst_id_;
        o_w_t->Raw<float>()[i * m + j] = e.edge_weight_;
        o_t_t->Raw<int32_t>()[i * m + j] = e.edge_type_;
      }
    }
  }
}

REGISTER_OP_KERNEL("API_LOCAL_SAMPLE_L", LocalSampleLayer);

}  // namespace euler
