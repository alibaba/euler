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

static const size_t kTraverseBound = (1 << 15);  // 32K
static const size_t kMaxSampleTime = 50;

class SampleNodeOp: public OpKernel {
 public:
  explicit SampleNodeOp(const std::string& name): OpKernel(name) { }

  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
};

void SampleNodeOp::Compute(const DAGNodeProto& node_def, OpKernelContext* ctx) {
  if (node_def.inputs_size() != 2) {
    EULER_LOG(ERROR) << "Invalid input arguments for SampleNode";
    return;
  }

  Tensor* node_type_t = nullptr;
  auto s = ctx->tensor(node_def.inputs(0), &node_type_t);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Retrieve node_type input for SampleNode failed!";
    return;
  }

  Tensor* count_t = nullptr;
  s = ctx->tensor(node_def.inputs(1), &count_t);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Retrieve count input for SampleNode failed!";
    return;
  }

  std::vector<int32_t> node_types(node_type_t->NumElements());
  std::copy(node_type_t->Raw<int32_t>(),
            node_type_t->Raw<int32_t>() + node_type_t->NumElements(),
            node_types.begin());
  std::unordered_set<int32_t> node_type_set(node_types.begin(),
                                            node_types.end());
  int count = *count_t->Raw<int32_t>();
  NodeIdVec node_id_vec;
  if (node_types[0] == -1 && node_def.dnf_size() > 0) {  // Sample all type
    node_id_vec = SampleByIndex(node_def, count, ctx);
  } else {
    auto graph = EulerGraph();
    node_id_vec = euler::SampleNode(node_types, count);
    if (node_def.dnf_size() > 0) {
      node_id_vec.clear();
      auto index_result = QueryIndex(node_def, ctx);
      if (index_result->size() < kTraverseBound) {
        auto ids = index_result->GetIds();
        auto weights = index_result->GetWeights();
        size_t cur = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
          auto node = graph->GetNodeByID(ids[i]);
          if (node != nullptr &&
              node_type_set.find(node->GetType()) != node_type_set.end()) {
            ids[cur] = ids[i];
            weights[cur] = ids[i];
            ++cur;
          }
        }
        ids.resize(cur);
        weights.resize(cur);
        if (!ids.empty()) {
          using CWC = euler::common::CompactWeightedCollection<uint64_t>;
          CWC sampler;
          sampler.Init(ids, weights);
          while (node_id_vec.size() < static_cast<size_t>(count)) {
            node_id_vec.emplace_back(sampler.Sample().first);
          }
        }
      } else {
        size_t sample_time = 0;
        while (node_id_vec.size() < static_cast<size_t>(count) &&
               sample_time < kMaxSampleTime) {
          auto sample_result = index_result->Sample(count);
          for (auto& item : sample_result) {
            auto node = graph->GetNodeByID(item.first);
            if (node != nullptr &&
                node_type_set.find(node->GetType()) != node_type_set.end()) {
              node_id_vec.emplace_back(item.first);
              sample_time = 0;
              if (node_id_vec.size() == static_cast<size_t>(count)) {
                break;
              }
            }
          }
          ++sample_time;
        }
      }
    }
  }

  if (node_id_vec.size() != static_cast<size_t>(count)) {
    EULER_LOG(ERROR) << "Expect sample count: " << count
                     << ", real got:" << node_id_vec.size();
    return;
  }

  std::string output_name = OutputName(node_def, 0);
  TensorShape shape({ static_cast<size_t>(count) });
  Tensor* output = nullptr;
  s = ctx->Allocate(output_name, shape, DataType::kInt64, &output);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor failed!";
    return;
  }

  memcpy(output->Raw<int64_t>(), node_id_vec.data(),
         node_id_vec.size() * sizeof(node_id_vec[0]));
}

REGISTER_OP_KERNEL("API_SAMPLE_NODE", SampleNodeOp);

}  // namespace euler
