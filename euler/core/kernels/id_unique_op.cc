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
#include <unordered_map>

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/common/str_util.h"
#include "euler/common/logging.h"
#include "euler/common/data_types.h"

namespace euler {
class IdUnique: public OpKernel {
 public:
  explicit IdUnique(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

void IdUnique::Compute(const DAGNodeProto& node_def,
                       OpKernelContext* ctx) {
  // get ids tensor
  Tensor* ids_tensor = nullptr;
  ctx->tensor(node_def.inputs(0), &ids_tensor);
  if (ids_tensor->Shape().Size() == 1) {  // node ids
    std::unordered_map<uint64_t, int32_t> ids_map;
    ids_map.reserve(ids_tensor->NumElements());
    std::vector<uint64_t> unique_vec;
    unique_vec.reserve(ids_tensor->NumElements());
    int32_t cnt = 0;
    for (int32_t i = 0; i < ids_tensor->NumElements(); ++i) {
      if (ids_map.find(ids_tensor->Raw<uint64_t>()[i]) == ids_map.end()) {
        ids_map[ids_tensor->Raw<uint64_t>()[i]] = cnt++;
        unique_vec.push_back(ids_tensor->Raw<uint64_t>()[i]);
      }
    }
    // output
    std::string unique_id_name = OutputName(node_def, 0);
    std::string gather_idx_name = OutputName(node_def, 1);
    TensorShape unique_id_shape({ids_map.size()});
    TensorShape gather_idx_shape(
                {static_cast<uint64_t>(ids_tensor->NumElements())});
    Tensor* unique_id_t = nullptr, *gather_idx_t = nullptr;
    ctx->Allocate(unique_id_name, unique_id_shape, kUInt64, &unique_id_t);
    ctx->Allocate(gather_idx_name, gather_idx_shape, kInt32, &gather_idx_t);
    std::copy(unique_vec.begin(), unique_vec.end(),
              unique_id_t->Raw<uint64_t>());
    for (int32_t i = 0; i < ids_tensor->NumElements(); ++i) {
      gather_idx_t->Raw<int32_t>()[i] = ids_map.at(
          ids_tensor->Raw<uint64_t>()[i]);
    }
  } else {  // edge ids
    std::vector<euler::common::EdgeID> eids(ids_tensor->NumElements() / 3);
    for (size_t i = 0; i < eids.size(); ++i) {
      eids[i] = euler::common::EdgeID(
          ids_tensor->Raw<uint64_t>()[i * 3],
          ids_tensor->Raw<uint64_t>()[i * 3 + 1],
          static_cast<int32_t>(ids_tensor->Raw<uint64_t>()[i * 3 + 2]));
    }
    std::unordered_map<euler::common::EdgeID, int32_t,
        euler::common::EdgeIDHashFunc, euler::common::EdgeIDEqualKey> ids_map;
    ids_map.reserve(ids_tensor->NumElements() / 3);
    std::vector<uint64_t> unique_vec;
    unique_vec.reserve(ids_tensor->NumElements());
    for (size_t i = 0; i < eids.size(); ++i) {
      if (ids_map.find(eids[i]) == ids_map.end()) {
        size_t cnt = ids_map.size();
        ids_map[eids[i]] = cnt;
        unique_vec.push_back(std::get<0>(eids[i]));
        unique_vec.push_back(std::get<1>(eids[i]));
        unique_vec.push_back(std::get<2>(eids[i]));
      }
    }
    // output
    std::string unique_id_name = OutputName(node_def, 0);
    std::string gather_idx_name = OutputName(node_def, 1);
    TensorShape unique_id_shape({ids_map.size(), 3});
    TensorShape gather_idx_shape({eids.size()});
    Tensor* unique_id_t = nullptr, *gather_idx_t = nullptr;
    ctx->Allocate(unique_id_name, unique_id_shape, kUInt64, &unique_id_t);
    ctx->Allocate(gather_idx_name, gather_idx_shape, kInt32, &gather_idx_t);
    std::copy(unique_vec.begin(), unique_vec.end(),
              unique_id_t->Raw<uint64_t>());
    for (size_t i = 0; i < eids.size(); ++i) {
      gather_idx_t->Raw<int32_t>()[i] = ids_map.at(eids[i]);
    }
  }
}

REGISTER_OP_KERNEL("ID_UNIQUE", IdUnique);

}  // namespace euler
