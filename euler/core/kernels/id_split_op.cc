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

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/core/api/api.h"
#include "euler/client/client_manager.h"
#include "euler/client/query_proxy.h"
#include "euler/common/str_util.h"

namespace euler {

class IDSplit: public OpKernel {
 public:
  explicit IDSplit(const std::string& name) : OpKernel(name) {
    ClientManager* cm = ClientManager::GetInstance();
    std::string partition_num_str;
    if (!cm->RetrieveMeta("num_partitions", &partition_num_str)) {
      EULER_LOG(FATAL) << "get num partition error";
    }
    partition_number_ = atoi(partition_num_str.c_str());
    if (partition_number_ <= 0) {
      EULER_LOG(FATAL) << "invalid num partition";
    }
  }
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
 private:
  int32_t partition_number_;

  int32_t GetShardId(NodeId id, int32_t shard_num) {
    return (id % partition_number_) % shard_num;
    // return id % shard_num;
  }
};

void IDSplit::Compute(const DAGNodeProto& node_def,
                      OpKernelContext* ctx) {
  // get split num
  int32_t split_num = QueryProxy::GetInstance()->GetShardNum();
  // get ids tensor
  Tensor* ids_tensor = nullptr;
  ctx->tensor(node_def.inputs(0), &ids_tensor);

  if (ids_tensor->Shape().Size() == 1) {  // node ids
    int32_t ids_num = ids_tensor->NumElements();
    // calculate shard ids tensor and merge idx tensor
    std::vector<std::vector<NodeId>> shard_ids(split_num);
    std::vector<std::vector<int32_t>> merge_idx(split_num);
    for (int32_t i = 0; i < split_num; ++i) {
      shard_ids[i].reserve(ids_num / split_num);
      merge_idx[i].reserve(ids_num / split_num);
    }
    auto ids = ids_tensor->Raw<NodeId>();
    for (int32_t i = 0; i < ids_num; ++i) {
      NodeId id = ids[i];
      int32_t shard_no = GetShardId(id, split_num);
      shard_ids[shard_no].push_back(id);
      merge_idx[shard_no].push_back(i);
    }
    // output tensor
    for (int32_t i = 0; i < split_num; ++i) {
      // split ids
      int32_t output_idx = i * 2;
      std::string split_ids_out_name = OutputName(node_def, output_idx);

      Tensor* split_ids_t = nullptr;
      DataType type = kUInt64;
      TensorShape shape({shard_ids[i].size()});
      ctx->Allocate(split_ids_out_name, shape, type, &split_ids_t);
      std::copy(shard_ids[i].begin(), shard_ids[i].end(),
                split_ids_t->Raw<NodeId>());

      // merge idx
      ++output_idx;
      std::string merge_idx_name =
        OutputName(node_def, output_idx);
      Tensor* merge_idx_t = nullptr;
      type = kInt32;
      TensorShape merge_idx_shape({merge_idx[i].size()});
      ctx->Allocate(merge_idx_name, merge_idx_shape, type, &merge_idx_t);
      std::copy(merge_idx[i].begin(), merge_idx[i].end(),
                merge_idx_t->Raw<int32_t>());
    }
  } else {  // edge ids
    size_t id_shape = ids_tensor->Shape().Dims()[1];
    int32_t ids_num = ids_tensor->NumElements() / id_shape;
    // calculate shard ids tensor and merge idx tensor
    std::vector<std::vector<uint64_t>> shard_ids(split_num);
    std::vector<std::vector<int32_t>> merge_idx(split_num);
    for (int32_t i = 0; i < split_num; ++i) {
      shard_ids[i].reserve(ids_num * id_shape / split_num);
      merge_idx[i].reserve(ids_num / split_num);
    }
    auto ids = ids_tensor->Raw<uint64_t>();
    for (int32_t i = 0; i < ids_num; ++i) {
      int32_t shard_no = GetShardId(ids[i * id_shape], split_num);
      for (size_t j = 0; j < id_shape; ++j) {
        shard_ids[shard_no].push_back(ids[i * id_shape + j]);
      }
      merge_idx[shard_no].push_back(i);
    }
    // output tensor
    for (int32_t i = 0; i < split_num; ++i) {
      // split ids
      int32_t output_idx = i * 2;
      std::string split_ids_out_name = OutputName(node_def, output_idx);

      Tensor* split_ids_t = nullptr;
      DataType type = kInt64;
      TensorShape shape({shard_ids[i].size() / id_shape, id_shape});
      ctx->Allocate(split_ids_out_name, shape, type, &split_ids_t);
      std::copy(shard_ids[i].begin(), shard_ids[i].end(),
                split_ids_t->Raw<uint64_t>());

      // merge idx
      ++output_idx;
      std::string merge_idx_name =
        OutputName(node_def, output_idx);
      Tensor* merge_idx_t = nullptr;
      type = kInt32;
      TensorShape merge_idx_shape({merge_idx[i].size()});
      ctx->Allocate(merge_idx_name, merge_idx_shape, type, &merge_idx_t);
      std::copy(merge_idx[i].begin(), merge_idx[i].end(),
                merge_idx_t->Raw<int32_t>());
    }
  }
}

REGISTER_OP_KERNEL("ID_SPLIT", IDSplit);
}  // namespace euler
