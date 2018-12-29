/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

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

#include <memory>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "euler/client/graph.h"
#include "tf_euler/utils/sparse_tensor_builder.h"

namespace tensorflow {

extern std::unique_ptr<euler::client::Graph>& Graph();

class GetSortedFullNeighbor: public AsyncOpKernel {
 public:
  explicit GetSortedFullNeighbor(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) { }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
};

void GetSortedFullNeighbor::ComputeAsync(
    OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0);
  auto edge_types = ctx->input(1);

  auto nodes_flat = nodes.flat<int64>();
  std::vector<euler::client::NodeID> node_ids(nodes_flat.size());
  for (int i = 0; i < nodes_flat.size(); ++i) {
    node_ids[i] = nodes_flat(i);
  }

  auto etypes_flat = edge_types.flat<int32>();
  std::vector<int> etypes(etypes_flat.size());
  for (int i = 0; i < etypes_flat.size(); ++i) {
    etypes[i] = etypes_flat(i);
  }

  auto callback =
      [ctx, done, this] (const euler::client::IDWeightPairVec& result) {
        SparseTensorBuilder<int64, 2> id_builder;
        SparseTensorBuilder<float, 2> weight_builder;
        SparseTensorBuilder<int32, 2> type_builder;
        for (size_t i = 0; i < result.size(); ++i) {
          for (size_t j = 0; j < result[i].size(); ++j) {
            id_builder.emplace(
                {static_cast<int64>(i), static_cast<int64>(j)},
                std::get<0>(result[i][j]));
            weight_builder.emplace(
                {static_cast<int64>(i), static_cast<int64>(j)},
                std::get<1>(result[i][j]));
            type_builder.emplace(
                {static_cast<int64>(i), static_cast<int64>(j)},
                std::get<2>(result[i][j]));
          }
        }

        // Set id sparse tensor
        (void) ctx->set_output("id_indices", id_builder.indices());
        (void) ctx->set_output("id_values", id_builder.values());
        (void) ctx->set_output("id_shape", id_builder.dense_shape());

        // set weight sparse tensor
        (void) ctx->set_output("weight_indices", weight_builder.indices());
        (void) ctx->set_output("weight_values", weight_builder.values());
        (void) ctx->set_output("weight_shape", weight_builder.dense_shape());

        // set type sparse tensor
        (void) ctx->set_output("type_indices", type_builder.indices());
        (void) ctx->set_output("type_values", type_builder.values());
        (void) ctx->set_output("type_shape", type_builder.dense_shape());

        done();
      };

  Graph()->GetSortedFullNeighbor(node_ids, etypes, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("GetSortedFullNeighbor").Device(DEVICE_CPU), GetSortedFullNeighbor);

}  // namespace tensorflow
