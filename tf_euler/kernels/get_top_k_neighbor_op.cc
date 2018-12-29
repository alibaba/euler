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

namespace tensorflow {

extern std::unique_ptr<euler::client::Graph>& Graph();

class GetTopKNeighbor: public AsyncOpKernel {
 public:
  explicit GetTopKNeighbor(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("k", &k_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_node", &default_node_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  int k_;
  int default_node_;
};

void GetTopKNeighbor::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0);
  auto edge_types = ctx->input(1);
  TensorShape output_shape;
  output_shape.AddDim(nodes.shape().dim_size(0));
  output_shape.AddDim(k_);

  Tensor* output = nullptr;
  Tensor* weights = nullptr;
  Tensor* types = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(1, output_shape, &weights));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(2, output_shape, &types));

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

  auto output_data = output->flat<int64>().data();
  auto weights_data = weights->flat<float>().data();
  auto types_data = types->flat<int32>().data();
  auto output_size = output_shape.dim_size(0) * output_shape.dim_size(1);
  std::fill(output_data, output_data + output_size, default_node_);
  std::fill(weights_data, weights_data + output_size, 0.0);
  std::fill(types_data, types_data + output_size, -1);
  auto callback = [output_data, weights_data, types_data, done, this] (
      const euler::client::IDWeightPairVec& result) {
    for (size_t i = 0; i < result.size(); ++i) {
      auto data = output_data + i * k_;
      auto wdata = weights_data + i * k_;
      auto tdata = types_data + i * k_;
      for (size_t j = 0; j < result[i].size(); ++j) {
        data[j] = std::get<0>(result[i][j]);
        wdata[j] = std::get<1>(result[i][j]);
        tdata[j] = std::get<2>(result[i][j]);
      }
    }
    done();
  };

  Graph()->GetTopKNeighbor(node_ids, etypes, k_, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("GetTopKNeighbor").Device(DEVICE_CPU), GetTopKNeighbor);

}  // namespace tensorflow
