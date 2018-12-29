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

#include <string.h>

#include <memory>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "euler/client/graph.h"

namespace tensorflow {

extern std::unique_ptr<euler::client::Graph>& Graph();

class GetEdgeBinaryFeature: public AsyncOpKernel {
 public:
  explicit GetEdgeBinaryFeature(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_ids", &feature_ids_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  int64 N_;
  std::vector<int> feature_ids_;
};

void GetEdgeBinaryFeature::ComputeAsync(OpKernelContext* ctx,
                                        DoneCallback done) {
  auto edges = ctx->input(0);
  auto& shape = edges.shape();

  OP_REQUIRES(ctx, shape.dim_size(1) == 3,
              errors::InvalidArgument(
                  "Input `edges` must be a matrix with shape [n, 3]"));

  std::vector<Tensor*> outputs(N_, nullptr);
  TensorShape output_shape;
  output_shape.AddDim(shape.dim_size(0));
  for (auto i = 0; i < N_; ++i) {
    OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &outputs[i]));
  }

  auto edges_flat = edges.flat<int64>();
  std::vector<euler::client::EdgeID> edge_ids(shape.dim_size(0));
  for (size_t i = 0; i < edge_ids.size(); ++i) {
    edge_ids[i] = std::make_tuple(edges_flat(3 * i),
                                  edges_flat(3 * i + 1),
                                  static_cast<int32_t>(edges_flat(3 * i + 2)));
  }

  auto callback =
      [outputs, done, this] (const euler::client::BinaryFatureVec& result) {
        for (size_t i = 0; i < result.size(); ++i) {
          for (size_t j = 0; j < result[i].size(); ++j) {
            auto data = outputs[j]->flat<tensorflow::string>();
            data(i) = result[i][j];
          }
        }
        done();
      };

  Graph()->GetEdgeBinaryFeature(edge_ids, feature_ids_, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("GetEdgeBinaryFeature").Device(DEVICE_CPU), GetEdgeBinaryFeature);

}  // namespace tensorflow
