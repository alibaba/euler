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

class GetEdgeDenseFeature: public AsyncOpKernel {
 public:
  explicit GetEdgeDenseFeature(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_ids", &feature_ids_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dimensions", &dimensions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::vector<int> feature_ids_;
  std::vector<int> dimensions_;
  int N_;
};

void GetEdgeDenseFeature::ComputeAsync(OpKernelContext* ctx,
                                       DoneCallback done) {
  auto edges = ctx->input(0);
  auto& shape = edges.shape();
  OP_REQUIRES(ctx, shape.dim_size(1) == 3,
              errors::InvalidArgument(
                  "Input `edges` must be a matrix with shape [n, 3]"));

  std::vector<Tensor*> outputs(N_, nullptr);
  for (int i = 0; i < N_; ++i) {
    TensorShape output_shape;
    output_shape.AddDim(shape.dim_size(0));
    output_shape.AddDim(dimensions_[i]);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &outputs[i]));
    auto data = outputs[i]->flat<float>().data();
    auto end = data + shape.dim_size(0) * dimensions_[i];
    std::fill(data, end, 0.0);
  }

  auto edges_flat = edges.flat<int64>();
  std::vector<euler::client::EdgeID> edge_ids(shape.dim_size(0));
  for (size_t i = 0; i < edge_ids.size(); ++i) {
    edge_ids[i] = std::make_tuple(edges_flat(3 * i),
                                  edges_flat(3 * i + 1),
                                  static_cast<int32_t>(edges_flat(3 * i + 2)));
  }

  auto callback = [outputs, done, this] (
      const euler::client::FloatFeatureVec& result) {
    for (size_t i = 0; i < result.size(); ++i) {
      for (size_t j = 0; j < result[i].size(); ++j) {
        auto data = outputs[j]->flat<float>().data() + i * dimensions_[j];
        size_t dim = dimensions_[j];
        for (size_t k = 0; k < result[i][j].size() && k < dim; ++k) {
          data[k] = result[i][j][k];
        }
      }
    }
    done();
  };

  Graph()->GetEdgeFloat32Feature(edge_ids, feature_ids_, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("GetEdgeDenseFeature").Device(DEVICE_CPU), GetEdgeDenseFeature);

}  // namespace tensorflow
