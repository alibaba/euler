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

class GetDenseFeature: public AsyncOpKernel {
 public:
  explicit GetDenseFeature(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
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

void GetDenseFeature::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0);
  auto& shape = nodes.shape();
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

  auto nodes_flat = nodes.flat<int64>();
  std::vector<euler::client::NodeID> node_ids(shape.dim_size(0));
  for (size_t i = 0; i < node_ids.size(); ++i) {
    node_ids[i] = nodes_flat(i);
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

  Graph()->GetNodeFloat32Feature(node_ids, feature_ids_, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("GetDenseFeature").Device(DEVICE_CPU), GetDenseFeature);

}  // namespace tensorflow
