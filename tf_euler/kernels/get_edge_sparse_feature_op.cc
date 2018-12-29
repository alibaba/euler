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
#include "tf_euler/utils/sparse_tensor_builder.h"

namespace tensorflow {

extern std::unique_ptr<euler::client::Graph>& Graph();

class GetEdgeSparseFeature: public AsyncOpKernel {
 public:
  explicit GetEdgeSparseFeature(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_ids", &feature_ids_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_values", &default_values_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    OP_REQUIRES(ctx, default_values_.size() == feature_ids_.size(),
                errors::InvalidArgument(
                    "Require default_values.size() == default_values.size()"));
    OP_REQUIRES(ctx, static_cast<size_t>(N_) == feature_ids_.size(),
                errors::InvalidArgument("Require N == feature_ids.size"));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::vector<int> feature_ids_;
  std::vector<int> default_values_;
  int N_;
};


void GetEdgeSparseFeature::ComputeAsync(OpKernelContext* ctx,
                                        DoneCallback done) {
  auto edges = ctx->input(0);
  auto shape = edges.shape();
  OP_REQUIRES(ctx, shape.dim_size(1) == 3,
              errors::InvalidArgument(
                  "Input `edges` must be a matrix with shape [n, 3]"));

  auto edges_data = edges.flat<int64>();
  std::vector<euler::client::EdgeID> edge_ids(shape.dim_size(0));
  for (int i = 0; i < edge_ids.size(); ++i) {
    edge_ids[i] = std::make_tuple(edges_data(3 * i),
                                  edges_data(3 * i + 1),
                                  static_cast<int32_t>(edges_data(3 * i + 2)));
  }

  auto callback = [ctx, done, this] (
      const euler::client::UInt64FeatureVec& result) {
    std::vector<SparseTensorBuilder<int64, 2>> builders(N_);
    for (size_t i = 0; i < result.size(); ++i) {
      if (result[i].empty()) {
        for (int j = 0; j < N_; ++j) {
          builders[j].emplace({static_cast<int64>(i), 0}, default_values_[j]);
        }
      } else {
        for (size_t j = 0; j < result[i].size(); ++j) {
          auto& features = result[i][j];
          if (features.empty()) {
            builders[j].emplace({static_cast<int64>(i), 0}, default_values_[j]);
          }
          for (size_t k = 0; k < features.size(); ++k) {
            builders[j].emplace(
                {static_cast<int64>(i), static_cast<int64>(k)}, features[k]);
          }
        }
      }
    }

    OpOutputList indices, values, shape;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("indices", &indices), done);
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("values", &values), done);
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("dense_shape", &shape), done);

    for (size_t j = 0; j < N_; ++j) {
      indices.set(j, builders[j].indices());
      values.set(j, builders[j].values());
      shape.set(j, builders[j].dense_shape());
    }

    done();
  };

  Graph()->GetEdgeUint64Feature(edge_ids, feature_ids_, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("GetEdgeSparseFeature").Device(DEVICE_CPU), GetEdgeSparseFeature);

}  // namespace tensorflow
