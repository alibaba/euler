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

class SampleEdge: public AsyncOpKernel {
 public:
  explicit SampleEdge(OpKernelConstruction* ctx): AsyncOpKernel(ctx) { }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
};

void SampleEdge::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto count = ctx->input(0);
  auto edge_type = ctx->input(1);

  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(count.shape()),
              errors::InvalidArgument("count must be a scalar, saw shape: ",
                                      count.shape().DebugString()));
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(edge_type.shape()),
              errors::InvalidArgument("edge_type must be a scalar, saw shape: ",
                                      edge_type.shape().DebugString()));

  auto count_value = (count.scalar<int32>())();
  auto type_value = (edge_type.scalar<int32>())();

  TensorShape output_shape;
  output_shape.AddDim(count_value);
  output_shape.AddDim(3);

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  auto callback = [output, done] (const euler::client::EdgeIDVec& result) {
    auto data = output->flat<int64>().data();
    for (size_t i = 0; i < result.size(); ++i) {
      data[3 * i] = std::get<0>(result[i]);
      data[3 * i + 1] = std::get<1>(result[i]);
      data[3 * i + 2] = std::get<2>(result[i]);
    }
    done();
  };

  Graph()->SampleEdge(type_value, count_value, callback);
}

REGISTER_KERNEL_BUILDER(Name("SampleEdge").Device(DEVICE_CPU), SampleEdge);

}  // namespace tensorflow
