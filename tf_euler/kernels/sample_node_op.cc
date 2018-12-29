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

class SampleNode: public AsyncOpKernel {
 public:
  explicit SampleNode(OpKernelConstruction* ctx): AsyncOpKernel(ctx) { }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
};

void SampleNode::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto count = ctx->input(0);
  auto node_type = ctx->input(1);

  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(count.shape()),
              errors::InvalidArgument("count must be a scalar, saw shape: ",
                                      count.shape().DebugString()));
  OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(node_type.shape()),
              errors::InvalidArgument("node_type must be a scalar, saw shape: ",
                                      node_type.shape().DebugString()));

  auto count_value = (count.scalar<int32>())();
  auto type_value = (node_type.scalar<int32>())();

  TensorShape output_shape;
  output_shape.AddDim(count_value);

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  auto callback = [output, done] (const euler::client::NodeIDVec& result) {
    auto data = output->flat<int64>().data();
    for (size_t i = 0; i < result.size(); ++i) {
      data[i] = result[i];
    }
    done();
  };

  Graph()->SampleNode(type_value, count_value, callback);
}

REGISTER_KERNEL_BUILDER(Name("SampleNode").Device(DEVICE_CPU), SampleNode);

}  // namespace tensorflow
