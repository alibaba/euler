/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#include <string.h>

#include <memory>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "euler/client/graph.h"

namespace tensorflow {

extern std::unique_ptr<euler::client::Graph>& Graph();

class GetNodeType : public AsyncOpKernel {
 public:
  explicit GetNodeType(OpKernelConstruction* ctx): AsyncOpKernel(ctx) { }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
};

void GetNodeType::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0);
  auto nodes_flat = nodes.flat<int64>();
  std::vector<euler::client::NodeID> node_ids(nodes_flat.size());
  for (size_t i = 0; i < node_ids.size(); ++i) {
    node_ids[i] = nodes_flat(i);
  }

  TensorShape output_shape;
  output_shape.AddDim(node_ids.size());
  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  auto callback = [output, done] (const euler::client::TypeVec& result) {
    auto data = output->flat<int32>().data();
    for (size_t i = 0; i < result.size(); ++i) {
      data[i] = result[i];
    }
    done();
  };

  Graph()->GetNodeType(node_ids, callback);
}

REGISTER_KERNEL_BUILDER(Name("GetNodeType").Device(DEVICE_CPU), GetNodeType);

}  // namespace tensorflow
