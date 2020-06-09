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

#include <memory>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tf_euler/utils/euler_query_proxy.h"

namespace tensorflow {

class SampleNode: public AsyncOpKernel {
 public:
  explicit SampleNode(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("condition", &condition_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::string condition_;
};

void SampleNode::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto count = ctx->input(0);
  auto node_type = ctx->input(1);

  OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsScalar(count.shape()),
              errors::InvalidArgument("count must be a scalar, saw shape: ",
                                      count.shape().DebugString()), done);

  OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsScalar(node_type.shape()),
              errors::InvalidArgument("node_type must be a scalar, saw shape: ",
                                      node_type.shape().DebugString()), done);

  int32_t count_value = (count.scalar<int32>())();
  int32_t type_value = (node_type.scalar<int32>())();

  TensorShape output_shape;
  output_shape.AddDim(count_value);

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  char buffer[4096];
  if (!condition_.empty()) {
    int ret = snprintf(
        buffer, sizeof(buffer), "sampleN(node_type, count).has(%s).as(id)",
        condition_.c_str());
    if (ret < 0 || static_cast<size_t>(ret) > sizeof(buffer)) {
      EULER_LOG(ERROR) << "Can not build query, the condition is too long,"
                       << " condition: " << condition_;
      done();
      return;
    }
  } else {
    snprintf(buffer, sizeof(buffer), "sampleN(node_type, count).as(id)");
  }

  // build euler gremlin query
  auto query = new euler::Query(buffer);
  auto t_node_type = query->AllocInput("node_type", {}, euler::kInt32);
  euler::Tensor* t_count = query->AllocInput("count", {}, euler::kInt32);
  *(t_node_type->Raw<int32_t>()) = type_value;
  *(t_count->Raw<int32_t>()) = count_value;

  auto callback = [query, output, done] () {
    auto res = query->GetResult("id:0");
    auto res_data = res->Raw<uint64_t>();
    auto data = output->flat<int64>().data();
    if (res->NumElements() == 0) {
      EULER_LOG(FATAL) << "SampleNode Result Size 0! "
          << "Maybe caused by empty node_type or bad filter condition";
    }
    std::copy(res_data, res_data + res->NumElements(), data);
    delete query;
    done();
  };

  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(Name("SampleNode").Device(DEVICE_CPU), SampleNode);

}  // namespace tensorflow
