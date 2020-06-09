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

class SampleNWithTypes: public AsyncOpKernel {
 public:
  explicit SampleNWithTypes(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    query_str_ = "sampleNWithTypes(types, counts).as(n)";
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
 private:
  std::string query_str_;
};

void SampleNWithTypes::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto count = ctx->input(0);
  int32_t count_value = (count.scalar<int32>())();
  auto node_types = ctx->input(1);
  auto node_types_flat = node_types.flat<int32>();
  size_t types_size = node_types_flat.size();

  OP_REQUIRES_ASYNC(ctx, TensorShapeUtils::IsScalar(count.shape()),
      errors::InvalidArgument("count must be a scalar, saw shape: ",
      count.shape().DebugString()), done);

  auto query = new euler::Query(query_str_);
  auto t_types = query->AllocInput("types", {types_size}, euler::kInt32);
  auto t_counts = query->AllocInput("counts", {types_size}, euler::kInt32);
  for (size_t i = 0; i < types_size; i++) {
    t_types->Raw<int32_t>()[i] = node_types_flat(i);
  }
  for (size_t i = 0; i < types_size; i++) {
    t_counts->Raw<int32_t>()[i] = count_value;
  }

  TensorShape output_shape;
  output_shape.AddDim(types_size);
  output_shape.AddDim(count_value);

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  auto callback = [types_size, count_value, query, output, done]() {
    auto res = query->GetResult("n:1");
    auto res_data = res->Raw<uint64_t>();
    auto data = output->flat<int64>().data();
    if (res->NumElements() != types_size * count_value) {
      EULER_LOG(FATAL) << "samples size error, invalid node types!";
    }
    std::copy(res_data, res_data + res->NumElements(), data);
    delete query;
    done();
  };
  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("SampleNWithTypes").Device(DEVICE_CPU), SampleNWithTypes);

}  // namespace tensorflow
