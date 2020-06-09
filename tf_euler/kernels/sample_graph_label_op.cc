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

#include <string.h>

#include <memory>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tf_euler/utils/euler_query_proxy.h"
#include "euler/common/str_util.h"

namespace tensorflow {

class SampleGraphLabel: public AsyncOpKernel {
 public:
  explicit SampleGraphLabel(OpKernelConstruction* ctx): AsyncOpKernel(ctx) { }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
};

void SampleGraphLabel::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto batch_size = ctx->input(0);
  auto batch_size_flat = batch_size.flat<int32>();
  int32_t batch_size_num = batch_size_flat(0);

  Tensor* output = nullptr;
  TensorShape output_shape;
  output_shape.AddDim(batch_size_num);
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  auto query = new euler::Query(
      "API_SAMPLE_GRAPH_LABEL", "sample_graph", 1,
      {"count"}, {});
  euler::Tensor* count_t = query->AllocInput(
      "count", {1}, euler::DataType::kInt32);
  count_t->Raw<int32_t>()[0] = batch_size_num;

  auto callback = [output, done, query, batch_size_num, this]() {
    euler::Tensor* graph_labels = query->GetResult("sample_graph:0");
    std::string result_s(graph_labels->Raw<char>(),
                         graph_labels->NumElements());
    std::vector<std::string> results_vec = euler::Split(result_s, ",");
    if (results_vec.size() != batch_size_num) {
      EULER_LOG(FATAL) << "results_vec size != batch_size_num";
    }
    auto data = output->flat<tensorflow::string>();
    for (size_t i = 0; i < batch_size_num; ++i) {
      data(i) = results_vec[i];
    }
    delete query;
    done();
  };
  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("SampleGraphLabel").Device(DEVICE_CPU), SampleGraphLabel);

}  // namespace tensorflow
