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
#include "tf_euler/utils/sparse_tensor_builder.h"
#include "euler/common/str_util.h"

namespace tensorflow {

class GetGraphByLabel: public AsyncOpKernel {
 public:
  explicit GetGraphByLabel(OpKernelConstruction* ctx): AsyncOpKernel(ctx) { }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
};

void GetGraphByLabel::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto labels = ctx->input(0);
  auto labels_flat = labels.flat<string>();
  size_t labels_num = labels_flat.size();

  auto query = new euler::Query(
      "API_GET_GRAPH_BY_LABEL", "graphs", 2,
      {"labels"}, {});
  euler::Tensor* labels_t = query->AllocInput(
      "labels", {labels_num}, euler::DataType::kString);
  for (size_t i = 0; i < labels_num; ++i) {
    *(labels_t->Raw<std::string*>()[i]) = labels_flat(i);
  }

  auto callback = [done, query, ctx, labels_num, this]() {
    SparseTensorBuilder<int64, 2> nodes_builder;
    euler::Tensor* index = query->GetResult("graphs:0");
    euler::Tensor* value = query->GetResult("graphs:1");
    for (size_t i = 0; i < labels_num; ++i) {
      size_t start = index->Raw<int32_t>()[i * 2];
      size_t end = index->Raw<int32_t>()[i * 2 + 1];
      if (start == end) {
        nodes_builder.emplace({static_cast<int64>(i), 0}, 0);
      } else {
        for (size_t j = start; j < end; ++j) {
          nodes_builder.emplace(
              {static_cast<int64>(i), static_cast<int64>(j - start)},
              value->Raw<uint64_t>()[j]);
        }
      }
    }
    // Set id sparse tensor
    (void) ctx->set_output("nodes_indices", nodes_builder.indices());
    (void) ctx->set_output("nodes_values", nodes_builder.values());
    (void) ctx->set_output("nodes_shape", nodes_builder.dense_shape());

    delete query;
    done();
  };
  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("GetGraphByLabel").Device(DEVICE_CPU), GetGraphByLabel);

}  // namespace tensorflow
