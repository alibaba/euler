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
#include <sstream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tf_euler/utils/euler_query_proxy.h"

namespace tensorflow {
class SampleNeighbor: public AsyncOpKernel {
 public:
  explicit SampleNeighbor(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("count", &count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_node", &default_node_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("condition", &condition_));
    // Build euler gremlin query
    std::stringstream ss;
    if (condition_.empty()) {
      ss << "v(nodes).sampleNB(edge_types, nb_count,"
        << default_node_ << ").as(nb)";
    } else {
      ss << "v(nodes).sampleNB(edge_types, nb_count,"
        << default_node_ << ").has(" << condition_ << ").as(nb)";
    }
    query_str_ = ss.str();
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  int count_;
  int default_node_;
  std::string query_str_;
  std::string condition_;
};

void SampleNeighbor::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0);
  auto edge_types = ctx->input(1);

  auto nodes_flat = nodes.flat<int64>();
  size_t nodes_size = nodes_flat.size();
  auto etypes_flat = edge_types.flat<int32>();
  size_t etypes_size = etypes_flat.size();

  // Output
  TensorShape output_shape;
  output_shape.AddDim(nodes.shape().dim_size(0));
  output_shape.AddDim(count_);

  Tensor* output = nullptr;
  Tensor* weights = nullptr;
  Tensor* types = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(1, output_shape, &weights));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(2, output_shape, &types));

  auto output_data = output->flat<int64>().data();
  auto weights_data = weights->flat<float>().data();
  auto types_data = types->flat<int32>().data();
  auto output_size = output_shape.dim_size(0) * output_shape.dim_size(1);
  std::fill(output_data, output_data + output_size, default_node_);
  std::fill(weights_data, weights_data + output_size, 0.0);
  std::fill(types_data, types_data + output_size, -1);

  // Build euler gremlin query
  auto query =
      new euler::Query(query_str_);
  auto t_nodes =
      query->AllocInput("nodes", {nodes_size}, euler::kUInt64);
  auto t_edge_types =
      query->AllocInput("edge_types", {etypes_size}, euler::kInt32);

  auto t_count =
      query->AllocInput("nb_count", {1}, euler::kInt32);
  std::copy(nodes_flat.data(),
            nodes_flat.data() + nodes_flat.size(), t_nodes->Raw<int64_t>());
  std::copy(etypes_flat.data(),
            etypes_flat.data() + etypes_flat.size(),
            t_edge_types->Raw<int32_t>());
  *(t_count->Raw<int32_t>()) = count_;

  int count = count_;
  auto callback = [output_data, weights_data, types_data,
                   output_size, done, query, nodes_size, count] () {
    std::vector<std::string> res_names = {"nb:0", "nb:1", "nb:2", "nb:3"};
    auto results_map = query->GetResult(res_names);
    auto idx_t = results_map["nb:0"];
    auto nb_ptr = results_map["nb:1"];
    auto wei_ptr = results_map["nb:2"];
    auto type_ptr = results_map["nb:3"];
    auto idx_data = idx_t->Raw<int32_t>();
    auto nb_data = nb_ptr->Raw<int64_t>();
    auto wei_data = wei_ptr->Raw<float>();
    auto typ_data = type_ptr->Raw<int32_t>();

    for (size_t i = 0; i < nodes_size; ++i) {
      int start = idx_data[2 * i];
      int end = idx_data[2 * i + 1];
      if (nb_data[start] != euler::common::DEFAULT_UINT64) {
        std::copy(nb_data + start, nb_data + end, output_data + i * count);
        std::copy(wei_data + start, wei_data + end, weights_data + i * count);
        std::copy(typ_data + start, typ_data + end, types_data + i * count);
      }
    }

    delete query;
    done();
  };

  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("SampleNeighbor").Device(DEVICE_CPU), SampleNeighbor);

}  // namespace tensorflow
