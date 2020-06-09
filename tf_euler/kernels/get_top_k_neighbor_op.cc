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


class GetTopKNeighbor: public AsyncOpKernel {
 public:
  explicit GetTopKNeighbor(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("k", &k_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("condition", &condition_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_node", &default_node_));
    std::stringstream ss;
    if (!condition_.empty()) {
      ss << "v(nodes).outV(edge_types).has("
          << condition_ << ").order_by(weight, desc).limit("
          << k_ << ").as(nb)";
      query_str_ = ss.str();
    } else {
      ss << "v(nodes).outV(edge_types).order_by(weight, desc).limit("
          << k_ << ").as(nb)";
      query_str_ = ss.str();
    }
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  int k_;
  int default_node_;
  std::string query_str_;
  std::string condition_;
};

void GetTopKNeighbor::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0);
  auto edge_types = ctx->input(1);
  TensorShape output_shape;
  output_shape.AddDim(nodes.shape().dim_size(0));
  output_shape.AddDim(k_);

  Tensor* output = nullptr;
  Tensor* weights = nullptr;
  Tensor* types = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(1, output_shape, &weights));
  OP_REQUIRES_OK(ctx, ctx->allocate_output(2, output_shape, &types));

  auto nodes_flat = nodes.flat<int64>();
  size_t nodes_size = nodes_flat.size();

  auto etypes_flat = edge_types.flat<int32>();
  size_t etypes_size = etypes_flat.size();

  auto output_data = output->flat<int64>().data();
  auto weights_data = weights->flat<float>().data();
  auto types_data = types->flat<int32>().data();
  auto output_size = output_shape.dim_size(0) * output_shape.dim_size(1);
  std::fill(output_data, output_data + output_size, default_node_);
  std::fill(weights_data, weights_data + output_size, 0.0);
  std::fill(types_data, types_data + output_size, -1);

  // Build Euler query
  auto query = new euler::Query(query_str_);
  auto t_nodes = query->AllocInput("nodes", {nodes_size}, euler::kUInt64);
  auto t_edge_types = query->AllocInput(
      "edge_types", {etypes_size}, euler::kInt32);

  for (size_t i = 0; i < nodes_size; i++) {
    t_nodes->Raw<int64_t>()[i] = nodes_flat(i);
  }
  for (size_t i = 0; i < etypes_size; i++) {
    t_edge_types->Raw<int32_t>()[i] = etypes_flat(i);
  }

  auto callback = [output_data,
       weights_data, nodes_size, types_data,
       output_size, done, query, this] () {
    std::vector<std::string> res_names = {"nb:0", "nb:1", "nb:2", "nb:3"};
    auto results_map = query->GetResult(res_names);
    auto idx_ptr = results_map["nb:0"];
    auto nb_ptr = results_map["nb:1"];
    auto wei_ptr = results_map["nb:2"];
    auto type_ptr = results_map["nb:3"];
    if (idx_ptr->NumElements() != nodes_size * 2) {
          EULER_LOG(FATAL) << "Sparse Feature Result Index Num Error:"
              << idx_ptr ->NumElements()
              << "Expect: " << nodes_size * 2;
    }

    for (size_t i = 0; i < nodes_size; i++) {
      size_t start = idx_ptr->Raw<int32_t>()[i * 2];
      size_t end = idx_ptr->Raw<int32_t>()[i * 2 + 1];
      for (size_t j = start; j < end; ++j) {
        output_data[i * k_ + j - start] = nb_ptr->Raw<int64_t>()[j];
        weights_data[i * k_ + j - start] = wei_ptr->Raw<float>()[j];
        types_data[i * k_ + j - start] = type_ptr->Raw<int32_t>()[j];
      }
    }
    delete query;
    done();
  };
  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}


REGISTER_KERNEL_BUILDER(
    Name("GetTopKNeighbor").Device(DEVICE_CPU), GetTopKNeighbor);

}  // namespace tensorflow
