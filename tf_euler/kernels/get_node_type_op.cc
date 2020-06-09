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

namespace tensorflow {

class GetNodeType : public AsyncOpKernel {
 public:
  explicit GetNodeType(OpKernelConstruction* ctx): AsyncOpKernel(ctx) { }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
};

void GetNodeType::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0);
  auto nodes_data = nodes.flat<int64>().data();
  size_t nodes_size = nodes.flat<int64>().size();

  TensorShape output_shape;
  output_shape.AddDim(nodes_size);
  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  auto query = new euler::Query("v(nodes).label().as(l)");
  auto t_nodes = query->AllocInput("nodes", {nodes_size}, euler::kUInt64);
  std::copy(nodes_data, nodes_data + nodes_size, t_nodes->Raw<int64_t>());

  auto callback = [query, output, done] () {
    auto data = output->flat<int32>().data();
    auto res = query->GetResult("l:0");
    auto res_data = res->Raw<int32_t>();
    std::copy(res_data, res_data + res->NumElements(), data);
    delete query;
    done();
  };
  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(Name("GetNodeType").Device(DEVICE_CPU), GetNodeType);

class GetNodeTypeId : public OpKernel {
 public:
  explicit GetNodeTypeId(OpKernelConstruction* ctx): OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override {
    auto type_names = ctx->input(0);
    auto data = type_names.flat<std::string>();
    const auto& node_type_map =
        euler::QueryProxy::GetInstance()->graph_meta().node_type_map();
    if (data(0) != "-1") {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, type_names.shape(), &output));

      auto output_data = output->flat<int32>().data();
      for (int i = 0; i < data.size(); ++i) {
        auto it = node_type_map.find(data(i));
        OP_REQUIRES(ctx, (it != node_type_map.end()),
                    errors::InvalidArgument(
                        "Invalid node type name: ", data(i)));
        output_data[i] = it->second;
      }
    } else {
      TensorShape output_shape;
      output_shape.AddDim(node_type_map.size());
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
      auto output_data = output->flat<int32>().data();
      size_t idx = 0;
      for (auto it = node_type_map.begin(); it != node_type_map.end(); ++it) {
        output_data[idx++] = it->second;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("GetNodeTypeId").Device(DEVICE_CPU), GetNodeTypeId);

class GetEdgeTypeId : public OpKernel {
 public:
  explicit GetEdgeTypeId(OpKernelConstruction* ctx): OpKernel(ctx) { }

  void Compute(OpKernelContext* ctx) override {
    auto type_names = ctx->input(0);
    auto data = type_names.flat<std::string>();

    const auto& edge_type_map =
        euler::QueryProxy::GetInstance()->graph_meta().edge_type_map();
    if (data(0) != "-1") {
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, type_names.shape(), &output));

      auto output_data = output->flat<int32>().data();
      for (int i = 0; i < data.size(); ++i) {
        auto it = edge_type_map.find(data(i));
        OP_REQUIRES(ctx, (it != edge_type_map.end()),
                    errors::InvalidArgument(
                        "Invalid edge type name: ", data(i)));
        output_data[i] = it->second;
      }
    } else {
      TensorShape output_shape;
      output_shape.AddDim(edge_type_map.size());
      Tensor* output = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
      auto output_data = output->flat<int32>().data();
      size_t idx = 0;
      for (auto it = edge_type_map.begin(); it != edge_type_map.end(); ++it) {
        output_data[idx++] = it->second;
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("GetEdgeTypeId").Device(DEVICE_CPU), GetEdgeTypeId);


}  // namespace tensorflow
