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

#include "tf_euler/utils/sparse_tensor_builder.h"
#include "tf_euler/utils/euler_query_proxy.h"

namespace tensorflow {


class GetSortedFullNeighbor: public AsyncOpKernel {
 public:
  explicit GetSortedFullNeighbor(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("condition", &condition_));
    std::stringstream ss;
    for (size_t i = 0; i < 4; ++i) {
      ss.str("");
      ss << "nb:" << i;
      res_names_.emplace_back(ss.str());
    }
    ss.str("");
    if (!condition_.empty()) {
      ss << "v(nodes).outV(edge_types).has(" << condition_
          << ").order_by(id, asc).as(nb)";
      query_str_ = ss.str();
    } else {
      query_str_ = "v(nodes).outV(edge_types).order_by(id,asc).as(nb)";
    }
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::vector<std::string> res_names_;
  std::string query_str_;
  std::string condition_;
};

void GetSortedFullNeighbor::ComputeAsync(
    OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0);
  auto edge_types = ctx->input(1);

  auto nodes_flat = nodes.flat<int64>();
  size_t nodes_size = nodes_flat.size();

  auto etypes_flat = edge_types.flat<int32>();
  size_t etypes_size = etypes_flat.size();

  // Build Euler query
  auto query = new euler::Query(query_str_);

  auto t_nodes = query->AllocInput("nodes", {nodes_size}, euler::kUInt64);
  auto t_edge_types = query->AllocInput("edge_types",
                                        {etypes_size}, euler::kInt32);
  for (size_t i = 0; i < nodes_size; i++) {
    t_nodes->Raw<int64_t>()[i] = nodes_flat(i);
  }
  for (size_t i = 0; i < etypes_size; i++) {
    t_edge_types->Raw<int32_t>()[i] = etypes_flat(i);
  }

  auto callback =
      [ctx, done, nodes_size, query, this] () {
        auto results_map = query->GetResult(res_names_);
        SparseTensorBuilder<int64, 2> id_builder;
        SparseTensorBuilder<float, 2> weight_builder;
        SparseTensorBuilder<int32, 2> type_builder;
        auto id_idx = results_map["nb:0"];
        auto id_val = results_map["nb:1"];
        auto wei_val = results_map["nb:2"];
        auto type_val = results_map["nb:3"];

        if (id_idx->NumElements() != nodes_size * 2) {
          EULER_LOG(FATAL) << "Sparse Feature Result Index Num Error:"
              << id_idx->NumElements()
              <<"Expect: "<< nodes_size * 2;
        }
        for (size_t i = 0; i < nodes_size; i++) {
          size_t start = id_idx->Raw<int32_t>()[i * 2];
          size_t end = id_idx->Raw<int32_t>()[i * 2 + 1];
          for (size_t j = start; j < end; ++j) {
            id_builder.emplace(
                {static_cast<int64>(i), static_cast<int64>(j - start)},
                 id_val->Raw<uint64_t>()[j]);
            weight_builder.emplace(
                {static_cast<int64>(i), static_cast<int64>(j - start)},
                 wei_val->Raw<float>()[j]);
            type_builder.emplace(
                {static_cast<int64>(i), static_cast<int64>(j - start)},
                 type_val->Raw<int32_t>()[j]);
          }
        }

        // Set id sparse tensor
        (void) ctx->set_output("id_indices", id_builder.indices());
        (void) ctx->set_output("id_values", id_builder.values());
        (void) ctx->set_output("id_shape", id_builder.dense_shape());

        // set weight sparse tensor
        (void) ctx->set_output("weight_indices", weight_builder.indices());
        (void) ctx->set_output("weight_values", weight_builder.values());
        (void) ctx->set_output("weight_shape", weight_builder.dense_shape());

        // set type sparse tensor
        (void) ctx->set_output("type_indices", type_builder.indices());
        (void) ctx->set_output("type_values", type_builder.values());
        (void) ctx->set_output("type_shape", type_builder.dense_shape());

        delete query;
        done();
      };
  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("GetSortedFullNeighbor").Device(DEVICE_CPU), GetSortedFullNeighbor);

}  // namespace tensorflow
