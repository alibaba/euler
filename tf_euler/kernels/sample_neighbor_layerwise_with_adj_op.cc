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
#include <set>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tf_euler/utils/euler_query_proxy.h"
#include "tf_euler/utils/sparse_tensor_builder.h"
#include "euler/common/logging.h"
namespace tensorflow {

class SampleNeighborLayerwiseWithAdj: public AsyncOpKernel {
 public:
  explicit SampleNeighborLayerwiseWithAdj(OpKernelConstruction* ctx):
      AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("count", &count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("weight_func", &weight_func_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_node", &default_node_));
    // Build euler gremlin query
    std::stringstream ss;
    if (weight_func_.empty()) {
      ss << "v(nodes).sampleLNB(edge_types, n, m," << default_node_
          << ").as(nb)";
    } else {
      ss << "v(nodes).sampleLNB(edge_types, n, m," << weight_func_
          << "," << default_node_ << ").as(nb)";
    }
    query_str_ = ss.str();
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
 private:
  int count_;
  std::string weight_func_;
  int default_node_;
  std::string query_str_;
};

void SampleNeighborLayerwiseWithAdj::ComputeAsync(
    OpKernelContext* ctx, DoneCallback done) {
  // input
  auto nodes = ctx->input(0);
  auto edge_types = ctx->input(1);
  int batch_size = nodes.shape().dim_size(0);
  int last_layer_count = nodes.shape().dim_size(1);

  // output
  TensorShape output_shape;
  output_shape.AddDim(nodes.shape().dim_size(0));
  output_shape.AddDim(count_);

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  auto output_data = output->flat<int64>().data();
  auto output_size = output_shape.dim_size(0) * output_shape.dim_size(1);
  std::fill(output_data, output_data + output_size, default_node_);

  auto nodes_flat = nodes.flat<int64>();
  size_t nodes_size = nodes_flat.size();
  auto etypes_flat = edge_types.flat<int32>();
  size_t etypes_size = etypes_flat.size();

  // Build euler gremlin query
  auto query = new euler::Query(query_str_);
  auto t_nodes = query->AllocInput("nodes", {nodes_size}, euler::kUInt64);
  auto t_edge_types = query->AllocInput(
      "edge_types", {etypes_size}, euler::kInt32);
  auto t_last_layer_count = query->AllocInput("n", {1}, euler::kInt32);
  auto t_count = query->AllocInput("m", {1}, euler::kInt32);
  std::copy(nodes_flat.data(),
            nodes_flat.data() + nodes_flat.size(),
            t_nodes->Raw<int64_t>());
  std::copy(etypes_flat.data(),
            etypes_flat.data() + etypes_flat.size(),
            t_edge_types->Raw<int32_t>());

  *(t_count->Raw<int32_t>()) = count_;
  *(t_last_layer_count->Raw<int32_t>()) = last_layer_count;

  auto callback = [output_data, nodes_flat, last_layer_count,
       batch_size, done, query, ctx, this] () {
    std::vector<std::string> res_names = {"nb:0", "nb:1", "nb:2"};
    auto results_map = query->GetResult(res_names);
    auto nb_ptr = results_map["nb:2"];
    auto nb_data = nb_ptr->Raw<int64_t>();
    auto idx_ptr = results_map["nb:0"];
    auto idx_data = idx_ptr->Raw<int32_t>();
    auto val_ptr = results_map["nb:1"];
    auto val_data = val_ptr->Raw<int64_t>();
    std::copy(nb_data, nb_data + nb_ptr->NumElements(), output_data);
    SparseTensorBuilder<int64, 3> adj_builder;
    std::set<std::pair<int64_t, int64_t>> relation_set;

    for (size_t i = 0; i < batch_size; ++i) {
      relation_set.clear();
      for (size_t j = last_layer_count * i;
           j < last_layer_count * (i + 1); ++j) {
        int32_t begin = idx_data[j * 2];
        int32_t end = idx_data[j * 2 + 1];
        for (size_t k = begin; k < end; ++k) {
          relation_set.insert(std::make_pair(nodes_flat(j), val_data[k]));
        }
      }

      for (size_t j = 0; j < last_layer_count; ++j) {
        for (size_t k = 0; k < count_; ++k) {
          int64_t src_id = nodes_flat(j + last_layer_count * i);
          int64_t dst_id = nb_data[k + count_* i];
          if (relation_set.find(
                  std::make_pair(src_id, dst_id)) != relation_set.end()) {
            adj_builder.emplace(
                {static_cast<int64>(i),
                 static_cast<int64>(j),
                 static_cast<int64>(k)}, 1);
          } else if (j == last_layer_count -1 && k == count_ -1) {
            adj_builder.emplace(
                {static_cast<int64>(i),
                 static_cast<int64>(j),
                 static_cast<int64>(k)}, 0);
          }
        }
      }
    }

    (void) ctx->set_output("adj_indices", adj_builder.indices());
    (void) ctx->set_output("adj_values", adj_builder.values());
    (void) ctx->set_output("adj_shape", adj_builder.dense_shape());

    delete query;
    done();
  };

  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("SampleNeighborLayerwiseWithAdj").Device(DEVICE_CPU),
    SampleNeighborLayerwiseWithAdj);

}  // namespace tensorflow
