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
#include "euler/common/str_util.h"
#include "euler/common/logging.h"
#include "euler/common/data_types.h"

namespace tensorflow {
class SampleFanout: public AsyncOpKernel {
 public:
  explicit SampleFanout(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("count", &count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_node", &default_node_));
    // Build euler gremlin query
    std::stringstream ss;
    ss << "v(nodes)";
    size_t layer_cnt = count_.size();
    for (size_t i = 0; i < layer_cnt; ++i) {
      ss << ".sampleNB(et_" << i << ",nb_count_"
          << i << "," << default_node_ << ")" << ".as(nb_" << i << ")";
    }
    query_str_ = ss.str();
    for (size_t i = 0; i < layer_cnt; ++i) {
      for (size_t j = 0; j <= 3; ++j) {
       res_names_.push_back(euler::ToString("nb_", i, ":", j));
      }
    }
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::vector<int> count_;
  int default_node_;
  std::string query_str_;
  std::vector<std::string> res_names_;
};

void SampleFanout::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0);
  // edge_types rank == 2
  auto edge_types = ctx->input(1);
  auto layer_cnt = edge_types.shape().dim_size(0);

  auto nodes_flat = nodes.flat<int64>();
  size_t nodes_size = nodes_flat.size();
  auto etypes_flat = edge_types.flat<int32>();
  size_t etypes_size = etypes_flat.size() / layer_cnt;

  auto query = new euler::Query(query_str_);
  auto t_nodes = query->AllocInput("nodes", {nodes_size}, euler::kUInt64);
  for (size_t i = 0; i < nodes_size; ++i) {
    t_nodes->Raw<int64_t>()[i] = nodes_flat(i);
  }
  for (size_t i = 0; i < layer_cnt; ++i) {
    auto t_edge_types = query->AllocInput(euler::ToString("et_", i),
                                          {etypes_size}, euler::kInt32);
    auto t_count = query->AllocInput(euler::ToString("nb_count_", i),
                                     {1}, euler::kInt32);

    for (size_t j = 0; j < etypes_size; ++j) {
      t_edge_types->Raw<int32_t>()[j] = etypes_flat(i * etypes_size + j);
    }
    *(t_count->Raw<int32_t>()) = count_[i];
  }

  std::vector<Tensor*> outputs_node(layer_cnt, nullptr);
  std::vector<Tensor*> outputs_weight(layer_cnt, nullptr);
  std::vector<Tensor*> outputs_type(layer_cnt, nullptr);

  for (size_t i = 0; i < layer_cnt; ++i) {
    TensorShape output_shape;
    output_shape.AddDim(nodes_size);
    auto output_size = nodes_size;
    for (size_t j = 0; j <= i; ++j) {
      output_shape.AddDim(count_[j]);
      output_size *= count_[j];
    }
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(i, output_shape, &outputs_node[i]));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            layer_cnt + i, output_shape, &outputs_weight[i]));
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(
            2 * layer_cnt + i, output_shape, &outputs_type[i]));
    auto data_node = outputs_node[i]->flat<int64>().data();
    std::fill(data_node, data_node + output_size, default_node_);
    auto data_weight = outputs_weight[i]->flat<float>().data();
    std::fill(data_weight, data_weight + output_size, 0.0);
    auto data_type = outputs_type[i]->flat<int32>().data();
    std::fill(data_type, data_type + output_size, -1);
  }

  auto callback = [outputs_node, outputs_weight,
       outputs_type, layer_cnt, done, query, nodes_size, this] () {
    auto results_map = query->GetResult(res_names_);
    for (size_t i = 0; i < layer_cnt; ++i) {
      auto idx_ptr = results_map[euler::ToString("nb_", i, ":0")];
      auto nb_ptr = results_map[euler::ToString("nb_", i, ":1")];
      auto wei_ptr = results_map[euler::ToString("nb_", i, ":2")];
      auto type_ptr = results_map[euler::ToString("nb_", i, ":3")];
      auto idx_data = idx_ptr->Raw<int32_t>();
      auto nb_data = nb_ptr->Raw<int64_t>();
      auto wei_data = wei_ptr->Raw<float>();
      auto typ_data = type_ptr->Raw<int32_t>();
      for (size_t j = 0; j < idx_ptr->NumElements() / 2; ++j) {
        int start = idx_data[2 * j];
        int end = idx_data[2 * j + 1];
        if (nb_data[start] != euler::common::DEFAULT_UINT64) {
          std::copy(nb_data + start, nb_data + end,
                    outputs_node[i]->flat<int64>().data() + j * count_[i]);
          std::copy(wei_data + start, wei_data + end,
                    outputs_weight[i]->flat<float>().data() + j * count_[i]);
          std::copy(typ_data + start, typ_data + end,
                    outputs_type[i]->flat<int32>().data() + j * count_[i]);
        }
      }
    }
    delete query;
    done();
  };
  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("SampleFanout").Device(DEVICE_CPU), SampleFanout);

}  // namespace tensorflow
