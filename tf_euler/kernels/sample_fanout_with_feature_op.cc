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
#include "tf_euler/utils/sparse_tensor_builder.h"
#include "euler/common/str_util.h"
#include "euler/common/logging.h"
#include "euler/common/data_types.h"

namespace tensorflow {
class SampleFanoutWithFeature: public AsyncOpKernel {
 public:
  explicit SampleFanoutWithFeature(OpKernelConstruction* ctx):
      AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("count", &count_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_node", &default_node_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_feature_names", &sfns_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("sparse_default_values",
                                     &default_values_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_feature_names", &dfns_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dense_dimensions", &dimensions_));
    // Build euler gremlin query
    size_t layer_cnt = count_.size();
    std::stringstream ss;
    ss << ".values(";
    for (size_t i = 0; i < dfns_.size(); ++i) {
      if (i != 0) {
        ss << ",";
      }
      ss << "__" << dfns_[i];
    }
    for (size_t i = 0; i < sfns_.size(); ++i) {
      ss << ",";
      ss << "__" << sfns_[i];
    }
    ss << ")";
    std::string feature_str = ss.str();

    ss.str("");
    ss << "v(nodes).as(nb_0)";
    for (size_t i = 1; i <= layer_cnt; ++i) {
      ss << ".sampleNB(et_" << i << ",nb_count_"
          << i << "," << default_node_ << ")" << ".as(nb_" << i << ")";
    }
    for (size_t i = 0; i <= layer_cnt; ++i) {
      ss << ".v_select(nb_" << i << ")" << feature_str << ".as(fea_"
          << i <<")";
    }
    query_str_ = ss.str();
    for (size_t i = 0; i <= layer_cnt; ++i) {
      if (i != 0) {
        for (size_t j = 0; j <= 3; ++j) {
          res_names_.push_back(euler::ToString("nb_", i, ":", j));
        }
      }
      for (size_t j = 0; j < (dfns_.size() + sfns_.size()) * 2; ++j) {
        res_names_.push_back(euler::ToString("fea_", i, ":", j));
      }
    }
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::vector<int> count_;
  std::vector<std::string> dfns_;  // dense feature names
  std::vector<std::string> sfns_;  // sparse feature names
  std::vector<int> dimensions_;
  std::vector<int> default_values_;
  int default_node_;
  std::string query_str_;
  std::vector<std::string> res_names_;
};

void SampleFanoutWithFeature::ComputeAsync(
    OpKernelContext* ctx, DoneCallback done) {
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
  std::copy(nodes_flat.data(), nodes_flat.data() + nodes_flat.size(),
            t_nodes->Raw<int64_t>());
  for (size_t i = 0; i < layer_cnt; ++i) {
    auto t_edge_types = query->AllocInput(euler::ToString("et_", i + 1),
                                          {etypes_size}, euler::kInt32);
    auto t_count = query->AllocInput(
        euler::ToString("nb_count_", i + 1), {1}, euler::kInt32);
    for (size_t j = 0; j < etypes_size; ++j) {
      t_edge_types->Raw<int32_t>()[j] = etypes_flat(i * etypes_size + j);
    }
    *(t_count->Raw<int32_t>()) = count_[i];
  }

  for (size_t i = 0; i < dfns_.size(); ++i) {
    auto t_fid = query->AllocInput("__" + dfns_[i], {1}, euler::kString);
    *(t_fid->Raw<std::string*>()[0]) = "dense_" + dfns_[i];
  }
  for (size_t i = 0; i < sfns_.size(); ++i) {
    auto t_fid = query->AllocInput("__" + sfns_[i], {1}, euler::kString);
    *(t_fid->Raw<std::string*>()[0]) = "sparse_" + sfns_[i];
  }

  // build outputs tensor
  std::vector<Tensor*> outputs_node(layer_cnt, nullptr);
  std::vector<Tensor*> outputs_weight(layer_cnt, nullptr);
  std::vector<Tensor*> outputs_type(layer_cnt, nullptr);
  std::vector<Tensor*> outputs_dense((layer_cnt + 1) * dfns_.size(), nullptr);

  size_t offset = 3 * layer_cnt;
  for (size_t j = 0; j < dfns_.size(); ++j) {
    TensorShape output_dense_shape;
    output_dense_shape.AddDim(nodes_size);
    output_dense_shape.AddDim(dimensions_[j]);
    auto idx = j;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
            offset + idx, output_dense_shape, &outputs_dense[idx]));
    auto data_dense = outputs_dense[idx]->flat<float>().data();
    auto end = data_dense + nodes_size* dimensions_[j];
    std::fill(data_dense, end, 0.0);
  }
  for (size_t i = 0; i < layer_cnt; ++i) {
    TensorShape output_shape;
    output_shape.AddDim(nodes_size);
    auto output_size = nodes_size;
    for (size_t j = 0 ; j <= i ; ++j) {
      output_shape.AddDim(count_[j]);
      output_size *= count_[j];
    }
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
            i, output_shape, &outputs_node[i]));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
            layer_cnt + i, output_shape, &outputs_weight[i]));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(
            2 * layer_cnt + i, output_shape, &outputs_type[i]));
    auto data_node = outputs_node[i]->flat<int64>().data();
    std::fill(data_node, data_node + output_size, default_node_);
    auto data_weight = outputs_weight[i]->flat<float>().data();
    std::fill(data_weight, data_weight + output_size, 0.0);
    auto data_type = outputs_type[i]->flat<int32>().data();
    std::fill(data_type, data_type + output_size, -1);
    for (size_t j = 0; j < dfns_.size(); ++j) {
      TensorShape output_dense_shape;
      output_dense_shape.AddDim(output_size);
      output_dense_shape.AddDim(dimensions_[j]);
      auto idx = (i + 1) * dfns_.size() + j;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(
              offset + idx, output_dense_shape, &outputs_dense[idx]));
      auto data_dense = outputs_dense[idx]->flat<float>().data();
      auto end = data_dense + output_size * dimensions_[j];
      std::fill(data_dense, end, 0.0);
    }
  }

  auto callback = [ctx, outputs_node, outputs_weight,
       outputs_type, outputs_dense, layer_cnt, done,
       query, nodes_size, this] () {
    auto results_map = query->GetResult(res_names_);
    size_t output_size = nodes_size;
    std::vector<SparseTensorBuilder<int64, 2>>
        builders((layer_cnt + 1) * sfns_.size());
    for (size_t i = 0; i <= layer_cnt; ++i) {
      // fill neighbor data
      if (i > 0) {
        output_size *= count_[i-1];
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
            std::copy(
                nb_data + start, nb_data + end,
                outputs_node[i - 1]->flat<int64>().data() +
                j * count_[i - 1]);
            std::copy(
                wei_data + start, wei_data + end,
                outputs_weight[i - 1]->flat<float>().data() +
                j * count_[i - 1]);
            std::copy(
                typ_data + start, typ_data + end,
                outputs_type[i - 1]->flat<int32>().data() +
                j * count_[i - 1]);
          }
        }
      }
      // fill dense feature data
      for (size_t j = 0; j < dfns_.size(); ++j) {
        auto res = results_map[euler::ToString(
            "fea_", i, ":", j * 2 + 1)];
        auto res_data = res->Raw<float>();
        auto idx_ptr = results_map[euler::ToString(
            "fea_", i, ":", j * 2)];
        auto idx_data = idx_ptr->Raw<int32_t>();
        for (size_t k = 0; k < output_size; ++k) {
          size_t start = idx_data[k * 2];
          size_t end = idx_data[k * 2 + 1];
          if (start < end) {
            std::copy(res_data + start, res_data + end,
                  outputs_dense[i * dfns_.size() + j]->flat<float>().data()
                  + k * dimensions_[j]);
          }
        }
      }
      // build sparse feature data
      for (size_t j = dfns_.size(); j < dfns_.size() + sfns_.size(); ++j) {
        std::string fea_idx = euler::ToString("fea_", i, ":", j * 2);
        std::string fea_val = euler::ToString("fea_", i, ":", j * 2 + 1);
        for (size_t k = 0; k < output_size; ++k) {
          size_t start = results_map[fea_idx]->Raw<int32_t>()[k * 2];
          size_t end = results_map[fea_idx]->Raw<int32_t>()[k * 2 + 1];
          if (start == end) {
            builders[i * sfns_.size() + j - dfns_.size()]
                .emplace({static_cast<int64>(k), 0},
            default_values_[j - dfns_.size()]);
          } else {
            for (size_t l = start; l < end; ++l) {
              builders[i * sfns_.size() + j - dfns_.size()]
                  .emplace({static_cast<int64>(k),
              static_cast<int64>(l - start)},
              results_map[fea_val]->Raw<uint64_t>()[l]);
            }
          }
        }
      }
    }
    // fill sparse feature data
    OpOutputList indices, values, shape;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("indices", &indices), done);
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("values", &values), done);
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("dense_shape", &shape), done);
    for (size_t i = 0; i < (layer_cnt + 1) * sfns_.size(); ++i) {
      indices.set(i, builders[i].indices());
      values.set(i, builders[i].values());
      shape.set(i, builders[i].dense_shape());
    }
    delete query;
    done();
  };
  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("SampleFanoutWithFeature").Device(DEVICE_CPU),
    SampleFanoutWithFeature);

}  // namespace tensorflow
