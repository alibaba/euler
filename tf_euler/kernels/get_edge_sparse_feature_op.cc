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

namespace tensorflow {


class GetEdgeSparseFeature: public AsyncOpKernel {
 public:
  explicit GetEdgeSparseFeature(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_names", &feature_names_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("default_values", &default_values_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    OP_REQUIRES(ctx, default_values_.size() == feature_names_.size(),
                errors::InvalidArgument(
                    "Require default_values.size() == feature_names_.size()"));
    OP_REQUIRES(ctx, static_cast<size_t>(N_) == feature_names_.size(),
                errors::InvalidArgument("Require N == feature_names.size()"));
    std::stringstream ss;
    ss << "e(edges).values(";
    for (size_t i = 0; i < feature_names_.size(); ++i) {
      if (i != 0) {
        ss << ",";
      }
      ss << "__" << feature_names_[i];
    }
    ss << ").as(fea)";
    query_str_ = ss.str();
    for (size_t i = 0; i < feature_names_.size() * 2; ++i) {
      ss.str("");
      ss << "fea:" << i;
      res_names_.emplace_back(ss.str());
    }
  }


  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  std::vector<std::string> feature_names_;
  std::vector<int> default_values_;
  int N_;
  std::string query_str_;
  std::vector<std::string> res_names_;
};


void GetEdgeSparseFeature::ComputeAsync(OpKernelContext* ctx,
                                        DoneCallback done) {
  auto edges = ctx->input(0);
  auto shape = edges.shape();
  OP_REQUIRES(ctx, shape.dim_size(1) == 3,
              errors::InvalidArgument(
                  "Input `edges` must be a matrix with shape [n, 3]"));
  auto edges_flat = edges.flat<int64>();
  size_t edges_size = edges_flat.size();
  size_t edge_num  = edges_size / 3;
  auto query = new euler::Query(query_str_);
  auto t_edges = query->AllocInput("edges", {edge_num, 3}, euler::kUInt64);
  std::copy(edges_flat.data(), edges_flat.data() + edges_size,
            t_edges->Raw<int64_t>());
  for (size_t i = 0; i < feature_names_.size(); ++i) {
    auto t_fid = query->AllocInput("__" + feature_names_[i], {1},
                                   euler::kString);
    *(t_fid->Raw<std::string*>()[0]) = "sparse_" + feature_names_[i];
  }

  auto callback = [ctx, done, this, query, edge_num] () {
    std::stringstream ss;
    auto results_map = query->GetResult(res_names_);
    std::vector<SparseTensorBuilder<int64, 2>> builders(N_);
    for (size_t i = 0 ; i < feature_names_.size(); ++i) {
      ss.str("");
      ss << "fea:" << i * 2;
      std::string fea_idx = ss.str();

      ss.str("");
      ss << "fea:" << i * 2 + 1;
      std::string fea_val = ss.str();
      if (results_map[fea_idx]->NumElements() != edge_num * 2) {
        EULER_LOG(FATAL) << "Sparse Feature Result Index Num Error: " <<
            results_map[fea_idx]->NumElements()
            << "Expect: " << edge_num * 2;
      }

      for (size_t j = 0; j < edge_num; ++j) {
        size_t start = results_map[fea_idx]->Raw<int32_t>()[j * 2];
        size_t end = results_map[fea_idx]->Raw<int32_t>()[j * 2 + 1];
        if (start == end) {
          builders[i].emplace({static_cast<int64>(j), 0},
          default_values_[i]);
        } else {
          for (size_t k = start; k < end; ++k) {
            builders[i].emplace({static_cast<int64>(j),
            static_cast<int64>(k - start)},
            results_map[fea_val]->Raw<uint64_t>()[k]);
          }
        }
      }
    }

    OpOutputList indices, values, shape;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("indices", &indices), done);
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("values", &values), done);
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("dense_shape", &shape), done);
    for (size_t j = 0; j < N_; ++j) {
      indices.set(j, builders[j].indices());
      values.set(j, builders[j].values());
      shape.set(j, builders[j].dense_shape());
    }
    delete query;
    done();
  };

  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("GetEdgeSparseFeature").Device(DEVICE_CPU), GetEdgeSparseFeature);

}  // namespace tensorflow
