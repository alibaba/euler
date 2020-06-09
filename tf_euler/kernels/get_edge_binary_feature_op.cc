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

class GetEdgeBinaryFeature: public AsyncOpKernel {
 public:
  explicit GetEdgeBinaryFeature(OpKernelConstruction* ctx):
      AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_names", &feature_names_));
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
  int64 N_;
  std::vector<std::string> feature_names_;
  std::string query_str_;
  std::vector<std::string> res_names_;
};

void GetEdgeBinaryFeature::ComputeAsync(OpKernelContext* ctx,
                                        DoneCallback done) {
  auto edges = ctx->input(0);
  auto& shape = edges.shape();
  OP_REQUIRES(ctx, shape.dim_size(1) == 3,
              errors::InvalidArgument(
                  "Input `edges` must be a matrix with shape [n, 3]"));

  std::vector<Tensor*> outputs(N_, nullptr);
  TensorShape output_shape;
  output_shape.AddDim(shape.dim_size(0));
  for (auto i = 0; i < N_; ++i) {
    OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &outputs[i]));
  }

  auto edges_flat = edges.flat<int64>();
  size_t edges_size = edges_flat.size();
  size_t edge_num  = edges_size / 3;

  auto query = new euler::Query(query_str_);
  auto t_edges = query->AllocInput("edges", {edge_num, 3}, euler::kUInt64);
  std::copy(edges_flat.data(),
            edges_flat.data() + edges_size, t_edges->Raw<int64_t>());
  for (size_t i = 0; i < feature_names_.size(); ++i) {
    auto t_fid = query->AllocInput("__" + feature_names_[i], {1},
                                   euler::kString);
    *(t_fid->Raw<std::string*>()[0]) = "binary_" + feature_names_[i];
  }

  auto callback = [outputs, done, query, edge_num, this]() {
    std::stringstream ss;
    auto results_map = query->GetResult(res_names_);
    for (size_t i = 0 ; i < feature_names_.size(); ++i) {
      ss.str("");
      ss << "fea:" << i * 2;
      std::string fea_idx = ss.str();

      ss.str("");
      ss << "fea:" << i * 2 + 1;
      std::string fea_val = ss.str();

      if (results_map[fea_idx]->NumElements() != edge_num * 2) {
        EULER_LOG(FATAL) << "Binary Feature Result Index Num Error:" <<
            results_map[fea_idx]->NumElements() << "Expect: " << edge_num * 2;
      }
      for (size_t j = 0; j < edge_num; ++j) {
        size_t start = results_map[fea_idx]->Raw<int32_t>()[j * 2];
        size_t end = results_map[fea_idx]->Raw<int32_t>()[j * 2 + 1];
        auto data = outputs[i]->flat<tensorflow::string>();
        std::string f_v(end - start, 0);
        std::copy(results_map[fea_val]->Raw<char>() + start,
                  results_map[fea_val]->Raw<char>() + end,
                  f_v.begin());
        data(j) = f_v;
      }
    }
    delete query;
    done();
  };

  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("GetEdgeBinaryFeature").Device(DEVICE_CPU),
    GetEdgeBinaryFeature);

}  // namespace tensorflow
