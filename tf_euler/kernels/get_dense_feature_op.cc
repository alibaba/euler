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
#include <sstream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "euler/common/logging.h"
#include "tf_euler/utils/euler_query_proxy.h"

namespace tensorflow {

class GetDenseFeature: public AsyncOpKernel {
 public:
  explicit GetDenseFeature(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("feature_names", &feature_names_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dimensions", &dimensions_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    std::stringstream ss;
    ss << "v(nodes).values(";
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
  std::vector<int> dimensions_;
  std::string query_str_;
  std::vector<std::string> res_names_;
  int N_;
};

void GetDenseFeature::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0);
  auto& shape = nodes.shape();
  std::vector<Tensor*> outputs(N_, nullptr);
  for (int i = 0; i < N_; ++i) {
    TensorShape output_shape;
    output_shape.AddDim(shape.dim_size(0));
    output_shape.AddDim(dimensions_[i]);
    OP_REQUIRES_OK(ctx, ctx->allocate_output(i, output_shape, &outputs[i]));
    auto data = outputs[i]->flat<float>().data();
    auto end = data + shape.dim_size(0) * dimensions_[i];
    std::fill(data, end, 0.0);
  }
  auto nodes_flat = nodes.flat<int64>();
  size_t nodes_size = nodes_flat.size();
  auto query = new euler::Query(query_str_);
  auto t_nodes = query->AllocInput("nodes", {nodes_size}, euler::kUInt64);
  for (size_t i = 0; i < feature_names_.size(); ++i) {
    auto t_fid = query->AllocInput("__" + feature_names_[i], {1},
                                   euler::kString);
    *(t_fid->Raw<std::string*>()[0]) = "dense_" + feature_names_[i];
  }

  std::copy(nodes_flat.data(), nodes_flat.data() + nodes_flat.size(),
            t_nodes->Raw<int64_t>());

  auto callback = [outputs, done, query, nodes_size, this] () {
    std::stringstream ss;
    auto results_map = query->GetResult(res_names_);
    for (size_t i = 0; i < feature_names_.size(); ++i) {
      ss.str("");
      ss << "fea:" << i * 2;
      std::string fea_idx = ss.str();
      ss.str("");
      ss << "fea:" << i * 2 + 1;
      std::string fea_val = ss.str();
      auto idx_ptr = results_map[fea_idx];
      auto res = results_map[fea_val];
      auto idx_data = idx_ptr->Raw<int32_t>();
      auto res_data = res->Raw<float>();
      if (idx_ptr->NumElements() != nodes_size * 2) {
        EULER_LOG(FATAL) << "Feature Result Num Error:" <<
            idx_ptr->NumElements()
            << "Expect: " << nodes_size * 2;
      }
      for (size_t j = 0; j < nodes_size; ++j) {
        size_t start = idx_data[j * 2];
        size_t end = idx_data[j * 2 + 1];
        if (start < end) {
          std::copy(res_data + start, res_data + end,
                    outputs[i]->flat<float>().data() + j * dimensions_[i]);
        }
      }
    }
    delete query;
    done();
  };
  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("GetDenseFeature").Device(DEVICE_CPU), GetDenseFeature);

}  // namespace tensorflow
