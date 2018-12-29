/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "euler/client/graph.h"

namespace tensorflow {

class GenPair: public OpKernel {
 public:
  explicit GenPair(OpKernelConstruction* ctx): OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("left_win_size", &left_win_size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("right_win_size", &right_win_size_));
  }

  void Compute(OpKernelContext* ctx) override;

 private:
  int left_win_size_;
  int right_win_size_;
};

void GenPair::Compute(OpKernelContext* ctx) {
  auto paths = ctx->input(0);
  auto& paths_shape = paths.shape();
  auto batch_size = paths_shape.dim_size(0);
  auto path_len = paths_shape.dim_size(1);

  auto pair_count = path_len * (left_win_size_ + right_win_size_);
  for (int i = left_win_size_, j = 0; i > 0 && j < path_len; --i, ++j) {
    pair_count -= i;
  }
  for (int i = right_win_size_, j = 0; i > 0 && j < path_len; --i, ++j) {
    pair_count -= i;
  }

  std::vector<std::vector<int64>> pairs(batch_size);
  for (auto& pair : pairs) {
    pair.reserve(path_len * (left_win_size_ + right_win_size_));
  }

  auto paths_data = paths.flat<int64>().data();
  for (int i = 0; i < batch_size; ++i) {
    auto path = paths_data + i * path_len;
    for (int j = 0; j < path_len; ++j) {
      int k = 0;
      while ((j - k - 1) >= 0 && k < left_win_size_) {
        pairs[i].push_back(path[j]);
        pairs[i].push_back(path[j - k - 1]);
        ++k;
      }

      k = 0;
      while ((j + k + 1) < path_len && k < right_win_size_) {
        pairs[i].push_back(path[j]);
        pairs[i].push_back(path[j + k + 1]);
        ++k;
      }
    }
  }

  TensorShape output_shape;
  output_shape.AddDim(batch_size);
  output_shape.AddDim(pair_count);
  output_shape.AddDim(2);

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  auto output_data = output->flat<int64>().data();
  for (int i = 0; i < batch_size; ++i) {
    memcpy(output_data + i * pair_count * 2, pairs[i].data(),
           pairs[i].size() * sizeof(int64));
  }
}

REGISTER_KERNEL_BUILDER(Name("GenPair").Device(DEVICE_CPU), GenPair);

}  // namespace tensorflow
