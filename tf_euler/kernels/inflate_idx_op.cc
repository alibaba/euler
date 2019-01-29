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
#include <unordered_map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
  
  class InflateIdx: public OpKernel {
  public:
    explicit InflateIdx(OpKernelConstruction* ctx): OpKernel(ctx) {
    }

    void Compute(OpKernelContext* ctx) override;

  private:
  };

  void InflateIdx::Compute(OpKernelContext* ctx) {
    auto in_idx = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(in_idx.shape()),
                errors::InvalidArgument("InflateIdx expects a 1-D vector."));
    auto len = in_idx.shape().dim_size(0);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in_idx.shape(), &output));
    
    auto in_data = in_idx.flat<int32>();
    std::unordered_set<int32> cnt_set;
    for (int i = 0; i < len; ++i) {
      cnt_set.insert(in_data(i));
    }
    auto unique_cnt = cnt_set.size();
    std::vector<int32> sub_cnt(unique_cnt, 0);
    std::vector<int32> partial_sum(unique_cnt, 0);
    
    for (int i = 0; i < len; ++i) {
      OP_REQUIRES(ctx, in_data(i) >= 0 && in_data(i) < unique_cnt,
                  errors::InvalidArgument("expect input idx in [0,unique_cnt)."));
      sub_cnt[in_data(i)]++;
    }
    
    for (int i=1; i < unique_cnt; ++i) {
      partial_sum[i] = partial_sum[i-1] + sub_cnt[i-1];
    }

    std::vector<int32> idx_offsets = partial_sum;

    auto output_data = output->flat<int32>();
    for (int i = 0; i < len; ++i) {
      output_data(i) = idx_offsets[in_data(i)]++;
    }
  }

  REGISTER_KERNEL_BUILDER(Name("InflateIdx").Device(DEVICE_CPU), InflateIdx);

}  // namespace tensorflow
