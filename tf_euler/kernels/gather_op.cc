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

namespace tensorflow {

template <typename T, typename Index>
class GatherOp : public OpKernel {
 public:
  explicit GatherOp(OpKernelConstruction* c) : OpKernel(c) {
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& params = c->input(0);
    const Tensor& indices = c->input(1);

    int num_slides = indices.shape().dim_size(0);
    int slide_size = params.shape().dim_size(1);

    Tensor* out;
    TensorShape out_shape{num_slides, slide_size};
    OP_REQUIRES_OK(c, c->allocate_output(0, out_shape, &out));

    const T* params_base = &params.flat<T>()(0);
    const Index* indices_base = &indices.flat<Index>()(0);
    T* out_base = &out->flat<T>()(0);

    #pragma omp parallel for
    for (int i = 0; i < num_slides; ++i) {
      std::memcpy(&out_base[i * slide_size],
          &params_base[indices_base[i] * slide_size],
          slide_size * sizeof(T));
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MPGather")
  .Device(DEVICE_CPU)
  .TypeConstraint<float>("T")
  .TypeConstraint<int32>("Tindices"),
  GatherOp<float, int32>);

}  // namespace tensorflow
