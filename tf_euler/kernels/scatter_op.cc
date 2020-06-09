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
namespace {

template <typename T, typename Index>
class ScatterAddOp : public OpKernel {
 public:
  explicit ScatterAddOp(OpKernelConstruction* c) : OpKernel(c) {
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& updates = c->input(0);
    const Tensor& indices = c->input(1);
    const Tensor& size = c->input(2);

    Index size_value = size.scalar<Index>()();
    int slide_size = updates.shape().dim_size(1);

    Tensor* out;
    TensorShape out_shape{size_value, slide_size};
    OP_REQUIRES_OK(c, c->allocate_output(0, out_shape, &out));

    const T* updates_base = &updates.flat<T>()(0);
    const Index* indices_base = &indices.flat<Index>()(0);
    T* out_base = &out->flat<T>()(0);
    std::fill_n(out_base, size_value * slide_size, 0);

    int num_slides = indices.shape().dim_size(0);
    for (int i = 0; i < num_slides; ++i) {
      for (int j = 0; j < slide_size; ++j) {
        out_base[(indices_base[i] * slide_size) + j] +=
        updates_base[i * slide_size + j];
      }
    }
  }
};

template <typename T, typename Index>
class ScatterMaxOp : public OpKernel {
 public:
  explicit ScatterMaxOp(OpKernelConstruction* c) : OpKernel(c) {
  }

  void Compute(OpKernelContext* c) override {
    const Tensor& updates = c->input(0);
    const Tensor& indices = c->input(1);
    const Tensor& size = c->input(2);

    Index size_value = size.scalar<Index>()();
    int slide_size = updates.shape().dim_size(1);

    Tensor* out;
    TensorShape out_shape{size_value, slide_size};
    OP_REQUIRES_OK(c, c->allocate_output(0, out_shape, &out));

    const T* updates_base = &updates.flat<T>()(0);
    const Index* indices_base = &indices.flat<Index>()(0);
    T* out_base = &out->flat<T>()(0);
    std::fill_n(out_base, size_value * slide_size, -1e9);

    int num_slides = indices.shape().dim_size(0);
    for (int i = 0; i < num_slides; ++i) {
      for (int j = 0; j < slide_size; ++j) {
        int out_offset = (indices_base[i] * slide_size) + j;
        int updates_offset = i * slide_size + j;
        if (updates_base[updates_offset] > out_base[out_offset]) {
          out_base[out_offset] = updates_base[updates_offset];
        }
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("MPScatterAdd")
  .Device(DEVICE_CPU)
  .TypeConstraint<float>("T")
  .TypeConstraint<int32>("Tindices"),
  ScatterAddOp<float, int32>);

REGISTER_KERNEL_BUILDER(Name("MPScatterMax")
  .Device(DEVICE_CPU)
  .TypeConstraint<float>("T")
  .TypeConstraint<int32>("Tindices"),
  ScatterMaxOp<float, int32>);

}  // namespace
}  // namespace tensorflow
