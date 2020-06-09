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
#include <math.h>

#include <memory>
#include <vector>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
template <typename T>
class SparseGather : public OpKernel {
 public:
  explicit SparseGather(OpKernelConstruction* ctx): OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override;
  void GatherWithIndex(OpKernelContext* ctx, const Tensor& gather_idx,
                            const Tensor& sp_indices,
                            const Tensor& sp_values, int64 elem_cnt);
  void GatherWithBinarySearch(OpKernelContext* ctx, const Tensor& gather_idx,
                            const Tensor& sp_indices,
                            const Tensor& sp_values, int64 elem_cnt,
                            int64 gather_cnt);
};

template <typename T>
void SparseGather<T>::Compute(OpKernelContext* ctx) {
  auto gather_idx = ctx->input(0);
  OP_REQUIRES(ctx, TensorShapeUtils::IsVector(gather_idx.shape()),
    errors::InvalidArgument("SparseGather: GatherIdx expects a 1-D vector."));
  auto gather_cnt = gather_idx.shape().dim_size(0);

  auto sp_indices = ctx->input(1);
  OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(sp_indices.shape()),
    errors::InvalidArgument("SparseGather: sp_indices expects a 2-D Matrix."));
  auto in_len = sp_indices.shape().dim_size(0);
  auto indices_col_cnt = sp_indices.shape().dim_size(1);

  auto sp_values = ctx->input(2);
  OP_REQUIRES(ctx, TensorShapeUtils::IsVector(sp_values.shape()),
              errors::InvalidArgument(
                  "SparseGather: sp_values expects a 1-D vector."));
  OP_REQUIRES(
      ctx, sp_values.shape().dim_size(0) == in_len,
      errors::InvalidArgument(
          "SparseGather: len of sp_indices and sp_values should match"));

  auto sp_dense_shape = ctx->input(3);
  OP_REQUIRES(ctx, TensorShapeUtils::IsVector(sp_dense_shape.shape()),
    errors::InvalidArgument(
      "SparseGather: sp_dense_shape expects a 1-D vector."));
  OP_REQUIRES(
      ctx,
      sp_dense_shape.shape().dim_size(0) == sp_indices.shape().dim_size(1),
      errors::InvalidArgument(
          "SparseGather: sp_dense_shape and sp_indices shape mismatch."));

  auto sp_dense_shape_flat = sp_dense_shape.vec<int64>();
  double binary_search_cost = log(in_len) / log(2) * gather_cnt;

  if (binary_search_cost < in_len) {
    GatherWithBinarySearch(ctx, gather_idx, sp_indices, sp_values,
                           sp_dense_shape_flat(0), gather_cnt);
  } else {
    GatherWithIndex(ctx, gather_idx, sp_indices, sp_values,
                    sp_dense_shape_flat(0));
  }

  Tensor* output_shape = nullptr;
  OP_REQUIRES_OK(ctx,
    ctx->allocate_output(2, TensorShape({sp_dense_shape.dim_size(0)}),
    &output_shape));
  auto output_shape_flat = output_shape->vec<int64>();
  output_shape_flat(0) = gather_cnt;
  for (int i = 1; i < sp_dense_shape.dim_size(0); ++i) {
    output_shape_flat(i) = sp_dense_shape_flat(i);
  }
}

static int64 lower_bound(const int64 t, const Tensor& tensor,
                         const int64 col_cnt, int64 l, int64 h) {
  auto tensor_flat = tensor.flat<int64>();
  while (l <= h) {
    auto m = (l + h) >> 1;
    if (tensor_flat(m * col_cnt) < t) {
      l = m + 1;
    } else if (tensor_flat(m * col_cnt) > t) {
      h = m - 1;
    } else {
      int64 n_idx = lower_bound(t, tensor, col_cnt, l, m - 1);
      if (n_idx >= 0) {
        return n_idx;
      } else {
        return m;
      }
    }
  }
  return -1;
}

template <typename T>
void SparseGather<T>::GatherWithBinarySearch(OpKernelContext* ctx,
                            const Tensor& gather_idx,
                            const Tensor& sp_indices,
                            const Tensor& sp_values, int64 elem_cnt,
                            int64 gather_cnt) {
  // binary search solution, more efficient when gather_cnt is small.
  auto in_len = sp_indices.shape().dim_size(0);
  auto indices_col_cnt = sp_indices.shape().dim_size(1);
  auto sp_indices_flat = sp_indices.flat<int64>();
  auto sp_values_flat = sp_values.flat<T>();
  auto gather_idx_flat = gather_idx.flat<int64>();

  std::vector<int64> low_offset(gather_cnt, 0);
  std::vector<int64> high_offset(gather_cnt, 0);

  for (int i=0; i < gather_cnt; ++i) {
    auto curr_id = gather_idx_flat(i);
    OP_REQUIRES(ctx, curr_id < elem_cnt,
      errors::InvalidArgument("SparseGather: gather idx out of range."));
    low_offset[i] = lower_bound(curr_id, sp_indices,
                                indices_col_cnt, 0, in_len-1);
    if (curr_id == elem_cnt - 1) {
      high_offset[i] = in_len - 1;
    } else {
      high_offset[i] = lower_bound(curr_id+1, sp_indices,
                                   indices_col_cnt, 0, in_len-1) - 1;
    }
  }

  int64 out_len = 0;
  for (int i=0; i < gather_idx.dim_size(0); ++i) {
    out_len += high_offset[i] - low_offset[i] + 1;
  }
  Tensor* output_indices = nullptr;
  Tensor* output_values = nullptr;

  OP_REQUIRES_OK(
      ctx, ctx->allocate_output(0, TensorShape({out_len, indices_col_cnt}),
                                &output_indices));

  OP_REQUIRES_OK(ctx,
                 ctx->allocate_output(1, TensorShape({out_len}),
                                      &output_values));

  auto output_indices_flat = output_indices->flat<int64>();
  auto output_values_flat = output_values->flat<T>();

  int64 target_offset = 0;
  for (int i=0; i < gather_cnt; ++i) {
    auto offset = low_offset[i];
    auto next_offset = high_offset[i] + 1;
    for (auto j = offset; j < next_offset; ++j) {
      output_indices_flat(target_offset * indices_col_cnt) = i;
      output_values_flat(target_offset) = sp_values_flat(j);

      for (auto k = 1; k < indices_col_cnt; k++) {
        output_indices_flat(target_offset * indices_col_cnt + k)
            = sp_indices_flat(j * indices_col_cnt + k);
      }
      ++target_offset;
    }
  }
}

template <typename T>
void SparseGather<T>::GatherWithIndex(OpKernelContext* ctx,
                                      const Tensor& gather_idx,
                                      const Tensor& sp_indices,
                                      const Tensor& sp_values, int64 elem_cnt) {
  std::vector<int64> elem_offset(elem_cnt+1, 0);

  auto in_len = sp_indices.shape().dim_size(0);
  elem_offset[elem_cnt] = in_len;

  auto indices_col_cnt = sp_indices.shape().dim_size(1);
  auto sp_indices_flat = sp_indices.flat<int64>();
  auto sp_values_flat = sp_values.flat<T>();

  int64 curr = 0;
  for (int i = 0; i < in_len; i += 1) {
    if (sp_indices_flat(indices_col_cnt * i) != curr) {
      curr = sp_indices_flat(indices_col_cnt * i);
      elem_offset[curr] = i;
      OP_REQUIRES(
          ctx, curr < elem_cnt,
          errors::InvalidArgument("SparseGather: input indices out of range."));
    }
  }

  auto gather_idx_flat = gather_idx.flat<int64>();
  int64 out_len = 0;
  for (int i = 0; i < gather_idx.dim_size(0); ++i) {
    auto curr_id = gather_idx_flat(i);
    OP_REQUIRES(ctx, curr_id < elem_cnt,
      errors::InvalidArgument("SparseGather: gather idx out of range."));
    out_len += elem_offset[curr_id + 1] - elem_offset[curr_id];
  }
  Tensor* output_indices = nullptr;
  Tensor* output_values = nullptr;

  OP_REQUIRES_OK(ctx,
    ctx->allocate_output(0, TensorShape({out_len, indices_col_cnt}),
    &output_indices));

  OP_REQUIRES_OK(ctx,
    ctx->allocate_output(1, TensorShape({out_len}),
    &output_values));

  auto output_indices_flat = output_indices->flat<int64>();
  auto output_values_flat = output_values->flat<T>();

  int64 target_offset = 0;
  for (int i = 0; i < gather_idx.dim_size(0); ++i) {
    auto offset = elem_offset[gather_idx_flat(i)];
    auto next_offset = elem_offset[gather_idx_flat(i) + 1];
    for (auto j = offset; j < next_offset; ++j) {
      output_indices_flat(target_offset * indices_col_cnt) = i;
      output_values_flat(target_offset) = sp_values_flat(j);

      for (auto k = 1; k < indices_col_cnt; k++) {
        output_indices_flat(target_offset * indices_col_cnt + k)
          = sp_indices_flat(j * indices_col_cnt + k);
      }
      ++target_offset;
    }
  }
}

#define REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
    Name("SparseGather").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
    SparseGather<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
}  // namespace tensorflow
