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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

REGISTER_OP("MPGather")
    .Input("params: T")
    .Input("indices: Tindices")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(
        [] (shape_inference::InferenceContext* c) {
          shape_inference::ShapeHandle params, indices;
          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &params));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices));

          c->set_output(0, c->Matrix(
          shape_inference::InferenceContext::kUnknownDim,
          c->Dim(params, 1)));

          return Status::OK();
    });

REGISTER_OP("MPScatterAdd")
    .Input("updates: T")
    .Input("indices: Tindices")
    .Input("size: Tindices")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(
        [] (shape_inference::InferenceContext* c) {
          shape_inference::ShapeHandle updates, indices, size;
          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &updates));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &size));

          c->set_output(0, c->Matrix(
                  shape_inference::InferenceContext::kUnknownDim,
                  c->Dim(updates, 1)));

          return Status::OK();
    });

REGISTER_OP("MPScatterMax")
    .Input("updates: T")
    .Input("indices: Tindices")
    .Input("size: Tindices")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tindices: {int32, int64}")
    .SetShapeFn(
        [] (shape_inference::InferenceContext* c) {
          shape_inference::ShapeHandle updates, indices, size;
          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &updates));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &size));

          c->set_output(0, c->Matrix(
                  shape_inference::InferenceContext::kUnknownDim,
                  c->Dim(updates, 1)));

          return Status::OK();
    });

}  // namespace tensorflow
