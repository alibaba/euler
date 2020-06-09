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

#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;


///////////////////////////// Get Node Feature OP /////////////////////////////


REGISTER_OP("GetBinaryFeature")
    .Input("nodes: int64")
    .Output("features: N *string")
    .Attr("feature_names: list(string)")
    .Attr("N: int >= 0")
    .SetShapeFn(
        [] (InferenceContext* c) {
          int N;
          ShapeHandle nodes;
          std::vector<std::string> feature_names;

          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &nodes));
          TF_RETURN_IF_ERROR(c->GetAttr("feature_names", &feature_names));
          TF_RETURN_IF_ERROR(c->GetAttr("N", &N));

          if (static_cast<size_t>(N) != feature_names.size()) {
            return Status(
                error::INVALID_ARGUMENT,
                "Invalid dimension or feature names size");
          }

          std::vector<DimensionHandle> dims;
          dims.emplace_back(c->Dim(nodes, 0));

          for (int i = 0; i < N; ++i) {
            c->set_output(i, c->MakeShape(dims));
          }
          return Status::OK();})
    .Doc(R"doc(
GetBinaryFeature.

Get binary features for nodes.

nodes: Input, nodes to get binary features for
features: Output, string tensor for the result binary features
feature_names: feature names to retrieve

)doc");

REGISTER_OP("GetSparseFeature")
    .Input("nodes: int64")
    .Output("indices: N * int64")
    .Output("values: N * int64")
    .Output("dense_shape: N * int64")
    .Attr("feature_names: list(string)")
    .Attr("default_values: list(int)")
    .Attr("N: int >= 0")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
GetSparseFeature.

Get Sparse feature for nodes

nodes: Input, target nodes
indices: Output, sparse tensor index, N * int64
values: Output, sparse tensor values, N * int64
dense_shape: Output, sparse tensor dense shape, N * int64
feature_names: feature names
N: sizeof feature ids

)doc");

REGISTER_OP("GetDenseFeature")
    .Input("nodes: int64")
    .Output("features: N * float")
    .Attr("feature_names: list(string)")
    .Attr("dimensions: list(int)")
    .Attr("N: int")
    .SetShapeFn(
        [] (InferenceContext* c) {
          int N;
          ShapeHandle nodes;
          std::vector<int> dimensions;
          std::vector<std::string> feature_names;

          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &nodes));
          TF_RETURN_IF_ERROR(c->GetAttr("feature_names", &feature_names));
          TF_RETURN_IF_ERROR(c->GetAttr("dimensions", &dimensions));
          TF_RETURN_IF_ERROR(c->GetAttr("N", &N));

          if (dimensions.size() != feature_names.size() ||
              dimensions.size() != static_cast<size_t>(N)) {
            return Status(
                error::INVALID_ARGUMENT,
                "Invalid dimension or feature ids size");
          }

          for (int i = 0; i < N; ++i) {
            std::vector<DimensionHandle> dims;
            dims.emplace_back(c->Dim(nodes, 0));
            dims.emplace_back(c->MakeDim(dimensions[i]));
            c->set_output(i, c->MakeShape(dims));
          }
          return Status::OK();})
    .Doc(R"doc(
GetDenseFeature.

Get dense features for nodes.

nodes: Input, nodes to get dense features for
features: Output, N float tensors for the result dense features
feature_names: feature names to retrieve
dimensions: dimension for each feature
N: size of feature ids

)doc");

///////////////////////////// Get Edge Feature OP /////////////////////////////


REGISTER_OP("GetEdgeBinaryFeature")
    .Input("edges: int64")
    .Output("features: N * string")
    .Attr("feature_names: list(string)")
    .Attr("N: int")
    .SetShapeFn(
        [] (InferenceContext* c) {
          int N;
          ShapeHandle edges;
          std::vector<string> feature_names;

          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &edges));
          TF_RETURN_IF_ERROR(c->GetAttr("feature_names", &feature_names));
          TF_RETURN_IF_ERROR(c->GetAttr("N", &N));

          if (static_cast<size_t>(N) != feature_names.size()) {
            return Status(
                error::INVALID_ARGUMENT,
                "Invalid dimension or feature ids size");
          }

          std::vector<DimensionHandle> dims;
          dims.emplace_back(c->Dim(edges, 0));

          for (int i = 0; i < N; ++i) {
            c->set_output(i, c->MakeShape(dims));
          }
          return Status::OK();})
    .Doc(R"doc(
GetBinaryFeature.

Get binary features for edges.

edges: Input, edges to get binary features for
features: Output, string tensor for the result binary features
feature_names: feature names to retrieve

)doc");


REGISTER_OP("GetEdgeSparseFeature")
    .Input("edges: int64")
    .Output("indices: N * int64")
    .Output("values: N * int64")
    .Output("dense_shape: N * int64")
    .Attr("feature_names: list(string)")
    .Attr("default_values: list(int)")
    .Attr("N: int >= 0")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
GetEdgeSparseFeature.

Get Sparse feature for edges

edges: Input, target edges
indices: Output, sparse tensor index, N * int64
values: Output, sparse tensor values, N * int64
dense_shape: Output, sparse tensor dense shape, N * int64
feature_names: feature names
N: sizeof feature ids

)doc");


REGISTER_OP("GetEdgeDenseFeature")
    .Input("edges: int64")
    .Output("features: N * float")
    .Attr("feature_names: list(string)")
    .Attr("dimensions: list(int)")
    .Attr("N: int")
    .SetShapeFn(
        [] (InferenceContext* c) {
          int N;
          ShapeHandle edges;
          std::vector<int> dimensions;
          std::vector<string> feature_names;

          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &edges));
          TF_RETURN_IF_ERROR(c->GetAttr("feature_names", &feature_names));
          TF_RETURN_IF_ERROR(c->GetAttr("dimensions", &dimensions));
          TF_RETURN_IF_ERROR(c->GetAttr("N", &N));

          if (dimensions.size() != feature_names.size() ||
              dimensions.size() != static_cast<size_t>(N)) {
            return Status(
                error::INVALID_ARGUMENT,
                "Invalid dimension or feature ids size");
          }

          for (int i = 0; i < N; ++i) {
            std::vector<DimensionHandle> dims;
            dims.emplace_back(c->Dim(edges, 0));
            dims.emplace_back(c->MakeDim(dimensions[i]));
            c->set_output(i, c->MakeShape(dims));
          }
          return Status::OK();
        })
    .Doc(R"doc(
GetEdgeDenseFeature.

Get dense features for edges.

edges: Input, edges to get dense features for
features: Output, N float tensors for the result dense features
feature_names: feature names to retrieve
dimensions: dimension for each feature
N: size of feature ids

)doc");

}  // namespace tensorflow
