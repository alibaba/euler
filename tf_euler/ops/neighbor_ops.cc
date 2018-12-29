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

#include <vector>
#include <iostream>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

REGISTER_OP("GetFullNeighbor")
    .Input("nodes: int64")
    .Input("edge_types: int32")
    .Output("id_indices: int64")
    .Output("id_values: int64")
    .Output("id_shape: int64")
    .Output("weight_indices: int64")
    .Output("weight_values: float")
    .Output("weight_shape: int64")
    .Output("type_indices: int64")
    .Output("type_values: int32")
    .Output("type_shape: int64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
GetFullNeighbor.

Get all neighbors of the target nodes.
Return a sparse id tensor and a sparse weight tensor

nodes: Input, target nodes to get neghbors for
edge_types: Input, the outing edge types to sample neighbors for
id_indices: Output, the result sparse id tensor indices
id_values: Output, the result sparse id tensor values
id_shape: Output, the result sparse id tensor shape
weight_indices: Output, the result sparse weight tensor indices
weight_values: Output, the result sparse weight tensor values
weight_shape: Output, the result sparse weight tensor shape
type_indices: Output, the result sparse type tensor indices
type_values: Output, the result sparse type tensor values
type_shape: Output, the result sparse type tensor shape

)doc");

REGISTER_OP("GetSortedFullNeighbor")
    .Input("nodes: int64")
    .Input("edge_types: int32")
    .Output("id_indices: int64")
    .Output("id_values: int64")
    .Output("id_shape: int64")
    .Output("weight_indices: int64")
    .Output("weight_values: float")
    .Output("weight_shape: int64")
    .Output("type_indices: int64")
    .Output("type_values: int32")
    .Output("type_shape: int64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
GetSortedFullNeighbor.

Get all neighbors of the target nodes, sorted by neighbor id.
Return a sparse id tensor and a sparse weight tensor

nodes: Input, target nodes to get neghbors for
edge_types: Input, the outing edge types to sample neighbors for
id_indices: Output, the result sparse id tensor indices
id_values: Output, the result sparse id tensor values
id_shape: Output, the result sparse id tensor shape
weight_indices: Output, the result sparse weight tensor indices
weight_values: Output, the result sparse weight tensor values
weight_shape: Output, the result sparse weight tensor shape
type_indices: Output, the result sparse type tensor indices
type_values: Output, the result sparse type tensor values
type_shape: Output, the result sparse type tensor shape

)doc");

REGISTER_OP("GetTopKNeighbor")
    .Input("nodes: int64")
    .Input("edge_types: int32")
    .Output("neighbors: int64")
    .Output("weights: float")
    .Output("types: int32")
    .Attr("k: int")
    .Attr("default_node: int = -1")
    .SetShapeFn(
        [] (InferenceContext* c) {
          ShapeHandle nodes;
          ShapeHandle edge_types;
          int k = 0;
          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &nodes));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &edge_types));
          TF_RETURN_IF_ERROR(c->GetAttr("k", &k));

          std::vector<DimensionHandle> dims;
          dims.emplace_back(c->Dim(nodes, 0));
          dims.emplace_back(c->MakeDim(k));
          c->set_output(0, c->MakeShape(dims));
          c->set_output(1, c->MakeShape(dims));

          return Status::OK();})
    .Doc(R"doc(
GetTopKNeighbor

Get Top K Neighbors for nodes.

nodes: Input, the nodes to get neighbors for
edge_types: Input, the outing edge types to get neighbors for
neighbors: Output, the result of get top k neighbors
k: Number of top neighbor for each node
default_node: default filling node if node has no or not enough neighbor

)doc");

REGISTER_OP("SampleNeighbor")
    .Input("nodes: int64")
    .Input("edge_types: int32")
    .SetIsStateful()
    .Output("neighbors: int64")
    .Output("weights: float")
    .Output("types: int32")
    .Attr("count: int")
    .Attr("default_node: int = -1")
    .SetShapeFn(
        [] (InferenceContext* c) {
          ShapeHandle nodes;
          ShapeHandle edge_types;
          int count = 0;
          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &nodes));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &edge_types));
          TF_RETURN_IF_ERROR(c->GetAttr("count", &count));

          std::vector<DimensionHandle> dims;
          dims.emplace_back(c->Dim(nodes, 0));
          dims.emplace_back(c->MakeDim(count));
          c->set_output(0, c->MakeShape(dims));
          c->set_output(1, c->MakeShape(dims));

          return Status::OK();})
    .Doc(R"doc(
SampleNeighbor

Sample Neighbors for nodes.

nodes: Input, the nodes to sample neighbors for
edge_types: Input, the outing edge types to sample neighbors for
neighbors: Output, the sample result
count: sample neighbor count for each node
default_node: default filling node if node has no neighbor

)doc");

}  // namespace tensorflow
