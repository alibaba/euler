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
    .Attr("condition: string = ''")
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
condition: condition string for filter
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
    .Attr("condition: string = ''")
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
condition: condition string for filter
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
    .Attr("condition: string = ''")
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
condition: condition string for filter
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
    .Attr("condition: string = ''")
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
condition: condition string for filter

)doc");

REGISTER_OP("SampleNeighborLayerwiseWithAdj")
    .Input("last_layer_nodes: int64")
    .Input("edge_types: int32")
    .SetIsStateful()
    .Output("neighbors: int64")
    .Output("adj_indices: int64")
    .Output("adj_values: int64")
    .Output("adj_shape: int64")
    .Attr("count: int")
    .Attr("weight_func: string")
    .Attr("default_node: int = -1")
    .SetShapeFn(
        [] (InferenceContext* c) {
          ShapeHandle last_layer_nodes;
          ShapeHandle edge_types;
          int count = 0;
          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &last_layer_nodes));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &edge_types));
          TF_RETURN_IF_ERROR(c->GetAttr("count", &count));

          std::vector<DimensionHandle> dims;
          dims.emplace_back(c->Dim(last_layer_nodes, 0));
          dims.emplace_back(c->MakeDim(count));
          c->set_output(0, c->MakeShape(dims));

          return Status::OK();})
    .Doc(R"doc(
SampleNeighborLayerwiseWithAdj

Sample Neighbors in a layer-wise manner.


last_layer_nodes: Input, the nodes from last layer from which sampling distribution
is constructed. The shape is [batch_size, last_layer_count]

edge_types: Input, the outing edge types to sample neighbors for

neighbors: Output, the sample result, the shape is [batch_size, count]

adj_indices : Output, Sparse Tensor, the adj result between these two layers.
adj_values : Output, Sparse Tensor, the adj result between these two layers.
adj_shape : Output, Sparse Tensor, the adj result between these two layers.

count: sample neighbor count for the next layer
weight_func: weight postprocess function, like sqrt etc
default_node: default filling node if node has no neighbor

)doc");


REGISTER_OP("SampleFanout")
    .Input("nodes: int64")
    .Input("edge_types: int32")
    .SetIsStateful()
    .Output("neighbors: N * int64")
    .Output("weights: N * float")
    .Output("types: N * int32")
    .Attr("count: list(int)")
    .Attr("default_node: int = -1")
    .Attr("N: int")
    .SetShapeFn(
        [] (InferenceContext* c) {
          ShapeHandle nodes;
          ShapeHandle edge_types;
          std::vector<int> count;
          int N;
          DimensionHandle edge_types_dim_1;
          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &nodes));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &edge_types));
          TF_RETURN_IF_ERROR(c->GetAttr("count", &count));
          TF_RETURN_IF_ERROR(c->GetAttr("N", &N));
          TF_RETURN_IF_ERROR(c->WithValue(
                  c->Dim(edge_types, 0), count.size(), &edge_types_dim_1));

          if (count.size()!= static_cast<size_t>(N)) {
            return Status(
                error::INVALID_ARGUMENT,
                "Invalid count or edge_types size");
          }
          size_t layer_cnt = count.size();
          for (size_t i = 0; i < layer_cnt; ++i) {
            std::vector<DimensionHandle> dims;
            dims.emplace_back(c->Dim(nodes, 0));
            for (size_t j = 0; j <= i; ++j) {
              dims.emplace_back(c->MakeDim(count[j]));
            }
            c->set_output(i, c->MakeShape(dims));
            c->set_output(layer_cnt + i, c->MakeShape(dims));
            c->set_output(2 * layer_cnt + i, c->MakeShape(dims));
          }
          return Status::OK();})
    .Doc(R"doc(
SampleFanout
Sample Fanout Neighbors for nodes.
nodes: Input, the nodes to sample neighbors for
edge_types: Input, the outing edge types for each layer
neighbors: Output, the sample result nodes, N tensors. the shape should be [#nodes, count[0]] [#nodes, count[1]]...
weights: Output, the sample result weights, the shapes are the same as neighbors.
types: Output, the sample result types, the shapes are the same as neighbors.

count: a list, sample neighbor count for each node in each layer.
default_node: default filling node if node has no neighbor
)doc");

REGISTER_OP("SampleFanoutWithFeature")
    .Input("nodes: int64")
    .Input("edge_types: int32")
    .SetIsStateful()
    .Output("neighbors: N * int64")
    .Output("weights: N * float")
    .Output("types: N * int32")
    .Output("dense_features: ND * float")
    .Output("indices: NS * int64")
    .Output("values: NS * int64")
    .Output("dense_shape:  NS * int64")
    .Attr("count: list(int)")
    .Attr("default_node: int = -1")
    .Attr("sparse_feature_names: list(string)")
    .Attr("sparse_default_values: list(int)")
    .Attr("dense_feature_names: list(string)")
    .Attr("dense_dimensions: list(int)")
    .Attr("N: int >= 0")
    .Attr("ND: int >= 0")
    .Attr("NS: int >= 0")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
SampleFanoutWithFeature
Sample Fanout Neighbors for nodes and their features.
nodes: Input, the nodes to sample neighbors for
edge_types: Input, the outing edge types for each layer
neighbors: Output, the sample result nodes, N tensors. the shape should be [#nodes, count[0]] [#nodes, count[1]]...
weights: Output, the sample result weights, the shapes are the same as neighbors.
types: Output, the sample result types, the shapes are the same as neighbors.
indices: Output, sparse tensorf index for sparse feature.
values: Output, sparse tensor values, NS * int64
dense_shape: Output, sparse tensor dense shape, NS * int64
dense_features: Output, ND float tensors for the result dense features.

count: a list, sample neighbor count for each node in each layer.
default_node: default filling node if node has no neighbor.
N : size of count.
ND : the number of dense feature tensors. should be (layer_count + 1) * dense_feature_num
NS : the number of sparse feature tensors. should be (layer_count + 1) * sparse_feature_num
)doc");

}  // namespace tensorflow
