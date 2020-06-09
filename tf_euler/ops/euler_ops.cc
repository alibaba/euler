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

REGISTER_OP("SparseGetAdj")
    .Input("batch_nodes: int64")
    .Input("batch_nb_nodes: int64")
    .Input("edge_types: int32")
    .SetIsStateful()
    .Output("adj_indices: int64")
    .Output("adj_values: int64")
    .Output("adj_shape: int64")
    .Attr("N: int")
    .Attr("M: int")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
SparseGetAdj
euler op: API_SPARSE_GEN_ADJ and API_SPARSE_GET_ADJ
)doc");

REGISTER_OP("SampleGraphLabel")
    .Input("batch_num: int32")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Output("features: string")
    .Doc(R"doc(
SampleGraphLabel
euler op: API_SAMPLE_GRAPH_LABEL)doc");

REGISTER_OP("GetGraphByLabel")
    .Input("graph_labels: string")
    .SetIsStateful()
    .SetShapeFn(shape_inference::UnknownShape)
    .Output("nodes_indices: int64")
    .Output("nodes_values: int64")
    .Output("nodes_shape: int64")
    .Doc(R"doc(
GetGraphByLabel
euler op: API_GET_GRAPH_BY_LABEL)doc");

}  // namespace tensorflow
