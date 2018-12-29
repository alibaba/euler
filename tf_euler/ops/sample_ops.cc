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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

namespace tensorflow {

REGISTER_OP("SampleNode")
    .Input("count: int32")
    .Input("node_type: int32")
    .SetIsStateful()
    .Output("nodes: int64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
SampleNode

Sample nodes by type, using as negative sample.
See https://arxiv.org/abs/1607.00653 for reference.

count: Input, sample nodes count
node_type: Input, sample node type
nodes: Output, sample result nodes

)doc");

REGISTER_OP("SampleEdge")
    .Input("count: int32")
    .Input("edge_type: int32")
    .SetIsStateful()
    .Output("edges: int64")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
SampleEdge

Sample edges by type, using as negative sample.
See https://arxiv.org/abs/1607.00653 for reference.

count: Input, sample edges count
edge_type: Input, sample edge type
edges: Output, sample result edges

)doc");

}  // namespace tensorflow
