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

REGISTER_OP("GetNodeType")
    .Input("nodes: int64")
    .Output("types: int32")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
GetNodeType.

Get types of nodes.

nodes: Input, nodes to get types for)doc");


REGISTER_OP("GetNodeTypeId")
    .Input("type_names: string")
    .Output("type_ids: int32")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
GetNodeTypeId.

Get type ids by node type names.

type_names: Input, node type names to get type ids for)doc");

REGISTER_OP("GetEdgeTypeId")
    .Input("type_names: string")
    .Output("type_ids: int32")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
GetEdgeTypeId.

Get type ids by edge type names.

type_names: Input, node type names to get type ids for)doc");

}  // namespace tensorflow
