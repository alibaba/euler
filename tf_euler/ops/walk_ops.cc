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

namespace tensorflow {

using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;
using shape_inference::DimensionHandle;

REGISTER_OP("GenPair")
    .Input("paths: int64")
    .Output("pairs: int64")
    .Attr("left_win_size: int")
    .Attr("right_win_size: int")
    .SetShapeFn(
        [] (InferenceContext* c) {
          ShapeHandle paths;
          int left_win_size;
          int right_win_size;
          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &paths));
          TF_RETURN_IF_ERROR(c->GetAttr("left_win_size", &left_win_size));
          TF_RETURN_IF_ERROR(c->GetAttr("right_win_size", &right_win_size));

          auto batch_size = c->Value(c->Dim(paths, 0));
          auto path_len = c->Value(c->Dim(paths, 1));
          int64 pair_count = -1;
          if (path_len > 0) {
            pair_count = path_len * (left_win_size + right_win_size);
            for (int i = left_win_size, j = 0;
                 i > 0 && j < path_len; --i, ++j) {
              pair_count -= i;
            }

            for (int i = right_win_size, j = 0;
                 i > 0 && j < path_len; --i, ++j) {
              pair_count -= i;
            }
          }

          std::vector<DimensionHandle> dims;
          dims.emplace_back(c->MakeDim(batch_size));
          dims.emplace_back(c->MakeDim(pair_count));
          dims.emplace_back(c->MakeDim(2));
          c->set_output(0, c->MakeShape(dims));

          return Status::OK();})
    .Doc(R"doc(
GenPair.

Generate positive pair sample for train and test.

paths: Input, node paths for generating node pairs
pairs: Ouput, generate node pair result
left_win_size: left window size
right_win_size: right window size

)doc");


REGISTER_OP("RandomWalk")
    .Input("nodes: int64")
    .Input("edge_types: walk_len * int32")
    .SetIsStateful()
    .Output("samples: int64")
    .Attr("walk_len: int")
    .Attr("p: float = 1.0")
    .Attr("q: float = 1.0")
    .Attr("default_node: int = -1")
    .SetShapeFn(
        [] (InferenceContext* c) {
          ShapeHandle nodes;
          int walk_len = 0;
          TF_RETURN_IF_ERROR(c->GetAttr("walk_len", &walk_len));
          TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &nodes));

          std::vector<DimensionHandle> dims;
          dims.emplace_back(c->Dim(nodes, 0));
          dims.emplace_back(c->MakeDim(walk_len + 1));
          c->set_output(0, c->MakeShape(dims));

          return Status::OK();})
    .Doc(R"doc(
Random Walk.

Performs random walk from a serias of start nodes.
See https://arxiv.org/abs/1607.00653 for reference.

nodes: Input tensor, start nodes.
edge_types: Edge type for each walk step
samples: Output tensor, the sample result
walk_len: the walk length from every node
p: Return parameter, see https://arxiv.org/abs/1607.00653
q: In-out parameter, sess https://arxiv.org/abs/1607.00653
default_node: filling nodes if the target node has no neighbors

)doc");

}  // namespace tensorflow
