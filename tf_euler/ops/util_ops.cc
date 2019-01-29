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

  REGISTER_OP("InflateIdx")
  .Attr("T: {int32}")
  .Input("idx: T")
  .Output("out_idx: T")
  .SetShapeFn(shape_inference::UnchangedShape)
  .Doc(R"doc(
InflateIdx
tf.unique/tf.unique_with_counts will generate the index of each input
value in the unique output. 
Now we suppose the uniqued values are repeated with the original count info.
And we would like each idx refer to a unique place of the inflated values.
idx: the input idx vector
out_idx: the modified idx vector with the same shape
)doc");

} // namespace tensorflow
