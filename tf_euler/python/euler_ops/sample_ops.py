# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import os

import tensorflow as tf

from tf_euler.python.euler_ops import base

ALL_NODE_TYPE = -1

sample_node = base._LIB_OP.sample_node
sample_edge = base._LIB_OP.sample_edge

def _iter_body(i, state):
  y, count, n, ta = state
  curr_result = sample_node(count[i] * n, y[i])
  curr_result = tf.reshape(curr_result, [-1, n])
  out_ta = ta.write(i, curr_result)
  return i+1, (y, count, n, out_ta)

def sample_node_with_src(src_nodes, n, share_sample=False):
  """
  for each src node, sample "n" nodes with the same type

  Args:
    src_nodes: A 1-d `Tensor` of `int64`
    n: A scalar value of int
  Returns:
    A 2-dim Tensor, the first dim should be equal to dim of src_node.
    The second dim should be equal to n.
  """
  types = base._LIB_OP.get_node_type(src_nodes)
  y, idx, count = tf.unique_with_counts(types, out_idx=tf.int32)

  if share_sample:
    count = tf.ones_like(count, dtype=tf.int32)
    out_idx = idx
  else:
    out_idx = base._LIB_OP.inflate_idx(idx)

  rows = tf.shape(y)[0]

  ta = tf.TensorArray(dtype=tf.int64, size=rows, infer_shape=False)
  init_state = (0, (y, count, n, ta))
  condition = lambda i, _: i < rows
  _, (_, _, _, ta_final) = tf.while_loop(condition, _iter_body, init_state)
  tensor_final = ta_final.concat()

  return tf.gather(params=tensor_final, indices=out_idx, axis=0)
