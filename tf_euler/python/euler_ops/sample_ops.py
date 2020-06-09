# Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.
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
from tf_euler.python.euler_ops import type_ops

sample_graph_label = base._LIB_OP.sample_graph_label

def _iter_body(i, state):
    y, count, n, ta = state
    curr_result = sample_node(count[i] * n, y[i])
    curr_result = tf.reshape(curr_result, [-1, n])
    out_ta = ta.write(i, curr_result)
    return i+1, (y, count, n, out_ta)


def sample_node(count, node_type, condition=''):
    """
    Sample Nodes by specific types

    Args:
    count: A scalar tensor specify sample count
    types: A scalar tensor specify sample type
    condition: A atrribute specify sample condition

    Return:
    A 1-d tensor of sample node ids
    """
    if node_type == '-1':
        types = -1
    else:
        types = type_ops.get_node_type_id(node_type)
    return base._LIB_OP.sample_node(count, types, condition)


def sample_edge(count, edge_type=None):
    """
    Sample Edges by specific types

    Args:
    count: A scalar tensor specify sample count
    types: A scalar tensor specify sample type

    Return:
    A 2-d tensor of sample edge ids
    """
    if edge_type == '-1':
        types = -1
    else:
        types = type_ops.get_edge_type_id(edge_type)
    return base._LIB_OP.sample_edge(count, types)


def sample_node_with_src(src_nodes, count):
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
    return base._LIB_OP.sample_n_with_types(count, types)

def get_graph_by_label(labels):
    res = base._LIB_OP.get_graph_by_label(labels)
    return tf.SparseTensor(*res[:3])
