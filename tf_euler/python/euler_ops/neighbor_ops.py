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

import tensorflow as tf

from tf_euler.python.euler_ops import base

sample_neighbor = base._LIB_OP.sample_neighbor
get_top_k_neighbor = base._LIB_OP.get_top_k_neighbor


def get_full_neighbor(nodes, edge_types):
  """
  Args:
    nodes: A `Tensor` of `int64`.
    edge_types: A 1-D `Tensor` of int32. Specify edge types to filter outgoing
      edges.

  Return:
    A tuple of `SparseTensor` (neibors, weights).
      neighbors: A `SparseTensor` of `int64`.
      weights: A `SparseTensor` of `float`.
      types: A `SparseTensor` of `int32`
  """
  sp_returns = base._LIB_OP.get_full_neighbor(nodes, edge_types)
  return tf.SparseTensor(*sp_returns[:3]), tf.SparseTensor(*sp_returns[3:6]), \
         tf.SparseTensor(*sp_returns[6:])


def get_sorted_full_neighbor(nodes, edge_types):
  """
  Args:
    nodes: A `Tensor` of `int64`.
    edge_types: A 1-D `Tensor` of int32. Specify edge types to filter outgoing
      edges.

  Return:
    A tuple of `SparseTensor` (neibors, weights).
      neighbors: A `SparseTensor` of `int64`.
      weights: A `SparseTensor` of `float`.
      types: A `SparseTensor` of `int32`
  """
  sp_returns = base._LIB_OP.get_sorted_full_neighbor(nodes, edge_types)
  return tf.SparseTensor(*sp_returns[:3]), tf.SparseTensor(*sp_returns[3:6]), \
         tf.SparseTensor(*sp_returns[6:])


def sample_fanout(nodes, edge_types, counts, default_node=-1):
  """
  Sample multi-hop neighbors of nodes according to weight in graph.

  Args:
    nodes: A 1-D `Tensor` of `int64`.
    edge_types: A list of 1-D `Tensor` of int32. Specify edge types to filter
      outgoing edges in each hop.
    counts: A list of `int`. Specify the number of sampling for each node in
      each hop.
    default_node: A `int`. Specify the node id to fill when there is no neighbor
      for specific nodes.

  Return:
    A tuple of list: (samples, weights)
      samples: A list of `Tensor` of `int64`, with the same length as
        `edge_types` and `counts`, with shapes `[num_nodes]`,
        `[num_nodes * count1]`, `[num_nodes * count1 * count2]`, ...
      weights: A list of `Tensor` of `float`, with shapes
        `[num_nodes * count1]`, `[num_nodes * count1 * count2]` ...
      types: A list of `Tensor` of `int32`, with shapes
        `[num_nodes * count1]`, `[num_nodes * count1 * count2]` ...
  """
  neighbors_list = [tf.reshape(nodes, [-1])]
  weights_list = []
  type_list = []
  for hop_edge_types, count in zip(edge_types, counts):
    neighbors, weights, types = sample_neighbor(
        neighbors_list[-1], hop_edge_types, count, default_node=default_node)
    neighbors_list.append(tf.reshape(neighbors, [-1]))
    weights_list.append(tf.reshape(weights, [-1]))
    type_list.append(tf.reshape(weights, [-1]))
  return neighbors_list, weights_list, type_list
