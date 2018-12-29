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


def _get_sparse_feature(nodes_or_edges, feature_ids, op, default_values=None):
  if default_values is None:
    default_values = [0] * len(feature_ids)

  sp_returns = op(nodes_or_edges, feature_ids, default_values,
                  len(feature_ids))
  return [tf.SparseTensor(*sp_return) for sp_return in zip(*sp_returns)]


def get_sparse_feature(nodes, feature_ids, default_values=None):
  """
  Fetch sparse features of nodes.

  Args:
    nodes: A 1-d `Tensor` of `int64`.
    feature_ids: A list of `int`. Specify uint64 feature ids in graph to fetch
      features for nodes.
    default_values: A `int`. Specify value to fill when there is no specific
      features for specific nodes.

  Return:
    A list of `SparseTensor` with the same length as `feature_ids`.
  """
  return _get_sparse_feature(nodes, feature_ids,
                             base._LIB_OP.get_sparse_feature)


def get_edge_sparse_feature(edges, feature_ids, default_values=None):
  """
  Args:
    edges: A 2-D `Tensor` of `int64`, with shape `[num_edges, 3]`.
    feature_ids: A list of `int`. Specify uint64 feature ids in graph to fetch
      features for edges.
    default_values: A `int`. Specify value to fill when there is no specific
      features for specific edges.

  Return:
    A list of `SparseTensor` with the same length as `feature_ids`.
  """
  return _get_sparse_feature(edges, feature_ids,
                             base._LIB_OP.get_edge_sparse_feature)


def _get_dense_feature(nodes_or_edges, feature_ids, dimensions, op):
  return op(
      nodes_or_edges,
      feature_ids=feature_ids,
      dimensions=dimensions,
      N=len(feature_ids))


def get_dense_feature(nodes, feature_ids, dimensions):
  """
  Fetch dense features of nodes.

  Args:
    nodes: A 1-d `Tensor` of `int64`.
    feature_ids: A list of `int`. Specify float feature ids in graph to fetch
      features for nodes.
    dimensions: A list of `int`. Specify dimensions of each feature.

  Return:
    A list of `Tensor` with the same length as `feature_ids`.
  """
  return _get_dense_feature(nodes, feature_ids, dimensions,
                            base._LIB_OP.get_dense_feature)


def get_edge_dense_feature(edges, feature_ids, dimensions):
  """
  Fetch dense features of edges.

  Args:
    nodes: A 2-d `Tensor` of `int64`, with shape `[num_edges, 3]`.
    feature_ids: A list of `int`. Specify float feature ids in graph to fetch
      features for edges.
    dimensions: A list of `int`. Specify dimensions of each feature.

  Return:
    A list of `Tensor` with the same length as `feature_ids`.
  """
  return _get_dense_feature(edges, feature_ids, dimensions,
                            base._LIB_OP.get_edge_dense_feature)


def _get_binary_feature(nodes, feature_ids, op):
  return op(nodes, feature_ids, N=len(feature_ids))


def get_binary_feature(nodes, feature_ids):
  """
  Fetch binary features of nodes.

  Args:
    nodes: A 1-d `Tensor` of `int64`.
    feature_ids: A list of `int`. Specify uint64 feature ids in graph to fetch
      features for nodes.

  Return:
    A list of `String Tensor` with the same length as `feature_ids`.
  """
  return _get_binary_feature(nodes, feature_ids,
                             base._LIB_OP.get_binary_feature)


def get_edge_binary_feature(edges, feature_ids):
  """
  Fetch binary features of edges.

  Args:
    edges: A 2-d `Tensor` of `int64`, with shape `[num_edges, 3]`.
    feature_ids: A list of `int`. Specify uint64 feature ids in graph to fetch
      features for nodes.

  Return:
    A list of `String Tensor` with the same length as `feature_ids`.
  """
  return _get_binary_feature(edges, feature_ids,
                             base._LIB_OP.get_edge_binary_feature)
