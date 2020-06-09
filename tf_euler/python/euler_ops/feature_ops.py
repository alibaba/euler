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

import tensorflow as tf

from tf_euler.python.euler_ops import base


def _split_input_data(data_list, thread_num):
    size = tf.shape(data_list)[0]
    split_size = [size // thread_num] * (thread_num - 1)
    if thread_num == 1:
        split_size += [size]
    else:
        split_size += [-1]
    split_data_list = tf.split(data_list, split_size)
    return split_data_list


def _get_sparse_feature(nodes_or_edges, feature_names, op, thread_num,
                        default_values=None):
    feature_names = map(str, feature_names)
    if not hasattr(feature_names, '__len__'):
        feature_names = list(feature_names)

    if default_values is None:
        default_values = [0] * len(feature_names)

    split_data_list = _split_input_data(nodes_or_edges, thread_num)
    split_result_list = [op(split_data, feature_names, default_values,
                         len(feature_names)) for split_data in split_data_list]
    split_sp = []
    for i in range(len(split_result_list)):
        split_sp.append(
            [tf.SparseTensor(*sp) for sp in zip(*split_result_list[i])])
    split_sp_transpose = map(list, zip(*split_sp))
    return [tf.sparse_concat(axis=0, sp_inputs=sp, expand_nonconcat_dim=True)
            for sp in split_sp_transpose]


def get_sparse_feature(nodes, feature_names,
                       default_values=None, thread_num=1):
    """
    Fetch sparse features of nodes.

    Args:
      nodes: A 1-d `Tensor` of `int64`.
      feature_names: A list of `int`. Specify uint64 feature ids in graph to
        fetch features for nodes.
      default_values: A `int`. Specify value to fill when there is no specific
        features for specific nodes.

    Return:
      A list of `SparseTensor` with the same length as `feature_names`.
    """
    return _get_sparse_feature(nodes, feature_names,
                               base._LIB_OP.get_sparse_feature, thread_num)


def get_edge_sparse_feature(edges, feature_names,
                            default_values=None, thread_num=1):
    """
    Args:
      edges: A 2-D `Tensor` of `int64`, with shape `[num_edges, 3]`.
      feature_names: A list of `int`. Specify uint64 feature ids in graph to
        fetch features for edges.
      default_values: A `int`. Specify value to fill when there is no specific
        features for specific edges.

    Return:
      A list of `SparseTensor` with the same length as `feature_names`.
    """
    return _get_sparse_feature(edges, feature_names,
                               base._LIB_OP.get_edge_sparse_feature,
                               thread_num)


def _get_dense_feature(nodes_or_edges, feature_names, dimensions,
                       op, thread_num):
    feature_names = map(str, feature_names)
    if not hasattr(feature_names, '__len__'):
        feature_names = list(feature_names)

    split_data_list = _split_input_data(nodes_or_edges, thread_num)
    split_result_list = [op(split_data,
                            feature_names,
                            dimensions,
                            N=len(feature_names))
                         for split_data in split_data_list]
    split_result_list_transpose = map(list, zip(*split_result_list))
    return [tf.concat(split_dense, 0)
            for split_dense in split_result_list_transpose]


def get_dense_feature(nodes, feature_names, dimensions, thread_num=1):
    """
    Fetch dense features of nodes.

    Args:
      nodes: A 1-d `Tensor` of `int64`.
      feature_names: A list of `int`. Specify float feature ids in graph to
        fetch features for nodes.
      dimensions: A list of `int`. Specify dimensions of each feature.

    Return:
      A list of `Tensor` with the same length as `feature_names`.
    """
    return _get_dense_feature(nodes, feature_names, dimensions,
                              base._LIB_OP.get_dense_feature, thread_num)


def get_edge_dense_feature(edges, feature_names, dimensions, thread_num=1):
    """
    Fetch dense features of edges.

    Args:
      nodes: A 2-d `Tensor` of `int64`, with shape `[num_edges, 3]`.
      feature_names: A list of `int`. Specify float feature ids in graph to
        fetch features for edges.
      dimensions: A list of `int`. Specify dimensions of each feature.

    Return:
      A list of `Tensor` with the same length as `feature_names`.
    """
    return _get_dense_feature(edges, feature_names, dimensions,
                              base._LIB_OP.get_edge_dense_feature, thread_num)


def _get_binary_feature(nodes_or_edges, feature_names, op, thread_num):
    feature_names = map(str, feature_names)
    if not hasattr(feature_names, '__len__'):
        feature_names = list(feature_names)

    split_data_list = _split_input_data(nodes_or_edges, thread_num)
    split_result_list = [op(split_data, feature_names, N=len(feature_names))
                         for split_data in split_data_list]
    split_result_list_transpose = map(list, zip(*split_result_list))
    return [tf.concat(split_binary, 0)
            for split_binary in split_result_list_transpose]


def get_binary_feature(nodes, feature_names, thread_num=1):
    """
    Fetch binary features of nodes.

    Args:
      nodes: A 1-d `Tensor` of `int64`.
      feature_names: A list of `int`. Specify uint64 feature ids in graph to
        fetch features for nodes.

    Return:
      A list of `String Tensor` with the same length as `feature_names`.
    """
    return _get_binary_feature(nodes, feature_names,
                               base._LIB_OP.get_binary_feature, thread_num)


def get_edge_binary_feature(edges, feature_names, thread_num=1):
    """
    Fetch binary features of edges.

    Args:
      edges: A 2-d `Tensor` of `int64`, with shape `[num_edges, 3]`.
      feature_names: A list of `int`. Specify uint64 feature ids in graph to
        fetch features for nodes.

    Return:
      A list of `String Tensor` with the same length as `feature_names`.
    """
    return _get_binary_feature(edges, feature_names,
                               base._LIB_OP.get_edge_binary_feature,
                               thread_num)
