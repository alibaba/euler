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
from tf_euler.python.euler_ops import type_ops

_sample_neighbor = base._LIB_OP.sample_neighbor
_get_top_k_neighbor = base._LIB_OP.get_top_k_neighbor
_sample_fanout = base._LIB_OP.sample_fanout
_sample_neighbor_layerwise_with_adj = \
    base._LIB_OP.sample_neighbor_layerwise_with_adj
_sample_fanout_with_feature = base._LIB_OP.sample_fanout_with_feature


def sparse_get_adj(nodes, nb_nodes, edge_types, n=-1, m=-1):
    edge_types = type_ops.get_edge_type_id(edge_types)
    res = base._LIB_OP.sparse_get_adj(nodes, nb_nodes, edge_types, n, m)
    return tf.SparseTensor(*res[:3])


def sample_neighbor(nodes, edge_types, count, default_node=-1, condition=''):
    edge_types = type_ops.get_edge_type_id(edge_types)
    return _sample_neighbor(nodes, edge_types, count, default_node, condition)


def get_top_k_neighbor(nodes, edge_types, k, default_node=-1, condition=''):
    edge_types = type_ops.get_edge_type_id(edge_types)
    return _get_top_k_neighbor(nodes, edge_types, k, default_node, condition)


def sample_fanout_with_feature(nodes, edge_types, count, default_node,
                               dense_feature_names, dense_dimensions,
                               sparse_feature_names, sparse_default_values):
    edge_types = type_ops.get_edge_type_id(edge_types)
    res = _sample_fanout_with_feature(
        tf.reshape(nodes, [-1]), edge_types, count,
        default_node=default_node,
        sparse_feature_names=sparse_feature_names,
        sparse_default_values=sparse_default_values,
        dense_feature_names=dense_feature_names,
        dense_dimensions=dense_dimensions,
        N=len(count),
        ND=(len(count) + 1) * len(dense_feature_names),
        NS=(len(count) + 1) * len(sparse_feature_names))
    neighbors = [tf.reshape(nodes, [-1])]
    neighbors.extend([tf.reshape(i, [-1]) for i in res[0]])
    weights = res[1]
    types = res[2]
    dense_features = res[3]
    sparse_features = [tf.SparseTensor(*sp) for sp in zip(*res[4:7])]
    return neighbors, weights, types, dense_features, sparse_features


def sample_neighbor_layerwise(nodes, edge_types, count,
                              default_node=-1, weight_func=''):
    edge_types = type_ops.get_edge_type_id(edge_types)
    res = _sample_neighbor_layerwise_with_adj(nodes, edge_types, count,
                                              weight_func, default_node)
    return res[0], tf.SparseTensor(*res[1:4])


def get_full_neighbor(nodes, edge_types, condition=''):
    """
    Args:
      nodes: A `Tensor` of `int64`.
      edge_types: A 1-D `Tensor` of int32. Specify edge types to filter
        outgoing edges.

    Return:
      A tuple of `SparseTensor` (neibors, weights).
        neighbors: A `SparseTensor` of `int64`.
        weights: A `SparseTensor` of `float`.
        types: A `SparseTensor` of `int32`
    """
    edge_types = type_ops.get_edge_type_id(edge_types)
    sp_returns = base._LIB_OP.get_full_neighbor(nodes, edge_types, condition)
    return tf.SparseTensor(*sp_returns[:3]), \
        tf.SparseTensor(*sp_returns[3:6]), \
        tf.SparseTensor(*sp_returns[6:])


def get_sorted_full_neighbor(nodes, edge_types, condition=''):
    """
    Args:
      nodes: A `Tensor` of `int64`.
      edge_types: A 1-D `Tensor` of int32. Specify edge types to filter
        outgoing edges.

    Return:
      A tuple of `SparseTensor` (neibors, weights).
        neighbors: A `SparseTensor` of `int64`.
        weights: A `SparseTensor` of `float`.
        types: A `SparseTensor` of `int32`
    """
    edge_types = type_ops.get_edge_type_id(edge_types)
    sp_returns = base._LIB_OP.get_sorted_full_neighbor(nodes,
                                                       edge_types,
                                                       condition)
    return tf.SparseTensor(*sp_returns[:3]),\
        tf.SparseTensor(*sp_returns[3:6]),\
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
      default_node: A `int`. Specify the node id to fill when there is no
        neighbor for specific nodes.

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
    edge_types = [type_ops.get_edge_type_id(edge_type)
                  for edge_type in edge_types]
    neighbors_list = [tf.reshape(nodes, [-1])]
    weights_list = []
    type_list = []
    neighbors, weights, types = _sample_fanout(
        neighbors_list[-1],
        edge_types, counts,
        default_node=default_node,
        N=len(counts))
    neighbors_list.extend([tf.reshape(n, [-1]) for n in neighbors])
    weights_list.extend([tf.reshape(w, [-1]) for w in weights])
    type_list.extend([tf.reshape(t, [-1]) for t in types])
    return neighbors_list, weights_list, type_list


def sample_fanout_layerwise_each_node(nodes, edge_types, counts,
                                      default_node=-1):
    '''
      sample fanout layerwise for each node
    '''
    edge_types = [type_ops.get_edge_type_id(edge_type)
                  for edge_type in edge_types]
    neighbors_list = [tf.reshape(nodes, [-1])]
    adj_list = []
    for hop_edge_types, count in zip(edge_types, counts):
        if (len(neighbors_list) == 1):
            neighbors, _, _ = sample_neighbor(neighbors_list[-1],
                                              hop_edge_types,
                                              count,
                                              default_node=default_node)
            neighbors_list.append(tf.reshape(neighbors, [-1]))
        else:
            neighbors, adj = sample_neighbor_layerwise(
                tf.reshape(neighbors_list[-1], [-1, last_count]),
                hop_edge_types,
                count,
                default_node=default_node)
            neighbors_list.append(tf.reshape(neighbors, [-1]))
            adj_list.append(adj)
        last_count = count
    return neighbors_list, adj_list


def sample_fanout_layerwise(nodes, edge_types, counts,
                            default_node=-1, weight_func=''):
    edge_types = [type_ops.get_edge_type_id(edge_type)
                  for edge_type in edge_types]
    neighbors_list = [tf.reshape(nodes, [-1])]
    adj_list = []
    last_count = tf.size(nodes)
    for hop_edge_types, count in zip(edge_types, counts):
        neighbors, adj = sample_neighbor_layerwise(
            tf.reshape(neighbors_list[-1], [-1, last_count]),
            hop_edge_types,
            count,
            default_node=default_node,
            weight_func=weight_func)
        neighbors_list.append(tf.reshape(neighbors, [-1]))
        adj_list.append(adj)
        last_count = count
    return neighbors_list, adj_list


def get_multi_hop_neighbor(nodes, edge_types):
    """
    Get multi-hop neighbors with adjacent matrix.

    Args:
      nodes: A 1-D `tf.Tensor` of `int64`.
      edge_types: A list of 1-D `tf.Tensor` of `int32`. Specify edge types to
        filter outgoing edges in each hop.

    Return:
      A tuple of list: (nodes, adjcents)
        nodes: A list of N + 1 `tf.Tensor` of `int64`, N is the number of
          hops. Specify node set of each hop, including the root.
        adjcents: A list of N `tf.SparseTensor` of `int64`. Specify adjacent
          matrix between hops.
    """
    edge_types = [type_ops.get_edge_type_id(edge_type)
                  for edge_type in edge_types]
    nodes = tf.reshape(nodes, [-1])
    nodes_list = [nodes]
    adj_list = []
    for hop_edge_types in edge_types:
        neighbor, weight, _ = get_full_neighbor(nodes, hop_edge_types)
        next_nodes, next_idx = tf.unique(neighbor.values, out_idx=tf.int64)
        next_indices = tf.stack([neighbor.indices[:, 0], next_idx], 1)
        next_values = weight.values
        next_shape = tf.stack([tf.size(nodes), tf.size(next_nodes)])
        next_shape = tf.cast(next_shape, tf.int64)
        next_adj = tf.SparseTensor(next_indices, next_values, next_shape)
        next_adj = tf.sparse_reorder(next_adj)
        nodes_list.append(next_nodes)
        adj_list.append(next_adj)
        nodes = next_nodes
    return nodes_list, adj_list
