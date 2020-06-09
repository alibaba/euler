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

from tf_euler.python.utils import layers


def _sparse_ones_like(sp_tensor):
    return tf.SparseTensor(sp_tensor.indices,
                           tf.ones_like(sp_tensor.values),
                           sp_tensor.dense_shape)


def _sparse_eye(num_rows, dtype=tf.float32):
    return tf.SparseTensor(
        tf.stack([tf.range(num_rows)] * 2, axis=1),
        tf.ones(num_rows, dtype), tf.stack([num_rows] * 2))


class GCNAggregator(layers.Layer):

    def __init__(self, dim, activation=tf.nn.relu, renorm=False, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)
        self.renorm = renorm
        self.dense = layers.Dense(dim, activation=activation, use_bias=False)

    def call(self, inputs):
        self_embedding, neigh_embedding, adj = inputs
        adj = _sparse_ones_like(adj)

        degree = tf.reshape(tf.sparse_reduce_sum(adj, 1), [-1, 1])
        agg_embedding = tf.sparse_tensor_dense_matmul(adj, neigh_embedding)
        if self.renorm:
            agg_embedding = (self_embedding + agg_embedding) / (1. + degree)
        else:
            agg_embedding = self_embedding + agg_embedding / \
                tf.maximum(degree, 1e-7)
        return self.dense(agg_embedding)


class MeanAggregator(layers.Layer):

    def __init__(self, dim, activation=tf.nn.relu, concat=False, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)
        if concat:
            dim //= 2
        self.concat = concat
        self.self_layer = layers.Dense(
            dim, activation=activation, use_bias=False)
        self.neigh_layer = layers.Dense(
            dim, activation=activation, use_bias=False)

    def call(self, inputs):
        self_embedding, neigh_embedding, adj = inputs
        adj = _sparse_ones_like(adj)

        degree = tf.reshape(tf.sparse_reduce_sum(adj, 1), [-1, 1])
        agg_embedding = tf.sparse_tensor_dense_matmul(adj, neigh_embedding) / \
            tf.maximum(degree, 1e-7)

        from_self = self.self_layer(self_embedding)
        from_neighs = self.neigh_layer(agg_embedding)

        if self.concat:
            return tf.concat([from_self, from_neighs], 1)
        else:
            return tf.add(from_self, from_neighs)


class SingleAttentionAggregator(layers.Layer):
    def __init__(self, dim, activation=tf.nn.relu, renorm=False, **kwargs):
        super(SingleAttentionAggregator, self).__init__(**kwargs)
        self.dense = layers.Dense(dim, use_bias=False)
        self.self_layer = layers.Dense(1, use_bias=False)
        self.neigh_layer = layers.Dense(1, use_bias=False)
        self.activation = activation
        self.renorm = renorm

    def call(self, inputs):
        self_embedding, neigh_embedding, adj = inputs
        adj = _sparse_ones_like(adj)
        if self.renorm:
            eye = _sparse_eye(adj.dense_shape[0])
            adj = tf.sparse_concat(1, [eye, adj])

        if not self.renorm:
            from_all = self.dense(neigh_embedding)
            from_self = self.dense(self_embedding)
        else:
            all_embedding = tf.concat([self_embedding, neigh_embedding], 0)
            from_all = self.dense(all_embedding)
            from_self = from_all[:adj.dense_shape[0], :]

        self_weight = self.self_layer(from_self)
        all_weight = self.neigh_layer(from_all)
        coefficient = tf.sparse_add(adj * self_weight,
                                    adj * tf.reshape(all_weight, [1, -1]))
        coefficient = tf.SparseTensor(
            coefficient.indices, tf.nn.leaky_relu(coefficient.values),
            coefficient.dense_shape)
        coefficient = tf.sparse_softmax(coefficient)

        output = tf.sparse_tensor_dense_matmul(coefficient, from_all)
        if not self.renorm:
            output = from_self + output
        if self.activation:
            output = self.activation(output)
        return output


class AttentionAggregator(layers.Layer):

    def __init__(self, dim, head_num=4, activation=tf.nn.relu, renorm=False,
                 **kwargs):
        super(AttentionAggregator, self).__init__(**kwargs)
        dim //= head_num
        self.attentions = [SingleAttentionAggregator(dim, activation, renorm)
                           for _ in range(head_num)]

    def call(self, inputs):
        return tf.concat([attention(inputs)
                          for attention in self.attentions], 1)


aggregators = {
    'gcn': GCNAggregator,
    'mean': MeanAggregator,
    'attention': AttentionAggregator
}


def get(aggregator):
    return aggregators.get(aggregator)
