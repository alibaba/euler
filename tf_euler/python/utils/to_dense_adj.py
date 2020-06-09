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

import tensorflow as tf


def to_dense_adj(edge_index, batch, edge_attr=None):
    assert batch is not None
    batch_size = batch[-1] + 1

    num_nodes = tf.scatter_nd(
        tf.reshape(batch, [-1, 1]),
        tf.ones(batch.shape, dtype=tf.int32),
        shape=[batch_size])

    cum_nodes = tf.concat([tf.zeros(1, dtype=tf.int32),
                           tf.cumsum(num_nodes, axis=0)], axis=0)
    max_num_nodes = tf.reduce_max(num_nodes, axis=0)

    shape = [batch_size, max_num_nodes, max_num_nodes]
    if edge_attr is not None:
        edge_attr_shape = edge_attr.shape.as_list()
        shape = shape + edge_attr_shape[1:]

    edge_index_0 = tf.gather(batch, edge_index[0])  # [0, 0, 0, 1, 1, 1]

    cum_index = tf.gather(cum_nodes, edge_index_0)  # [0, 0, 0, 2, 2, 2]

    edge_index_1 = edge_index[0] - cum_index  # [0, 0, 1, 0, 1, 2]
    edge_index_2 = edge_index[1] - cum_index  # [0, 1, 0, 1, 2, 0]

    edge_index_0 = tf.reshape(edge_index_0, [-1, 1])
    edge_index_1 = tf.reshape(edge_index_1, [-1, 1])
    edge_index_2 = tf.reshape(edge_index_2, [-1, 1])

    idx = tf.concat([edge_index_0, edge_index_1, edge_index_2], axis=-1)

    if edge_attr is None:
        edge_attr = tf.ones(edge_index.shape[-1])
    adj = tf.scatter_nd(idx, edge_attr, shape)

    return adj
