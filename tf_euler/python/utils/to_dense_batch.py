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
from tf_euler.python.euler_ops import mp_ops


def to_dense_batch(x, batch):
    assert batch is not None
    batch_size = tf.reduce_max(batch) + 1
    num_nodes = mp_ops.scatter_('add', tf.ones([tf.shape(batch)[0], 1]),
                                batch, batch_size)
    num_nodes = tf.cast(tf.reshape(num_nodes, [-1]), dtype=tf.int32)

    cum_nodes = tf.concat([tf.zeros(1, dtype=tf.int32),
                           tf.cumsum(num_nodes, axis=0)], axis=0)
    max_num_nodes = tf.reduce_max(num_nodes)

    idx = tf.range(tf.reduce_sum(num_nodes))

    n = tf.gather(cum_nodes, batch)
    idx = idx - n + batch * max_num_nodes

    idx = tf.reshape(idx, [-1, 1])

    size = [batch_size * max_num_nodes, tf.shape(x)[-1]]

    out = tf.scatter_nd(idx, x, shape=size)

    out_size = [batch_size, max_num_nodes, tf.shape(x)[-1]]
    out = tf.reshape(out, out_size)

    mask = tf.scatter_nd(idx,
                         tf.ones(tf.shape(batch)[0]),
                         shape=[batch_size * max_num_nodes])
    return out, out_size, mask
