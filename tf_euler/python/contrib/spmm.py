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


def spmm_add(src, edge_index, size, flow='target_to_source'):
    adj_indices = tf.stack([edge_index[0], edge_index[1]], axis=1)
    adj_values = tf.ones(tf.shape(edge_index)[1])
    adj_shape = size
    adj = tf.SparseTensor(tf.cast(adj_indices, tf.int64),
                          adj_values,
                          adj_shape)
    adj = tf.sparse_reorder(adj)
    return tf.sparse_tensor_dense_matmul(adj, src)


def spmm_mean(src, edge_index, size, flow='target_to_source'):
    out = spmm_add(src, edge_index, size, flow)
    count = spmm_add(tf.ones([tf.shape(src)[0], 1]), edge_index, size, flow)
    return out / count


def spmm_(op, src, edge_index, size, flow='target_to_source'):
    return globals()['spmm_' + op](src, edge_index, size, flow)
