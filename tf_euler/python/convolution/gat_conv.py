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
import tf_euler

from tf_euler.python.convolution import conv
from tf_euler.python.euler_ops import mp_ops


class Attention(tf.layers.Layer):

    def __init__(self, dim):
        super(Attention, self).__init__(self)
        self.dim = dim

    def build(self, input_shape):
        self.fc = tf.layers.Dense(1, use_bias=False)
        self.built = True

    def call(self, inputs):
        return self.fc(inputs)


class GATConv(conv.Conv):

    def __init__(self, dim, improved=False, aggr='add', **kwargs):
        super(GATConv, self).__init__(aggr=aggr, **kwargs)

        self.dim = dim
        self.improved = improved

        self.fc = tf.layers.Dense(dim, use_bias=False)
        self.att_i = Attention(dim)
        self.att_j = Attention(dim)

    def __call__(self, x, edge_index, size=None, **kwargs):
        if isinstance(x, tf.Tensor):
            x = self.fc(x)
        else:
            x = (None if x[0] is None else self.fc(x[0]),
                 None if x[1] is None else self.fc(x[1]))

        gather_x, = self.gather_feature([x], edge_index)
        out = self.apply_edge(gather_x[0], gather_x[1], edge_index[0], size[0])
        out = mp_ops.scatter_(self.aggr, out, edge_index[0], size=size[0])
        out = self.apply_node(out, x[0])
        return out

    def apply_edge(self, x_i, x_j, edge_index_i, size_i):
        alpha = self.att_i(x_i) + self.att_j(x_j)
        alpha = tf.nn.leaky_relu(alpha)
        alpha = mp_ops.scatter_softmax(alpha, edge_index_i, size_i)

        x_j = x_j * tf.reshape(alpha, [-1, 1])
        return tf.reshape(x_j, [-1, self.dim])

    def apply_node(self, aggr_out, x):
        if self.improved:
            aggr_out = x + aggr_out

        return aggr_out
