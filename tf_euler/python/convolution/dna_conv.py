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


class GroupDense(tf.layers.Layer):
    def __init__(self, dim, groups=1,
                 activation=None,
                 use_bias=True,
                 **kwargs):
        super(GroupDense, self).__init__(**kwargs)
        self.dim = dim
        self.groups = groups
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.kernel = \
            tf.get_variable('kernel',
                            shape=[self.groups,
                                   input_shape[-1].value // self.groups,
                                   self.dim // self.groups])
        if self.use_bias:
            self.bias = tf.get_variable(
                    'bias',
                    shape=[self.dim])

    def call(self, inputs):
        if self.groups > 1:
            input_shape = inputs.shape
            output_shape = input_shape[:-1].concatenate(self.dim)
            output_shape = [d if d is not None else -1
                            for d in output_shape.as_list()]
            inputs = tf.reshape(inputs, [-1, self.groups,
                                         input_shape[-1] // self.groups])
            inputs = tf.transpose(inputs, [1, 0, 2])
            outputs = tf.matmul(inputs, self.kernel)
            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = tf.reshape(outputs, output_shape)
        else:
            outputs = tf.matmul(inputs, tf.reshape(self.weights,
                                                   [-1, self.dim]))
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs


def restricted_softmax(inputs, dim=-1, margin=0):
    input_max = tf.reduce_max(inputs, axis=dim, keepdims=True)[0]
    input_max = tf.clip_by_value(input_max,
                                 clip_value_min=0,
                                 clip_value_max=tf.float32.max)
    out = tf.exp((inputs - input_max))
    out = out / (tf.reduce_sum(out, axis=dim, keepdims=True) +
                 tf.exp(margin - input_max))
    return out


class DNAConv(conv.Conv):

    def __init__(self, dim, heads=1, groups=1, use_bias=True, **kwargs):
        super(DNAConv, self).__init__(aggr='mean', **kwargs)
        assert dim % heads == 0
        assert dim % groups == 0
        assert max(groups, heads) % min(groups, heads) == 0
        self.dim = dim
        self.heads = heads
        self.groups = groups
        self.use_bias = use_bias
        self.in_fc = tf.layers.Dense(dim, use_bias=False)
        self.lin_q = GroupDense(self.dim,
                                groups=self.groups,
                                use_bias=self.use_bias)
        self.lin_k = GroupDense(self.dim,
                                groups=self.groups,
                                use_bias=self.use_bias)
        self.lin_v = GroupDense(self.dim,
                                groups=self.groups,
                                use_bias=self.use_bias)

    @staticmethod
    def norm(edge_index, size):
        edge_weight = tf.ones([tf.shape(edge_index)[1], 1])

        def deg_inv_sqrt(i):
            deg = mp_ops.scatter_add(edge_weight, edge_index[i], size[i])
            return deg ** -0.5

        return tuple(map(deg_inv_sqrt, [0, 1]))

    def attention(self, query, key, value):
        assert query.shape.ndims == key.shape.ndims == value.shape.ndims >= 2
        assert query.shape[-1] == key.shape[-1]
        assert key.shape[-2] == value.shape[-2]

        score = tf.matmul(query, tf.transpose(key, [0, 1, 3, 2]))
        score = score / tf.sqrt(tf.cast(tf.shape(key)[-1], dtype=tf.float32))
        score = restricted_softmax(score, dim=-1)
        score = tf.matmul(score, value)
        return score

    def multi_head(self, query, key, value):
        query = self.lin_q(query)
        key = self.lin_k(key)
        value = self.lin_v(value)
        out_channels_per_head = self.dim // self.heads

        query_shape = [-1, tf.shape(query)[1],
                       self.heads, out_channels_per_head]
        query = tf.transpose(tf.reshape(query, query_shape), [1, 0, 2, 3])

        key_shape = [-1, tf.shape(key)[1], self.heads, out_channels_per_head]
        key = tf.transpose(tf.reshape(key, key_shape), [1, 0, 2, 3])

        value_shape = [-1, tf.shape(value)[1],
                       self.heads, out_channels_per_head]
        value = tf.transpose(tf.reshape(value, value_shape), [1, 0, 2, 3])

        out = self.attention(query, key, value)
        out = tf.transpose(out, [1, 0, 2, 3])
        out_shape = [-1, tf.shape(query)[1], self.dim]
        out = tf.reshape(out, out_shape)
        return out

    def __call__(self, x, edge_index, size=None, **kwargs):
        if isinstance(x, tf.Tensor):
            x = self.in_fc(x)
        else:
            x = (None if x[0] is None else self.in_fc(x[0]),
                 None if x[1] is None else self.in_fc(x[1]))
        norm = self.norm(edge_index, size)
        gather_x, gather_norm = self.gather_feature([x, norm], edge_index)
        out = self.apply_edge(gather_x[0],
                              gather_x[1],
                              gather_norm[0],
                              gather_norm[1])
        out = mp_ops.scatter_(self.aggr, out, edge_index[0], size=size[0])
        out = self.apply_node(out)
        return out

    def apply_edge(self, x_i, x_j, norm_i, norm_j):
        x_i = tf.expand_dims(x_i, 1)
        x_j = tf.expand_dims(x_j, 1)
        out = self.multi_head(x_i, x_j, x_j)
        out = tf.squeeze(out, 0)
        return norm_i * norm_j * out
