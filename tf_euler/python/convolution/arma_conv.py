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

from tf_euler.python.convolution import conv
from tf_euler.python.euler_ops import mp_ops


class ARMAConv(conv.Conv):

    def __init__(self, dim, K, num_layers,
                 shared_weights=False, act=tf.nn.relu, **kwargs):
        super(ARMAConv, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.T = num_layers
        self.shared_weights = shared_weights
        self.act = act
        self.dim = dim
        if self.shared_weights:
            self.ws = [tf.layers.Dense(self.K * dim, use_bias=False)]
            self.vs = [tf.layers.Dense(self.K * dim, use_bias=False)]
        else:
            self.ws = [tf.layers.Dense(self.K * dim, use_bias=False)
                       for _ in range(self.T)]
            self.vs = [tf.layers.Dense(self.K * dim, use_bias=False)
                       for _ in range(self.T)]

    @staticmethod
    def norm(edge_index, size):
        edge_weight = tf.ones([tf.shape(edge_index)[1], 1])

        def deg_inv_sqrt(i):
            deg = mp_ops.scatter_add(edge_weight, edge_index[i], size[i])
            return deg ** -0.5

        return tuple(map(deg_inv_sqrt, [0, 1]))

    def __call__(self, x, edge_index, size=None, **kwargs):
        norm = self.norm(edge_index, size)
        origin = x
        for t in range(self.T):
            x, gather_norm, = self.gather_feature([x, norm], edge_index)
            out = self.apply_edge(x[1], gather_norm[0], gather_norm[1], t)
            out = mp_ops.scatter_(self.aggr, out, edge_index[0], size=size[0])
            out = self.apply_node(out, origin[0], t)
            x = [out, origin[1]]
        out = tf.reshape(out, [-1, self.K, self.dim])
        return tf.reduce_mean(out, axis=1)

    def apply_edge(self, x_j, norm_i, norm_j, t):
        w = self.ws[0 if self.shared_weights else t]
        x_j = w(x_j)
        return norm_i * norm_j * x_j

    def apply_node(self, aggr_out, origin, t):
        v = self.vs[0 if self.shared_weights else t]
        out = v(origin)
        out = aggr_out + out
        if self.act:
            out = self.act(out)
        return out
