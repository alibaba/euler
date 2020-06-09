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


class AGNNConv(conv.Conv):

    def __init__(self, dim, **kwargs):
        super(AGNNConv, self).__init__(aggr='add', **kwargs)
        self.build = False

    def __call__(self, x, edge_index, size=None, **kwargs):
        if not self.build:
            self.build = True
            self.beta = tf.Variable([1.], name='beta', dtype=tf.float32)
        norm = \
            [tf.nn.l2_normalize(x[0], axis=-1) if x[0] is not None else None,
             tf.nn.l2_normalize(x[1], axis=-1) if x[1] is not None else None]
        gather_x, gather_norm, = self.gather_feature([x, norm], edge_index)
        out = self.apply_edge(edge_index[0],
                              gather_x[1],
                              gather_norm[0],
                              gather_norm[1],
                              size[0])
        out = mp_ops.scatter_(self.aggr, out, edge_index[0], size=size[0])
        out = self.apply_node(out)
        return out

    def apply_edge(self, edge_index_i, x_j, norm_i, norm_j, num_nodes):
        alpha = tf.reduce_sum(self.beta * (norm_i * norm_j),
                              axis=-1,
                              keep_dims=True)
        alpha = mp_ops.scatter_softmax(alpha, edge_index_i, num_nodes)
        return x_j * tf.reshape(alpha, [-1, 1])
