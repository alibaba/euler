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


class GCNConv(conv.Conv):

    def __init__(self, dim, **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)
        self.fc = tf.layers.Dense(dim, use_bias=False)

    @staticmethod
    def norm(edge_index, size):
        edge_weight = tf.ones([tf.shape(edge_index)[1], 1])

        def deg_inv_sqrt(i):
            deg = mp_ops.scatter_add(edge_weight, edge_index[i], size[i])
            return deg ** -0.5

        return tuple(map(deg_inv_sqrt, [0, 1]))

    def __call__(self, x, edge_index, size=None, **kwargs):
        norm = self.norm(edge_index, size)
        gather_x, gather_norm, = self.gather_feature([x, norm], edge_index)
        out = self.apply_edge(gather_x[1], gather_norm[0], gather_norm[1])
        out = mp_ops.scatter_(self.aggr, out, edge_index[0], size=size[0])
        out = self.apply_node(out)
        return out

    def apply_edge(self, x_j, norm_i, norm_j):
        return norm_i * norm_j * x_j

    def apply_node(self, aggr_out):
        return self.fc(aggr_out)
