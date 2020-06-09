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


class GINConv(conv.Conv):

    def __init__(self, dim, mlp=None, eps=0.0, train_eps=True, **kwargs):
        super(GINConv, self).__init__(aggr='add', **kwargs)
        if mlp is None:
            self.mlp = tf.layers.Dense(dim, use_bias=False)
        else:
            self.mlp = mlp
        self.eps_value = eps
        self.train_eps = train_eps
        self.build = False

    def __call__(self, x, edge_index, size=None, **kwargs):
        if not self.build:
            self.build = True
            if self.train_eps:
                self.eps = tf.Variable([self.eps_value],
                                       name='eps',
                                       dtype=tf.float32)
            else:
                self.eps = self.eps_value
        gather_x, = self.gather_feature([x], edge_index)
        out = self.apply_edge(gather_x[1])
        out = mp_ops.scatter_(self.aggr, out, edge_index[0], size=size[0])
        out = self.apply_node(out, x[0])
        return out

    def apply_edge(self, x_j):
        return x_j

    def apply_node(self, aggr_out, x):
        return self.mlp((1 + self.eps) * x + aggr_out)
