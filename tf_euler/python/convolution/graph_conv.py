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


class GraphConv(conv.Conv):

    def __init__(self, dim, **kwargs):
        super(GraphConv, self).__init__(aggr='mean', **kwargs)
        self.fc = tf.layers.Dense(dim, use_bias=False)
        self.liner = tf.layers.Dense(dim, use_bias=True)

    def __call__(self, x, edge_index, size=None, **kwargs):
        h = [None if x[0] is None else self.fc(x[0]),
             None if x[1] is None else self.fc(x[1])]
        gather_x, gather_h = self.gather_feature([x, h], edge_index)
        out = self.apply_edge(gather_h[1])
        out = mp_ops.scatter_(self.aggr, out, edge_index[0], size=size[0])
        out = self.apply_node(out, x[0])
        return out

    def apply_edge(self, x_j):
        return x_j

    def apply_node(self, aggr_out, x):
        return self.liner(x) + aggr_out
