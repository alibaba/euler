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


class GatedConv(conv.Conv):

    def __init__(self, dim, processing_steps=2, lstm_layers=2, **kwargs):
        super(GatedConv, self).__init__(aggr='add', **kwargs)
        self.dim = dim
        self.processing_steps = processing_steps
        self.lstm_layers = lstm_layers
        self.fc = [tf.layers.Dense(dim, use_bias=False)
                   for _ in range(processing_steps)]
        lstm_cells = [tf.nn.rnn_cell.GRUCell(num_units=dim)
                      for _ in range(lstm_layers)]
        self.rnn = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

    def __call__(self, x, edge_index, size=None, **kwargs):
        h = x
        for i in range(self.processing_steps):
            m = [None if h[0] is None else self.fc[i](h[0]),
                 None if h[1] is None else self.fc[i](h[1])]
            gather_x, = self.gather_feature([m], edge_index)
            out = self.apply_edge(gather_x[1])
            out = mp_ops.scatter_(self.aggr, out, edge_index[0], size=size[0])
            out = self.apply_node(out)
            out = tf.expand_dims(out, axis=1)
            hidden_state = [h[0] for _ in range(self.lstm_layers)]
            with tf.variable_scope('rnn', reuse=tf.AUTO_REUSE):
                out, _ = tf.nn.dynamic_rnn(self.rnn,
                                           out,
                                           initial_state=tuple(hidden_state),
                                           dtype=tf.float32)
            out = tf.reshape(out, [-1, self.dim])
            h = [out, h[1]]
        return out

    def apply_edge(self, x_j):
        return x_j
