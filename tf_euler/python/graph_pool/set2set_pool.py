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
from tf_euler.python.graph_pool.base_pool import Pooling


class Set2SetPool(Pooling):

    def __init__(self, dim, processing_steps, num_layers, aggr='add'):
        super(Set2SetPool, self).__init__(aggr=aggr)
        self.dim = dim
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=dim)
                      for _ in range(num_layers)]
        self.lstm = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)

    def __call__(self, inputs, index, size=None):
        size = tf.reduce_max(index) + 1 if size is None else size
        cell_in = tf.zeros([size, self.dim*2], dtype=tf.float32)
        hidden_state = self.lstm.zero_state(tf.shape(cell_in)[0],
                                            dtype=tf.float32)
        for i in range(self.processing_steps):
            q = tf.expand_dims(cell_in, axis=1)
            q, hidden_state = tf.nn.dynamic_rnn(self.lstm,
                                                q,
                                                initial_state=hidden_state,
                                                dtype=tf.float32)
            q = tf.reshape(q, [-1, self.dim])
            e = tf.reduce_sum((inputs * tf.gather(q, index)),
                              axis=-1,
                              keep_dims=True)
            a = mp_ops.scatter_softmax(e, index, size=size)
            r = mp_ops.scatter_(self.aggr, a * inputs, index, size=size)
            cell_in = tf.reshape(tf.concat([q, r], axis=-1), [-1, self.dim*2])
        return cell_in
