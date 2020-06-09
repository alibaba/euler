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


class AttentionPool(Pooling):

    def __init__(self, gate_nn=None, nn=None, aggr='add'):
        super(AttentionPool, self).__init__(aggr=aggr)
        if gate_nn is None:
            self.gate_nn = tf.layers.Dense(1, use_bias=False)
        else:
            self.gate_nn = gate_nn
        self.nn = nn
        self.aggr = aggr

    def __call__(self, inputs, index, size=None):
        size = tf.reduce_max(index) + 1 if size is None else size

        gate = self.gate_nn(inputs)
        inputs = self.nn(inputs) if self.nn is not None else inputs

        gate = mp_ops.scatter_softmax(gate, index, size=size)
        outputs = mp_ops.scatter_(self.aggr, gate * inputs, index, size=size)
        return outputs
