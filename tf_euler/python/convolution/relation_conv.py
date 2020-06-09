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
import collections

from tf_euler.python.convolution import conv
from tf_euler.python.euler_ops import mp_ops

_RGCN_UIDS = collections.defaultdict(lambda: 0)

def get_uid(name=''):
  _RGCN_UIDS[name] += 1
  return _RGCN_UIDS[name]


class RelationConv(conv.Conv):

    def __init__(self, fea_dim, dim, metapath,
                 total_relation_num, **kwargs):
        super(RelationConv, self).__init__(aggr='mean', **kwargs)
        self.fea_dim = fea_dim
        self.dim = dim
        self.relation_num = total_relation_num
        self.fc = tf.layers.Dense(dim, use_bias=False)
        self.built = False

    def build(self):
        num = self.relation_num
        dim = self.dim
        fea_dim = self.fea_dim
        self.matrix = tf.get_variable('matrix_' + str(get_uid('matrix')), shape=[num, dim, fea_dim], 
                                      initializer=tf.variance_scaling_initializer())

        self.built = True

    def __call__(self, x, edge_index, size=None, edge_attr=None, **kwarg):
        assert edge_attr is not None
        if not self.built:
            self.build()
        gather_x, = self.gather_feature([x], edge_index)
        out = self.apply_edge(gather_x[1], edge_attr)
        out = mp_ops.scatter_(self.aggr, out, edge_index[0], size=size[0])
        out = self.apply_node(x[0], out)
        return out

    def apply_edge(self, x_j, edge_attr):
        edge_attr, idx = tf.unique(edge_attr)
        matrix = tf.gather(self.matrix, edge_attr)
        matrix = tf.gather(matrix, idx)
        matrix = tf.reshape(matrix, [-1, self.dim, self.fea_dim])
        x_j = tf.expand_dims(x_j, -1)
        res = tf.matmul(matrix, x_j)
        return tf.reshape(res, [-1, self.dim])

    def apply_node(self, x, aggr_out):
        return self.fc(x) + aggr_out
