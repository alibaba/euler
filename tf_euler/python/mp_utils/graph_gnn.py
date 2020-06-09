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

from tf_euler.python.euler_ops import mp_ops
from tf_euler.python.mp_utils import utils
from tf_euler.python.graph_pool.base_pool import Pooling


class GraphGNNNet(object):

    def __init__(self, conv, dims,
                 fanouts, metapath,
                 node_pool=None,
                 graph_pool=Pooling,
                 add_self_loops=True,
                 jk_mode='concat'):
        conv_class = utils.get_conv_class(conv)
        flow_class = utils.get_flow_class('whole')
        self.whole_graph = True
        self.convs = []
        for dim in dims[:-1]:
            self.convs.append(self.get_conv(conv_class, dim))
        self.fc = tf.layers.Dense(dims[-1])
        self.sampler = flow_class(fanouts, [metapath[0]], add_self_loops)
        assert jk_mode in ['concat', 'maxpool']
        self.jk_mode = jk_mode
        if node_pool is not None:
            self.node_pool = [self.get_node_pool(node_pool)
                              for i in range(len(self.convs) // 2 + 1)]
        else:
            self.node_pool = None
        self.graph_pool = self.get_graph_pool(graph_pool)

    def get_node_pool(self, node_pool):
        raise NotImplementedError

    def get_graph_pool(self, graph_pool):
        return graph_pool('add')

    def get_conv(self, conv_class, dim):
        return conv_class(dim)

    def to_x(self, n_id):
        raise NotImplementedError

    def to_edge(self, n_id_src, n_id_dst, e_id):
        return e_id

    def get_edge_attr(self, block):
        n_id_dst = tf.cast(tf.expand_dims(block.n_id, -1),
                           dtype=tf.float32)
        n_id_src= mp_ops.gather(n_id_dst, block.res_n_id)
        n_id_src = mp_ops.gather(n_id_src,
                                 block.edge_index[0])
        n_id_dst = mp_ops.gather(n_id_dst,
                                 block.edge_index[1])
        n_id_src = tf.cast(tf.squeeze(n_id_src, -1), dtype=tf.int64)
        n_id_dst = tf.cast(tf.squeeze(n_id_dst, -1), dtype=tf.int64)
        edge_attr = self.to_edge(n_id_src, n_id_dst, block.e_id)
        return edge_attr

    def calculate_conv(self, conv, inputs, edge_index,
                       size=None, edge_attr=None):
        return conv(inputs, edge_index, size=size, edge_attr=edge_attr)

    def __call__(self, n_id, graph_index):
        data_flow = self.sampler(n_id)
        num_layers = len(self.convs)
        jk_hidden = []
        x = self.to_x(data_flow[0].n_id)
        block = data_flow[0]
        e_id, edge_index, size = block.e_id, block.edge_index, block.size
        for i, conv in zip(range(num_layers), self.convs):
            if e_id is None:
                edge_attr = None
            else:
                edge_attr = self.get_edge_attr(block)
            x_src, x_dst = x, None
            x = self.calculate_conv(conv,
                                    [x_src, x_dst],
                                    edge_index,
                                    size=size,
                                    edge_attr=edge_attr)
            x = tf.nn.relu(x)
            pool_out = self.graph_pool(x, graph_index)
            jk_hidden.append(pool_out)
            if self.node_pool is not None:
                if i % 2 == 0 and i != 0:
                    pool = self.node_pool[i // 2]
                    x, edge_index, _, graph_index, _, _ ,size = \
                        pool(x, edge_index, graph_index, size)
        if self.jk_mode == 'concat':
            pool_out = tf.concat(jk_hidden, axis=1)
        elif self.jk_mode == 'maxpool':
            pool_out = tf.reduce_sum(tf.stack(jk_hidden, 1), 1)
        out = self.fc(pool_out)
        return out

