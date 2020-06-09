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
from tf_euler.python.mp_utils.base_gnn import BaseGNNNet
from tf_euler.python.utils.encoders import ShallowEncoder


class GroupGNNNet(object):
    def __init__(self, gnns):
        self.group_gnn = gnns

    def __call__(self, group_n_id):
        group_x = []
        for n_id, gnn in zip(group_n_id, self.group_gnn):
            group_x.append(gnn(n_id))
        return group_x


class SharedGroupGNNNet(object):

    def __init__(self, conv, group_flow, dims,
                 group_fanouts, group_metapath,
                 add_self_loops=True):
        conv_class = utils.get_conv_class(conv)
        self.convs = []
        for dim in dims[:-1]:
            self.convs.append(self.get_conv(conv_class, dim))
        group_flow_class = [utils.get_flow_class(flow) for flow in group_flow]
        self.fc = tf.layers.Dense(dims[-1])
        if 'whole' in group_flow:
            raise ValueError('Group GNN does not support whole dataflow')
        self.group_sampler = [flow_class(fanouts, metapath, add_self_loops)
                              for flow_class, fanouts, metapath
                              in zip(group_flow_class, group_fanouts, group_metapath)]

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
                       size=None, edge_attr=None, edge_weight=None):
        return conv(inputs, edge_index, size=size, edge_attr=edge_attr, edge_weight=edge_weight)

    def __call__(self, group_n_id):
        group_x = []
        for sampler, n_id in zip(self.group_sampler, group_n_id):
            data_flow = sampler(n_id)
            num_layers = len(self.convs)
            x = self.to_x(data_flow[0].n_id)
            for i, conv, block in zip(range(num_layers), self.convs, data_flow):
                if block.e_id is None:
                    edge_attr = None
                else:
                    edge_attr = self.get_edge_attr(block)
                if block.edge_weight is None:
                    edge_weight = None
                else:
                    edge_weight = block.edge_weight
                x_src = mp_ops.gather(x, block.res_n_id)
                x_dst = x
                x = self.calculate_conv(conv,
                                        (x_src, x_dst),
                                        block.edge_index,
                                        size=block.size,
                                        edge_attr=edge_attr,
                                        edge_weight=edge_weight)
                x = tf.nn.relu(x)
            x = self.fc(x)
            group_x.append(x)
        return group_x

class SingleGNNNet(BaseGNNNet):
    def __init__(self, conv, flow, dims, fanouts, metapath, encoder, add_self_loops=False):
        super(SingleGNNNet, self).__init__(conv=conv,
                                           flow=flow,
                                           dims=dims,
                                           fanouts=fanouts,
                                           metapath=metapath,
                                           add_self_loops=add_self_loops)
        self.encoder = encoder

    def to_x(self, n_id):
        x = self.encoder(n_id)
        return x


class SharedGNNNet(SharedGroupGNNNet):
    def __init__(self, conv, group_flow, dims,
                 group_fanouts, group_metapath,
                 add_self_loops=True, **kwargs):
        super(SharedGNNNet, self).__init__(conv=conv,
                                                group_flow=group_flow,
                                                dims=dims,
                                                group_fanouts=group_fanouts,
                                                group_metapath=group_metapath,
                                                add_self_loops=add_self_loops)
        self.encoder = ShallowEncoder(**kwargs)

    def to_x(self, n_id):
        x = self.encoder(n_id)
        return x
