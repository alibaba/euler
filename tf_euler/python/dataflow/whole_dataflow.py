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
from tf_euler.python.dataflow.base_dataflow import DataFlow
from tf_euler.python.dataflow.neighbor_dataflow import NeighborDataFlow


class WholeDataFlow(NeighborDataFlow):
    def __init__(self, metapath,
                 add_self_loops=True,
                 **kwargs):
        super(WholeDataFlow, self).__init__(num_hops=len(metapath),
                                            add_self_loops=add_self_loops)
        self.neighbor_type = metapath[0]
        for n_type in metapath:
            if not n_type == self.neighbor_type:
                raise ValueError('Metapath should be the same in whole graph sampler.')

    def get_self_neighbors(self, n_id):
        edge_src = []
        edge_dst = []
        n_id = tf.reshape(n_id, [-1])
        one_neighbor = tf_euler.sparse_get_adj(n_id, n_id,
                                               self.neighbor_type, -1, -1)
        one_src = one_neighbor.indices[:, 0]
        one_dst = one_neighbor.indices[:, 1]
        edge_src = tf.cast(one_src, tf.int32)
        edge_dst = tf.cast(one_dst, tf.int32)
        return edge_src, edge_dst

    def produce_subgraph(self, n_id):
        inv = tf.range(tf.size(n_id))
        data_flow = DataFlow(n_id)
        edge_src, edge_dst = self.get_self_neighbors(n_id)
        new_n_id = n_id
        res_n_id = inv
        if self.add_self_loops:
            edge_dst = tf.concat([edge_dst, inv], axis=0)
            edge_src = tf.concat([edge_src, inv], axis=0)
        edge_index = tf.stack([edge_src, edge_dst], 0)
        e_id = None
        for i in range(self.num_hops):
            data_flow.append(new_n_id, res_n_id, e_id, edge_index)
        return data_flow
