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


class NeighborDataFlow(object):
    def __init__(self, num_hops,
                 add_self_loops=True,
                 **kwargs):
        self.num_hops = num_hops
        self.add_self_loops = add_self_loops

    def get_neighbors(self, n_id):
        '''
        The neighbor sampler in a mini-batch of n_id.
        It returns: neighbors: a list of 'tensor';
                    neighbor_src: a list of 'tensor'
        Input:
          n_id: input nodes
        Output:
          neighbors: [[n_id's neighbor], [n_id's neighbor's neighbor], ...]
          neighbor_src: [[n_neighbor_src], [n_neighbor_neighbor_src], ...]
        '''
        raise NotImplementedError()

    def produce_subgraph(self, n_id):
        n_id = tf.reshape(n_id, [-1])
        inv = tf.range(tf.size(n_id))
        last_idx = inv

        data_flow = DataFlow(n_id)
        n_neighbors, n_edge_src = self.get_neighbors(n_id)
        for i in range(self.num_hops):
            new_n_id = n_neighbors[i]
            edge_src = n_edge_src[i]

            new_n_id = tf.concat([new_n_id, n_id], axis=0)
            new_inv = tf.range(tf.size(new_n_id))
            res_n_id = new_inv[-tf.size(n_id):]
            if self.add_self_loops:
                edge_src = tf.concat([edge_src, last_idx], axis=0)
                last_idx = new_inv
            else:
                new_inv = new_inv[:-tf.size(n_id)]
                last_idx = new_inv

            n_id = new_n_id
            edge_dst = new_inv
            edge_index = tf.stack([edge_src, edge_dst], 0)
            e_id = None
            data_flow.append(new_n_id, res_n_id, e_id, edge_index)
        return data_flow

    def __call__(self, n_id):
        producer = self.produce_subgraph
        return producer(n_id)


class UniqueDataFlow(NeighborDataFlow):
    def __init__(self, num_hops,
                 add_self_loops=True):
        super(UniqueDataFlow, self).__init__(num_hops=num_hops,
                                             add_self_loops=add_self_loops)

    def produce_subgraph(self, n_id):
        n_id = tf.reshape(n_id, [-1])
        inv = tf.range(tf.size(n_id))
        last_idx = inv

        data_flow = DataFlow(n_id)
        n_neighbors, n_edge_src = self.get_neighbors(n_id)
        for i in range(self.num_hops):
            new_n_id = n_neighbors[i]
            edge_src = n_edge_src[i]

            new_n_id = tf.concat([new_n_id, n_id], axis=0)
            new_n_id, new_inv = tf.unique(new_n_id)
            res_n_id = new_inv[-tf.size(n_id):]
            if self.add_self_loops:
                edge_src = tf.concat([edge_src, last_idx], axis=0)
                last_idx = tf.range(tf.size(new_n_id))
            else:
                new_inv = new_inv[:-tf.size(n_id)]
                last_idx = new_inv

            n_id = new_n_id
            edge_dst = new_inv
            edge_index = tf.stack([edge_src, edge_dst], 0)
            e_id = None
            data_flow.append(new_n_id, res_n_id, e_id, edge_index)
        return data_flow
