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


class RelationDataFlow(object):
    def __init__(self, fanouts, metapath,
                 add_self_loops=True,
                 **kwargs):
        self.metapath = metapath

    def get_neighbors(self, n_id):
        neighbors = []
        types = []
        neighbor_src = []
        for i in range(len(self.metapath)):
            n_id = tf.reshape(n_id, [-1])
            one_neighbor, _, one_type = tf_euler.get_full_neighbor(n_id, self.metapath[i])
            neighbors.append(tf.reshape(one_neighbor.values, [-1]))
            types.append(tf.reshape(one_type.values, [-1]))
            one_indices = one_neighbor.indices[:, 0]
            neighbor_src.append(tf.cast(one_indices, tf.int32))
            new_n_id = tf.reshape(one_neighbor.values, [-1])
            n_id = tf.concat([new_n_id, n_id], axis=0)
            n_id, _ = tf.unique(n_id)

        return neighbors, types, neighbor_src

    def produce_subgraph(self, n_id):
        n_id = tf.reshape(n_id, [-1])
        inv = tf.range(tf.size(n_id))
        last_idx = inv

        data_flow = DataFlow(n_id)
        n_neighbors, types, n_edge_src = self.get_neighbors(n_id)
        for i in range(len(self.metapath)):
            new_n_id = n_neighbors[i]
            edge_src = n_edge_src[i]

            new_n_id = tf.concat([new_n_id, n_id], axis=0)
            new_n_id, new_inv = tf.unique(new_n_id)
            res_n_id = new_inv[-tf.size(n_id):]

            new_inv = new_inv[:-tf.size(n_id)]
            last_idx = new_inv

            n_id = new_n_id
            edge_dst = new_inv
            edge_index = tf.stack([edge_src, edge_dst], 0)
            e_id = types[i]
            data_flow.append(new_n_id, res_n_id, e_id, edge_index)
        return data_flow

    def __call__(self, n_id):
        producer = self.produce_subgraph
        return producer(n_id)
