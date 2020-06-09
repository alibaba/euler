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
from tf_euler.python.dataflow.neighbor_dataflow import UniqueDataFlow
from tf_euler.python.dataflow.neighbor_dataflow import NeighborDataFlow


class LayerwiseDataFlow(UniqueDataFlow):
    def __init__(self, fanouts, metapath,
                 add_self_loops=True,
                 **kwargs):
        super(LayerwiseDataFlow, self).__init__(num_hops=len(metapath),
                                                add_self_loops=add_self_loops)
        self.fanouts = fanouts
        self.metapath = metapath

    def get_neighbors(self, n_id):
        neighbors = []
        neighbor_src = []
        last_count = tf.size(n_id)
        total_fanout = 0
        for i in range(len(self.metapath)):
            if i == len(self.metapath) - 1:
                one_neighbor = tf_euler.get_full_neighbor(
                    tf.reshape(n_id, [-1]), self.metapath[i])[0]
                one_indices = one_neighbor.indices[:, 0]
                one_neighbor = one_neighbor.values
            else:
                total_fanout += self.fanouts[i]
                last_count = tf.size(n_id)
                unique_neighbor, adj = tf_euler.sample_neighbor_layerwise(
                    tf.reshape(n_id, [1, last_count]),
                    self.metapath[i],
                    total_fanout)
                neighbor_idx = adj.indices[:, 2]
                one_neighbor = tf.gather(tf.reshape(unique_neighbor, [-1]),
                                         neighbor_idx)
                one_indices = adj.indices[:, 1]
            neighbors.append(tf.reshape(one_neighbor, [-1]))
            neighbor_src.append(tf.cast(one_indices, tf.int32))
            new_n_id = tf.reshape(one_neighbor, [-1])
            n_id = tf.concat([new_n_id, n_id], axis=0)
            n_id, _ = tf.unique(tf.reshape(n_id, [-1]))
        return neighbors, neighbor_src


class LayerwiseEachDataFlow(NeighborDataFlow):
    def __init__(self, fanouts, metapath,
                 add_self_loops=True,
                 max_id=-1,
                 **kwargs):
        super(LayerwiseEachDataFlow, self).__init__(
            num_hops=len(metapath),
            add_self_loops=add_self_loops)
        self.fanouts = fanouts
        self.metapath = metapath
        self.max_id = max_id

    def get_neighbors_sage(self, n_id):
        neighbors = []
        neighbor_src = []
        hop_edge_types = self.metapath[0]
        count = self.fanouts[0]
        n_id = tf.reshape(n_id, [-1])
        one_neighbor, _w, _ = tf_euler.sample_neighbor(
            n_id, hop_edge_types, count, defulat_node=self.max_id+1)
        neighbors.append(tf.reshape(one_neighbor, [-1]))
        node_src = tf.range(tf.size(n_id))
        node_src = tf.tile(tf.reshape(node_src, [-1, 1]), [1, count])
        node_src = tf.reshape(node_src, [-1])
        neighbor_src.append(node_src)
        return neighbors, neighbor_src

    def get_neighbors_layer(self, n_id):
        neighbors = []
        neighbor_src = []
        last_count = self.fanouts[0]
        for hop_edge_types, count in zip(self.metapath[1:], self.fanouts[1:]):
            unique_neighbor, adj = tf_euler.sample_neighbor_layerwise(
                tf.reshape(n_id, [-1, last_count]), hop_edge_types, count)
            neighbor_idx = adj.indices[:, 2] + adj.indices[:, 0] * count
            one_neighbor = tf.gather(tf.reshape(unique_neighbor, [-1]),
                                     neighbor_idx)
            one_indices = adj.indices[:, 1] + adj.indices[:, 0] * last_count
            neighbors.append(tf.reshape(one_neighbor, [-1]))
            neighbor_src.append(tf.cast(one_indices, tf.int32))
            n_id = one_neighbor
            last_count = count
        return neighbors, neighbor_src

    def get_neighbors(self, n_id):
        neighbors = []
        neighbor_src = []
        sage_neighbors, sage_neighbor_src = self.get_neighbors_sage(n_id)
        layer_neighbors, layer_neighbor_src = \
            self.get_neighbors_layer(sage_neighbors[-1])
        neighbors.extend(sage_neighbors)
        neighbors.extend(layer_neighbors)
        neighbor_src.extend(sage_neighbor_src)
        neighbor_src.extend(layer_neighbor_src)
        return neighbors, neighbor_src
