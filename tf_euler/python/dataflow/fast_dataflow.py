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


class FastGCNDataFlow(UniqueDataFlow):
    def __init__(self, fanouts, metapath,
                 add_self_loops=True,
                 **kwargs):
        super(FastGCNDataFlow, self).__init__(num_hops=len(metapath),
                                              add_self_loops=add_self_loops)
        self.fanouts = fanouts
        self.metapath = metapath

    def get_neighbors(self, n_id):
        neighbors = []
        neighbor_src = []
        total_fanout = 0
        for i in range(len(self.metapath)):
            n_id = tf.reshape(n_id, [-1])
            if i == len(self.metapath) - 1:
                one_neighbor = tf_euler.get_full_neighbor(n_id,
                                                          self.metapath[i])[0]
            else:
                total_fanout += self.fanouts[i]
                one_neighbor = tf_euler.sample_node(total_fanout,
                                                    self.metapath[i][0])
                one_neighbor, _ = tf.unique(one_neighbor)
                one_neighbor = tf_euler.sparse_get_adj(
                    n_id, one_neighbor, self.metapath[i], -1, -1)
            neighbors.append(tf.reshape(one_neighbor.values, [-1]))
            one_indices = one_neighbor.indices[:, 0]
            neighbor_src.append(tf.cast(one_indices, tf.int32))
            new_n_id = tf.reshape(one_neighbor.values, [-1])
            n_id = tf.concat([new_n_id, n_id], axis=0)
            n_id, _ = tf.unique(n_id)

        return neighbors, neighbor_src
