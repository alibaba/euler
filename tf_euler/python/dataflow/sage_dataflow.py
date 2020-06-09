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

class SageDataFlow(UniqueDataFlow):
    def __init__(self, fanouts, metapath,
                 add_self_loops=True,
                 max_id=-1,
                 **kwargs):
        super(SageDataFlow, self).__init__(num_hops=len(metapath),
                                           add_self_loops=add_self_loops)
        self.fanouts = fanouts
        self.metapath = metapath
        self.max_id = max_id

    def get_neighbors(self, n_id):
        neighbors = []
        neighbor_src = []
        for hop_edge_types, count in zip(self.metapath, self.fanouts):
            n_id = tf.reshape(n_id, [-1])
            one_neighbor, _w, _ = tf_euler.sample_neighbor(
                n_id, hop_edge_types, count, default_node=self.max_id+1)
            neighbors.append(tf.reshape(one_neighbor, [-1]))
            node_src = tf.range(tf.size(n_id))
            node_src = tf.tile(tf.reshape(node_src, [-1, 1]), [1, count])
            node_src = tf.reshape(node_src, [-1])
            neighbor_src.append(node_src)
            new_n_id = tf.reshape(one_neighbor, [-1])
            n_id = tf.concat([new_n_id, n_id], axis=0)
            n_id, _ = tf.unique(tf.reshape(n_id, [-1]))
        return neighbors, neighbor_src
