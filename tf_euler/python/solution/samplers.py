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

class SampleNegWithTypes(object):
    def __init__(self, neg_type, num_negs=5):
        if not isinstance(neg_type, list):
            neg_type = [neg_type]
        self.num_negs = num_negs
        self.neg_type = neg_type

    def __call__(self, inputs):
        batch_size = tf.shape(inputs)[0]
        neg_nodes_group = []
        for n_type in self.neg_type:
            neg_node = tf_euler.sample_node(batch_size * self.num_negs, n_type)
            neg_node = tf.reshape(neg_node, [batch_size, self.num_negs])
            neg_nodes_group.append(neg_node)
        if len(self.neg_type) == 1:
            return neg_nodes_group[0]
        else:
            return neg_nodes_group

class SamplePosWithTypes(object):
    def __init__(self, edge_type, num_pos=1, max_id=-1):
        self.edge_type = edge_type
        self.num_pos = num_pos
        self.max_id = max_id

    def __call__(self, inputs):
        pos = tf_euler.sample_neighbor(inputs, self.edge_type, self.num_pos, self.max_id + 1)[0]
        return pos
