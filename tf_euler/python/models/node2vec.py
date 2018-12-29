# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
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

from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python.models import base


class Node2Vec(base.UnsupervisedModel):
  """
  """

  def __init__(self,
               node_type,
               edge_type,
               max_id,
               dim,
               walk_len=3,
               walk_p=1,
               walk_q=1,
               left_win_size=1,
               right_win_size=1,
               num_negs=5,
               *args,
               **kwargs):
    super(Node2Vec, self).__init__(node_type, edge_type, max_id, *args,
                                   **kwargs)
    self.node_type = node_type
    self.edge_type = edge_type
    self.max_id = max_id
    self.dim = dim
    self.walk_len = walk_len
    self.walk_p = walk_p
    self.walk_q = walk_q
    self.left_win_size = left_win_size
    self.right_win_size = right_win_size
    self.num_negs = num_negs

    self.batch_size_ratio = \
        euler_ops.gen_pair(tf.zeros([0, walk_len + 1], dtype=tf.int64),
                           left_win_size, right_win_size).shape[1]

    self.target_embedding = layers.Embedding(
        name='target_embedding', max_id=max_id + 1, dim=dim)
    self.context_embedding = layers.Embedding(
        name='context_embedding', max_id=max_id + 1, dim=dim)

  def to_sample(self, inputs):
    batch_size = tf.size(inputs)
    path = euler_ops.random_walk(
        inputs, [self.edge_type] * self.walk_len,
        p=self.walk_p,
        q=self.walk_q,
        default_node=self.max_id + 1)
    pair = euler_ops.gen_pair(path, self.left_win_size, self.right_win_size)
    num_pairs = pair.shape[1]
    print(num_pairs)
    src, pos = tf.split(pair, [1, 1], axis=-1)
    src = tf.reshape(src, [batch_size * num_pairs, 1])
    pos = tf.reshape(pos, [batch_size * num_pairs, 1])
    negs = euler_ops.sample_node(batch_size * num_pairs * self.num_negs,
                                 self.node_type)
    negs = tf.reshape(negs, [batch_size * num_pairs, self.num_negs])
    return src, pos, negs

  def target_encoder(self, inputs):
    return self.target_embedding(inputs)

  def context_encoder(self, inputs):
    return self.context_embedding(inputs)
