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

from tf_euler.python.mp_utils.base_gnn import BaseGNNNet
from tf_euler.python.mp_utils.base import SuperviseModel, UnsuperviseModel


class BaseNode2Vec(UnsuperviseModel):

    def __init__(self, node_type, edge_type, max_id,
                 walk_len=3, walk_p=1, walk_q=1,
                 left_win_size=1, right_win_size=1, num_negs=5,
                 metric='mrr', neg_condition='', *args, **kwargs):
        super(BaseNode2Vec, self).__init__(
            node_type, edge_type, max_id, num_negs, metric, *args, **kwargs)
        self.node_type = node_type
        self.edge_type = edge_type
        self.max_id = max_id
        self.walk_len = walk_len
        self.walk_p = walk_p
        self.walk_q = walk_q
        self.left_win_size = left_win_size
        self.right_win_size = right_win_size
        self.num_negs = num_negs
        self.neg_condition = neg_condition
        self.batch_size_ratio = \
            int(tf_euler.gen_pair(tf.zeros([0, walk_len + 1], dtype=tf.int64),
                                  left_win_size, right_win_size).shape[1])

    def to_sample(self, inputs):
        batch_size = tf.size(inputs)
        path = tf_euler.random_walk(
            inputs, [self.edge_type] * self.walk_len,
            p=self.walk_p,
            q=self.walk_q,
            default_node=self.max_id + 1)
        pair = tf_euler.gen_pair(path, self.left_win_size, self.right_win_size)
        num_pairs = pair.shape[1]
        src, pos = tf.split(pair, [1, 1], axis=-1)
        src = tf.reshape(src, [batch_size * num_pairs, 1])
        pos = tf.reshape(pos, [batch_size * num_pairs, 1])
        negs = tf_euler.sample_node(batch_size * num_pairs * self.num_negs,
                                    self.node_type,
                                    condition=self.neg_condition)
        negs = tf.reshape(negs, [batch_size * num_pairs, self.num_negs])
        return src, pos, negs


class DeepWalk(BaseNode2Vec):

    def __init__(self, node_type, edge_type, max_id,
                 dim, walk_len=3, walk_p=1, walk_q=1,
                 left_win_size=1, right_win_size=1, num_negs=5,
                 feature_idx=-1, feature_dim=0, use_id=True,
                 embedding_dim=16, metric='mrr', combiner='add',
                 neg_condition='', *args, **kwargs):
        super(DeepWalk, self).__init__(
            node_type, edge_type, max_id,
            walk_len=walk_len, walk_p=walk_p, walk_q=walk_q,
            left_win_size=left_win_size, right_win_size=right_win_size,
            num_negs=num_negs, metric=metric,
            neg_condition=neg_condition, *args, **kwargs)

        self.dim = dim
        self._target_encoder = tf_euler.utils.encoders.ShallowEncoder(
            dim=dim, feature_idx=feature_idx, feature_dim=feature_dim,
            max_id=max_id if use_id else -1,
            embedding_dim=embedding_dim,
            combiner=combiner)
        self._context_encoder = tf_euler.utils.encoders.ShallowEncoder(
            dim=dim, feature_idx=feature_idx, feature_dim=feature_dim,
            max_id=max_id if use_id else -1,
            embedding_dim=embedding_dim,
            combiner=combiner)

    def embed(self, inputs):
        return self._target_encoder(inputs)

    def embed_context(self, inputs):
        return self._context_encoder(inputs)
