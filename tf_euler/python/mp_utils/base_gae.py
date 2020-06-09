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

class BaseGraphAutoEncoder(object):
    def __init__(self, node_type, edge_type, max_id, num_negs=20):
        self.node_type = node_type
        self.edge_type = edge_type
        self.max_id = max_id
        self.num_negs = num_negs

    def to_sample(self, inputs):
        batch_size = tf.size(inputs)
        src = tf.expand_dims(inputs, -1)
        pos = tf_euler.sample_neighbor(inputs, self.edge_type,
                                       self.num_negs, self.max_id + 1)[0]
        negs = tf_euler.sample_node(batch_size * self.num_negs,
                self.node_type)
        negs = tf.reshape(negs, [batch_size, self.num_negs])
        return src, pos, negs

    def embed(self, n_id):
        raise NotImplementedError

    def __call__(self, inputs):
        src, pos, negs = self.to_sample(inputs)
        embedding = self.embed(src)  # [batch, 1, dim]
        embedding_pos = self.embed(pos)  # [batch, num_negs, dim]
        embedding_negs = self.embed(negs)  # [batch, num_negs, dim]

        # [batch, 1, num_negs]
        logits = tf.matmul(embedding, embedding_pos, transpose_b=True)
        # [batch, 1, num_negs]
        neg_logits = tf.matmul(embedding, embedding_negs, transpose_b=True)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(logits), logits=logits)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_logits), logits=neg_logits)
        loss = tf.reduce_mean(tf.concat([tf.reshape(true_xent, [-1, 1]),
                                         tf.reshape(negative_xent,
                                         [-1, 1])], 0))
        predict = tf.nn.sigmoid(logits)
        neg_predict = tf.nn.sigmoid(neg_logits)
        label = tf.ones_like(logits)
        neg_label = tf.zeros_like(neg_logits)
        acc = tf_euler.utils.metrics.acc_score(
                tf.concat([label, neg_label], axis=2),
                tf.concat([predict, neg_predict], axis=2))

        embedding = self.embed(inputs)

        return (embedding, loss, 'acc', acc)

