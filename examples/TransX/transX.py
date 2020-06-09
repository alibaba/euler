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

import tensorflow as tf
import tf_euler

import collections

ModelOutput = collections.namedtuple(
    'ModelOutput', ['embedding', 'loss', 'metric_name', 'metric'])


class TransX(tf_euler.utils.layers.Layer):
    def __init__(self, node_type, edge_type,
                 node_max_id, edge_max_id,
                 ent_dim, rel_dim,
                 num_negs=5, l1=True, metric_name='mrr',
                 corrupt='both',
                 **kwargs):
        super(TransX, self).__init__(**kwargs)
        self.node_type = node_type
        self.edge_type = edge_type
        self.node_max_id = node_max_id
        self.edge_max_id = edge_max_id
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.num_negs = num_negs
        self.l1 = l1
        self.metric_name = metric_name
        self.corrupt = corrupt
        self.entity_encoder = tf_euler.utils.layers.Embedding(node_max_id+1,
                                                              ent_dim)
        self.relation_encoder = tf_euler.utils.layers.Embedding(edge_max_id+1,
                                                                rel_dim)

    def generate_negative(self, batch_size):
        return tf_euler.sample_node(batch_size * self.num_negs, self.node_type)

    def generate_triplets(self, inputs):
        batch_size = tf.shape(inputs)[0]
        src, dst, _ = tf.split(inputs, [1, 1, 1], axis=1)
        rel = tf_euler.get_edge_dense_feature(inputs, ['id'], [1])
        rel = tf.cast(rel, tf.int64)
        neg = self.generate_negative(batch_size)
        src = tf.reshape(src, [batch_size, 1])
        dst = tf.reshape(dst, [batch_size, 1])
        neg = tf.reshape(neg, [batch_size, self.num_negs])
        rel = tf.reshape(rel, [batch_size, 1])
        return src, dst, neg, rel

    def norm_emb(self, embedding):
        output_shape = embedding.shape
        embedding = tf.reshape(embedding, [-1, self.ent_dim])
        embedding = tf.nn.l2_normalize(embedding, 1)
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
        embedding = tf.reshape(embedding, output_shape)
        return embedding

    def calculate_scores(self, src_emb, rel_emb, dst_emb):
        if self.l1:
            scores = tf.norm(src_emb + rel_emb - dst_emb, ord=1, axis=-1)
        else:
            scores = tf.norm(src_emb + rel_emb - dst_emb,
                             ord='euclidean', axis=-1)
        scores = tf.negative(scores)
        return scores

    def _mrr(self, pos_scores, neg_scores):
        scores_all = tf.concat([neg_scores, pos_scores], axis=2)
        size = tf.shape(scores_all)[2]
        _, indices_of_ranks = tf.nn.top_k(scores_all, k=size)
        _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        return tf.reduce_mean(tf.reciprocal(tf.to_float(ranks[:, :, -1] + 1)))

    def _mr(self, pos_scores, neg_scores):
        scores_all = tf.concat([neg_scores, pos_scores], axis=2)
        size = tf.shape(scores_all)[2]
        _, indices_of_ranks = tf.nn.top_k(scores_all, k=size)
        ranks = tf.argmax(indices_of_ranks, -1)
        return tf.reduce_mean(ranks)

    def _hit10(self, pos_scores, neg_scores):
        scores_all = tf.concat([neg_scores, pos_scores], axis=2)
        size = tf.shape(scores_all)[2]
        _, indices_of_ranks = tf.nn.top_k(scores_all, k=size)
        ranks = tf.argmax(indices_of_ranks, -1)
        return tf.reduce_mean(tf.cast(tf.less(ranks, 10), tf.float32))

    def loss_fn(self, pos_scores, neg_scores):
        raise NotImplementedError()

    def calculate_energy(self, src_emb, dst_emb, neg_emb, rel_emb):
        # expand true triplets embedding dims to fit corrupted trplets
        src_expand_emb = tf.tile(src_emb, [1, self.num_negs, 1])
        rel_expand_emb = tf.tile(rel_emb, [1, self.num_negs, 1])
        dst_expand_emb = tf.tile(dst_emb, [1, self.num_negs, 1])

        pos_scores = self.calculate_scores(src_emb, rel_emb, dst_emb)
        pos_scores = tf.reshape(pos_scores, [-1, 1, 1])
        if self.corrupt == 'front':
            neg_scores = self.calculate_scores(neg_emb,
                                               rel_expand_emb,
                                               dst_expand_emb)
            neg_scores = tf.reshape(neg_scores, [-1, 1, self.num_negs])
        elif self.corrupt == 'tail':
            neg_scores = self.calculate_scores(src_expand_emb,
                                               rel_expand_emb,
                                               neg_emb)
            neg_scores = tf.reshape(neg_scores, [-1, 1, self.num_negs])
        else:
            front_neg_scores = self.calculate_scores(neg_emb,
                                                     rel_expand_emb,
                                                     dst_expand_emb)
            tail_neg_scores = self.calculate_scores(src_expand_emb,
                                                    rel_expand_emb,
                                                    neg_emb)
            neg_scores = tf.reshape(tf.concat([front_neg_scores,
                                              tail_neg_scores], axis=-1),
                                    [-1, 1, self.num_negs*2])

        loss = self.loss_fn(pos_scores, neg_scores)
        if self.metric_name == 'mrr':
            metric = self._mrr(pos_scores, neg_scores)
        elif self.metric_name == 'mr':
            metric = self._mr(pos_scores, neg_scores)
        elif self.metric_name == 'hit10':
            metric = self._hit10(pos_scores, neg_scores)
        else:
            msg = 'Metric name :{} not in list [mrr, mr, hit10]'\
                  .format(self.metric_name)
            raise ValueError(msg)
        return loss, metric

    def generate_embedding(self, src, dst, neg, rel):
        raise NotImplementedError()

    def call(self, inputs):
        src, dst, neg, rel = self.generate_triplets(inputs)
        src_emb, dst_emb, neg_emb, rel_emb = \
            self.generate_embedding(src, dst, neg, rel)
        loss, metric = self.calculate_energy(src_emb, dst_emb,
                                             neg_emb, rel_emb)
        src_emb = tf.reshape(src_emb, [-1, self.ent_dim])
        dst_emb = tf.reshape(dst_emb, [-1, self.ent_dim])
        rel_emb = tf.reshape(rel_emb, [-1, self.rel_dim])
        return ModelOutput(embedding=[src_emb, rel_emb, dst_emb],
                           loss=loss,
                           metric_name=self.metric_name,
                           metric=metric)
