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

class DistMult(tf_euler.utils.layers.Layer):
    def __init__(self, node_type, edge_type,
                 node_max_id, edge_max_id,
                 ent_dim, rel_dim,
                 num_negs=5, margin=1,
                 metric_name='mrr',
                 corrupt='both',
                 l2_regular=False,
                 regular_param=0.0001,
                 **kwargs):
        super(DistMult, self).__init__(**kwargs)
        self.node_type = node_type
        self.edge_type = edge_type
        self.node_max_id = node_max_id
        self.edge_max_id = edge_max_id
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.num_negs = num_negs
        self.metric_name = metric_name
        self.corrupt = corrupt
        self.entity_encoder = tf_euler.utils.layers.Embedding(node_max_id+1,
                                                              ent_dim)
        self.relation_encoder = tf_euler.utils.layers.Embedding(edge_max_id+1,
                                                                rel_dim)
        self.margin = margin
        self.metric_class = tf_euler.utils.metrics.get(metric_name)
        self.l2_regular = l2_regular
        self.regular_param = regular_param

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
        M = tf.matrix_diag(rel_emb)
        scores = tf.einsum('aijk,aik->aij', M, dst_emb)
        scores = tf.reduce_sum(src_emb * scores, axis=-1, keep_dims=True)

        return scores

    def loss_fn(self, pos_scores, neg_scores):
        pos_scores = tf.reshape(pos_scores, [-1, 1, 1])
        if self.corrupt == 'both':
            neg_scores_mean = tf.reduce_mean(tf.reshape(neg_scores,
                                                        [-1, 2*self.num_negs]),
                                             axis=-1, keep_dims=True)
        else:
            neg_scores_mean = tf.reduce_mean(tf.reshape(neg_scores,
                                                        [-1, self.num_negs]),
                                             axis=-1, keep_dims=True)
        neg_scores_mean = tf.reshape(neg_scores_mean, [-1, 1, 1])
        loss = tf.reduce_mean(tf.maximum(self.margin + neg_scores_mean -
                                        pos_scores, 0)) 
        if self.l2_regular:
            loss += self.regular_param * tf.reduce_sum(tf.pow(self.entity_encoder.embeddings, 2))
            loss += self.regular_param * tf.reduce_sum(tf.pow(self.relation_encoder.embeddings, 2))
        return loss

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
        metric = self.metric_class(pos_scores, neg_scores)
        return loss, metric

    def generate_embedding(self, src, dst, neg, rel):
        src_emb = self.entity_encoder(src)
        dst_emb = self.entity_encoder(dst)
        neg_emb = self.entity_encoder(neg)
        rel_emb = self.relation_encoder(rel)
        src_emb = self.norm_emb(src_emb)
        dst_emb = self.norm_emb(dst_emb)
        neg_emb = self.norm_emb(neg_emb)
        rel_emb = self.norm_emb(rel_emb)
        return src_emb, dst_emb, neg_emb, rel_emb

    def call(self, inputs):
        src, dst, neg, rel = self.generate_triplets(inputs)
        src_emb, dst_emb, neg_emb, rel_emb = \
            self.generate_embedding(src, dst, neg, rel)
        loss, metric = self.calculate_energy(src_emb, dst_emb,
                                             neg_emb, rel_emb)
        src_emb = tf.reshape(src_emb, [-1, self.ent_dim])
        dst_emb = tf.reshape(dst_emb, [-1, self.ent_dim])
        rel_emb = tf.reshape(rel_emb, [-1, self.rel_dim])
        return ([src_emb, rel_emb, dst_emb], loss, self.metric_name, metric)
