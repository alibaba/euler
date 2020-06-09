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

from transX import TransX
import tensorflow as tf
import tf_euler


class TransE(TransX):
    def __init__(self, node_type, edge_type,
                 node_max_id, edge_max_id,
                 ent_dim, rel_dim,
                 num_negs=5, margin=1.,
                 l1=True, metric_name='mrr',
                 corrupt='both',
                 *args, **kwargs):
        super(TransE, self).__init__(node_type, edge_type,
                                     node_max_id, edge_max_id,
                                     ent_dim, rel_dim,
                                     num_negs=num_negs, l1=l1,
                                     metric_name=metric_name,
                                     corrupt=corrupt,
                                     *args, **kwargs)
        if ent_dim != rel_dim:
            raise ValueError('Entity dim and Relation dim \
                              should be equal in TransE')
        self.margin = margin

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
        return loss
