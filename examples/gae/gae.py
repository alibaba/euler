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

from tf_euler.python.utils.encoders import ShallowEncoder
from tf_euler.python.mp_utils.base_gnn import BaseGNNNet
from tf_euler.python.mp_utils.base_gae import BaseGraphAutoEncoder

class GNN(BaseGNNNet):

    def __init__(self, conv, flow,
                 dims, fanouts, metapath,
                 feature_idx, feature_dim,
                 add_self_loops=False):
        super(GNN, self).__init__(conv=conv,
                                  flow=flow,
                                  dims=dims,
                                  fanouts=fanouts,
                                  metapath=metapath,
                                  add_self_loops=add_self_loops)
        if not isinstance(feature_idx, list):
            feature_idx = [feature_idx]
        if not isinstance(feature_dim, list):
            feature_dim = [feature_dim]
        self.feature_idx = feature_idx
        self.feature_dim = feature_dim

    def to_x(self, n_id):
        x, = tf_euler.get_dense_feature(n_id,
                                        self.feature_idx,
                                        self.feature_dim)
        return x

class GraphAutoEncoder(BaseGraphAutoEncoder):

    def __init__(self, node_encoder, dims, fanouts, metapath,
                 feature_idx, feature_dim,
                 node_type, edge_type, max_id, num_negs=5):
        super(GraphAutoEncoder, self).__init__(node_type, edge_type,
                                               max_id, num_negs)
        if node_encoder == 'gcn':
            self.gnn = GNN('gcn', 'full', dims, None, metapath,
                           feature_idx, feature_dim)
        else:
            self.gnn = GNN('sage', 'sage', dims, fanouts, metapath,
                           feature_idx, feature_dim)

        self.dim = dims[-1]

    def embed(self, n_id):
        batch_size = tf.shape(n_id)[0]
        n_id = tf.reshape(n_id, [-1])
        emb = self.gnn(n_id)
        emb = tf.reshape(emb, [batch_size, -1, self.dim])
        return emb

class VariationalGraphAutoEncoder(BaseGraphAutoEncoder):

    def __init__(self, radius, node_encoder, dims, fanouts, metapath,
                 feature_idx, feature_dim,
                 node_type, edge_type, max_id, num_negs=5, train=True):
        super(VariationalGraphAutoEncoder, self).__init__(
                node_type, edge_type, max_id, num_negs)
        self.log_var_encoder = ShallowEncoder(
                dim = dims[-1], feature_idx = -1, max_id = max_id,
                combiner = 'add')
        if node_encoder == 'gcn':
            self.gnn = GNN('gcn', 'full', dims, None, metapath,
                           feature_idx, feature_dim)
        else:
            self.gnn = GNN('sage', 'sage', dims, fanouts, metapath,
                           feature_idx, feature_dim)
        self.dim = dims[-1]
        self.train = train
        self.radius = radius

    def kl(self, mu, log_var):
        tmp = -0.5 * (log_var - tf.math.exp(log_var) - tf.math.pow(mu, 2) + 1)
        return tf.reshape(tmp, [-1])

    def embed(self, n_id):
        batch_size = tf.shape(n_id)[0]
        n_id = tf.reshape(n_id, [-1])
        mu_emb = self.gnn(n_id)
        # log_var = self.gnn(n_id)
        log_var = self.log_var_encoder(n_id)
        emb = mu_emb + self.radius * tf.random.normal(tf.shape(log_var)) *  \
              tf.math.sqrt(tf.math.exp(log_var))
        emb = tf.reshape(emb, [batch_size, -1, self.dim])
        if self.train:
            return mu_emb, log_var, emb
        else:
            return mu_emb, log_var,  \
                   tf.reshape(mu_emb, [batch_size, -1, self.dim])

    def __call__(self, inputs):
        src, pos, negs = self.to_sample(inputs)
        # [batch, 1, dim]
        mu_emb, log_var_emb, embedding = self.embed(src)
        # [batch, num_negs, dim]
        mu_emb_pos, log_var_emb_pos, embedding_pos = self.embed(pos)
        # [batch, num_negs, dim]
        mu_emb_negs, log_var_emb_negs, embedding_negs = self.embed(negs)

        # [batch, 1, num_negs]
        logits = tf.matmul(
                 embedding, embedding_pos, transpose_b=True)
        # [batch, 1, num_negs]
        neg_logits = tf.matmul(
                     embedding, embedding_negs, transpose_b=True)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(logits), logits=logits)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_logits), logits=neg_logits)
        loss = tf.reduce_mean(
               tf.concat([tf.reshape(true_xent, [-1, 1]),
               tf.reshape(negative_xent, [-1, 1])], 0))

        kl = self.kl(mu_emb, log_var_emb)
        kl = tf.concat([kl, self.kl(mu_emb_pos, log_var_emb_pos)], 0)
        kl = tf.concat([kl, self.kl(mu_emb_negs, log_var_emb_negs)], 0)

        loss = loss + tf.reduce_mean(kl)

        predict = tf.nn.sigmoid(logits)
        neg_predict = tf.nn.sigmoid(neg_logits)
        label = tf.ones_like(logits)
        neg_label = tf.zeros_like(neg_logits)
        acc = tf_euler.utils.metrics.acc_score(
                tf.concat([label, neg_label], axis=2),
                tf.concat([predict, neg_predict], axis=2))

        embedding = self.embed(inputs)

        return (embedding, loss, 'acc', acc)
