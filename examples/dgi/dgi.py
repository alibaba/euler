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


class DGIBase(object):
  def __init__(self, node_type, edge_type,
               max_id, dim, num_negs=5, metric="mrr",
               **kwargs):
    self.node_type = node_type
    self.edge_type = edge_type
    self.max_id = max_id
    self.num_negs = num_negs
    self.dim = dim
    self.kernel = tf_euler.utils.layers.Dense(dim,use_bias=False)
    self.metric_class = tf_euler.utils.metrics.get(metric)
    self.metric_name = metric

  def target_encoder(self, inputs):
    raise NotImplementedError()

  def decoder(self, embedding, embedding_pos, embedding_negs):
    logits = tf.matmul(self.kernel(embedding), embedding_pos, transpose_b=True)
    neg_logits = tf.matmul(self.kernel(embedding_negs), embedding_pos, transpose_b=True)
    metric = self.metric_class(logits, neg_logits)

    true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(logits), logits=logits)
    negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(neg_logits), logits=neg_logits)
    loss = tf.reduce_mean(tf.concat([tf.reshape(true_xent, [-1, 1]),
                                    tf.reshape(negative_xent, [-1, 1])], 0))
    return loss, metric

  def __call__(self, inputs):
    src = tf.expand_dims(inputs, -1)
    embedding, embedding_negs = self.target_encoder(src)
    embedding_fusion = self.readout_func(embedding)

    loss, metric = self.decoder(embedding, embedding_fusion, embedding_negs)
    embedding = self.target_encoder(inputs)[0]
    return (embedding, loss, self.metric_name, metric)

  def readout_func(self, inputs):
    size = tf.shape(inputs)[0]
    res = tf.sigmoid(tf.reduce_mean(inputs, axis=0, keep_dims=True))
    res = tf.tile(res,[size,1,1])
    return res


class DGI(DGIBase):
  def __init__(self, node_type, edge_type, max_id,
               metapath, fanouts, dim, aggregator='mean', concat=False,
               feature_idx=-1, feature_dim=0, use_feature=None, use_id=False,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, use_residual=False,
               num_negs = 5, metric = "mrr",
               *args, **kwargs):
    super(DGI, self).__init__(node_type, edge_type, max_id,
                              dim, num_negs, metric,*args, **kwargs)

    self._target_encoder = tf_euler.python.utils.encoders.ShuffleSageEncoder(
        metapath, fanouts, dim, aggregator, concat,
        feature_idx=feature_idx, feature_dim=feature_dim,
        max_id=max_id, use_id=use_id,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id,
        embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
        use_residual=use_residual)

  def target_encoder(self, inputs):
    return self._target_encoder(inputs)
