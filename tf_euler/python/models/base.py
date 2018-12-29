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

import collections

import tensorflow as tf

from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python import metrics

ModelOutput = collections.namedtuple(
    'ModelOutput', ['embedding', 'loss', 'metric_name', 'metric'])


class Model(layers.Layer):
  """
  """

  def __init__(self, **kwargs):
    super(Model, self).__init__(**kwargs)
    self.batch_size_ratio = 1


class UnsupervisedModel(Model):
  """
  Base model for unsupervised network embedding model.
  """

  def __init__(self,
               node_type,
               edge_type,
               max_id,
               num_negs=5,
               xent_loss=False,
               **kwargs):
    super(UnsupervisedModel, self).__init__(**kwargs)
    self.node_type = node_type
    self.edge_type = edge_type
    self.max_id = max_id
    self.num_negs = num_negs
    self.xent_loss = xent_loss

  def to_sample(self, inputs):
    batch_size = tf.size(inputs)
    src = tf.expand_dims(inputs, -1)
    pos = euler_ops.sample_neighbor(inputs, self.edge_type, 1,
                                    self.max_id + 1)[0]
    negs = euler_ops.sample_node(batch_size * self.num_negs, self.node_type)
    negs = tf.reshape(negs, [batch_size, self.num_negs])
    return src, pos, negs

  def target_encoder(self, inputs):
    raise NotImplementedError()

  def context_encoder(self, inputs):
    raise NotImplementedError()

  def _mrr(self, aff, aff_neg):
    aff_all = tf.concat([aff_neg, aff], axis=2)
    size = tf.shape(aff_all)[2]
    _, indices_of_ranks = tf.nn.top_k(aff_all, k=size)
    _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
    return tf.reduce_mean(tf.reciprocal(tf.to_float(ranks[:, :, -1] + 1)))

  def decoder(self, embedding, embedding_pos, embedding_negs):
    logits = tf.matmul(embedding, embedding_pos, transpose_b=True)
    neg_logits = tf.matmul(embedding, embedding_negs, transpose_b=True)
    mrr = self._mrr(logits, neg_logits)
    if self.xent_loss:
      true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.ones_like(logits), logits=logits)
      negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.zeros_like(neg_logits), logits=neg_logits)
      loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
    else:
      neg_cost = tf.reduce_logsumexp(neg_logits, axis=2)
      loss = tf.reduce_sum(logits - neg_cost)
    return loss, mrr

  def call(self, inputs):
    src, pos, negs = self.to_sample(inputs)
    embedding = self.target_encoder(src)
    embedding_pos = self.context_encoder(pos)
    embedding_negs = self.context_encoder(negs)
    loss, mrr = self.decoder(embedding, embedding_pos, embedding_negs)
    embedding = self.target_encoder(inputs)
    return ModelOutput(
        embedding=embedding, loss=loss, metric_name='mrr', metric=mrr)


class UnsupervisedModelV2(Model):
  """
  Base model for unsupervised network embedding model.
  """

  def __init__(self,
               node_type,
               edge_type,
               max_id,
               num_negs=20,
               xent_loss=False,
               **kwargs):
    super(UnsupervisedModelV2, self).__init__(**kwargs)
    self.node_type = node_type
    self.edge_type = edge_type
    self.max_id = max_id
    self.num_negs = num_negs
    self.xent_loss = xent_loss

  def sample_positives(self, inputs):
    batch_size = tf.size(inputs)
    src = tf.expand_dims(inputs, -1)
    pos = euler_ops.sample_neighbor(inputs, self.edge_type, 1,
                                    self.max_id + 1)[0]
    return src, pos

  def sample_negatives(self):
    negs = euler_ops.sample_node(self.num_negs, self.node_type)
    negs.set_shape([self.num_negs])
    return negs

  def target_encoder(self, inputs):
    raise NotImplementedError()

  def context_encoder(self, inputs):
    raise NotImplementedError()

  def _mrr(self, aff, aff_neg):
    aff_all = tf.concat([aff_neg, aff], axis=2)
    size = tf.shape(aff_all)[2]
    _, indices_of_ranks = tf.nn.top_k(aff_all, k=size)
    _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
    return tf.reduce_mean(tf.reciprocal(tf.to_float(ranks[:, :, -1] + 1)))

  def decoder(self, embedding, embedding_pos, embedding_negs):
    logits = tf.matmul(embedding, embedding_pos, transpose_b=True)
    neg_logits = tf.tensordot(embedding, embedding_negs, [[-1], [-1]])
    mrr = self._mrr(logits, neg_logits)
    if self.xent_loss:
      true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.ones_like(logits), logits=logits)
      negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.zeros_like(neg_logits), logits=neg_logits)
      loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
    else:
      neg_cost = tf.reduce_logsumexp(neg_logits, axis=2)
      loss = tf.reduce_sum(logits - neg_cost)
    loss = loss / tf.to_float(tf.size(logits))
    return loss, mrr

  def call(self, inputs):
    src, pos = self.sample_positives(inputs)
    negs = self.sample_negatives()

    embedding = self.target_encoder(src)
    embedding_pos = self.context_encoder(pos)
    embedding_negs = self.context_encoder(negs)

    loss, mrr = self.decoder(embedding, embedding_pos, embedding_negs)
    embedding = self.target_encoder(inputs)
    return ModelOutput(
        embedding=embedding, loss=loss, metric_name='mrr', metric=mrr)


class SupervisedModel(Model):
  """
  Base model for supervised network embedding model.
  """

  def __init__(self,
               label_idx,
               label_dim,
               num_classes=None,
               sigmoid_loss=False,
               **kwargs):
    super(SupervisedModel, self).__init__()
    self.label_idx = label_idx
    self.label_dim = label_dim
    if num_classes is None:
      num_classes = label_dim
    if label_dim > 1 and label_dim != num_classes:
      raise ValueError('laben_dim must match num_classes.')
    self.num_classes = num_classes
    self.sigmoid_loss = sigmoid_loss

    self.predict_layer = layers.Dense(num_classes)

  def encoder(self, inputs):
    raise NotImplementedError()

  def decoder(self, embeddings, labels):
    logits = self.predict_layer(embeddings)
    if self.sigmoid_loss:
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels, logits=logits)
      predictions = tf.nn.sigmoid(logits)
      predictions = tf.floor(predictions + 0.5)
    else:
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits)
      predictions = tf.nn.softmax(logits)
      predictions = tf.one_hot(
          tf.argmax(predictions, axis=1), self.num_classes)
    loss = tf.reduce_mean(loss)
    return predictions, loss

  def call(self, inputs):
    labels = euler_ops.get_dense_feature(inputs, [self.label_idx],
                                         [self.label_dim])[0]
    if self.label_dim == 1:
      labels = tf.one_hot(tf.to_int64(tf.squeeze(labels)), self.num_classes)

    embedding = self.encoder(inputs)
    predictions, loss = self.decoder(embedding, labels)
    f1 = metrics.f1_score(labels, predictions, name='f1')

    return ModelOutput(
        embedding=embedding, loss=loss, metric_name='f1', metric=f1)
