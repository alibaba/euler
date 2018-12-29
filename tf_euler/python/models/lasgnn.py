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

from tf_euler.python import encoders
from tf_euler.python import layers
from tf_euler.python.models import base


class Attention(layers.Layer):
  """
  Dot-product attention.
  """

  def __init__(
      self,
      kernel_initializer=lambda: tf.uniform_unit_scaling_initializer(0.36)):
    super(Attention, self).__init__()
    self.kernel_initializer = kernel_initializer

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    self.kernel = tf.get_variable(
        'kernel',
        shape=[input_shape[-2].value, input_shape[-1].value],
        initializer=self.kernel_initializer())
    self.built = True

  def call(self, inputs):
    """Applys attention to inputs.

    Args:
      inputs: A `tf.Tensor` of shape [batch_size, ..., num_values, value_dim].

    Returns:
      A `tf.Tensor` of shape [batch_size, ..., value_dim].
    """
    similarity = tf.reduce_sum(inputs * self.kernel, axis=-1)
    coefficient = tf.nn.softmax(similarity, axis=-1)
    output = inputs * tf.expand_dims(coefficient, -1)
    return tf.reduce_sum(output, -2)


def _collapse_last_two_dim(x):
  batch_dims = tf.shape(x)[:-2]
  new_shape = tf.concat([batch_dims, [x.shape[-2].value * x.shape[-1].value]],
                        0)
  return tf.reshape(x, new_shape)


def _cosine(x, y):
  normalized_x = tf.nn.l2_normalize(x, axis=-1)
  normalized_y = tf.nn.l2_normalize(y, axis=-1)
  return tf.reduce_sum(normalized_x * normalized_y, -1, True)


class LasGNN(base.Model):
  """
  Simplified version of the model present in the papaer "Toward Label &
  Structure Learning using Graph Neural Networks for Semi-supervised Ad
  Retrieval."
  """

  def __init__(self,
               metapaths_of_groups,
               fanouts,
               dim,
               feature_ixs,
               feature_dims,
               aggregator='mean',
               concat=False,
               share_aggregator=False,
               *args,
               **kwargs):
    super(LasGNN, self).__init__(*args, **kwargs)
    shared_embeddings = encoders.SparseSageEncoder.create_sparse_embeddings(
        feature_dims)
    if share_aggregator:
      shared_aggregators = encoders.SparseSageEncoder.create_aggregators(
          dim, len(fanouts), aggregator, concat=concat)
    else:
      shared_aggregators = None
    self._sparse_sage_encoders = [[
        encoders.SparseSageEncoder(
            metapath,
            fanouts,
            dim,
            feature_ixs,
            feature_dims,
            shared_embeddings=shared_embeddings,
            aggregator=aggregator,
            concat=concat,
            shared_aggregators=shared_aggregators)
        for metapath in metapaths_of_group
    ] for metapaths_of_group in metapaths_of_groups]
    self._attention_of_group = [Attention() for _ in metapaths_of_group]
    self._target_feed_forward = layers.Dense(dim)
    self._context_feed_forward = layers.Dense(dim)

  def call(self, inputs):
    """Applys LasGNN to samples.

    Args:
      inputs: A list of `Tensor`.
        inputs[0]: label, 2-D `Tensor` of `float`, [batch_size, 1]
        inputs[1]: target node, 2-D `Tensor` of `int64`, [batch_size, 1]
        inputs[2:]: context node groups, 2-D `Tensor`s of `int64`,
          [batch_size, num_nodes_of_group]
    """
    label = inputs[0]
    node_groups = inputs[1:]

    node_groups_embedding = [
        tf.stack([metapath_sage(node_group)
                  for metapath_sage in node_group_encoders], axis=-2)
        for node_group_encoders, node_group
        in zip(self._sparse_sage_encoders, node_groups)
    ] # yapf: disable
    node_groups_embedding = [
        _collapse_last_two_dim(attention(embedding))  # flatten nodes of group
        for attention, embedding
        in zip(self._attention_of_group, node_groups_embedding)
    ] # yapf: disable

    target_embedding_flat = node_groups_embedding[0]
    context_embedding_flat = tf.concat(node_groups_embedding[1:], axis=-1)

    target_embedding = self._target_feed_forward(target_embedding_flat)
    context_embedding = self._context_feed_forward(context_embedding_flat)

    cosine = _cosine(target_embedding, context_embedding)

    logit = cosine * 5.0
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit)
    loss = tf.reduce_mean(loss)
    _, auc = tf.metrics.auc(label, tf.nn.sigmoid(logit), num_thresholds=5000)

    return base.ModelOutput(
        embedding=target_embedding, loss=loss, metric_name='auc', metric=auc)
