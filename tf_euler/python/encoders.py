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

from tf_euler.python import aggregators
from tf_euler.python import layers
from tf_euler.python import euler_ops
from tf_euler.python.utils import embedding as utils_embedding
from tf_euler.python.utils import context as utils_context


class ShallowEncoder(layers.Layer):
  """
  Basic encoder combining embedding of node id and dense feature.
  """

  def __init__(self, dim, feature_idx=-1, feature_dim=0, max_id=-1,
               use_feature=True, use_id=False, **kwargs):
    super(ShallowEncoder, self).__init__(**kwargs)
    if not use_feature and not use_id:
      raise ValueError('Either use_feature or use_id must be True.')
    self.dim = dim
    self.use_id = use_feature
    self.use_feature = use_feature
    if use_id:
      self.embedding = layers.Embedding(dim, max_id)
    if use_feature:
      self.dense = layers.Dense(self.dim)
    self.feature_idx = feature_idx
    self.feature_dim = feature_dim


  def call(self, inputs):
    embeddings = []
    if self.use_id:
      embeddings.append(self.embedding(inputs))
    if self.use_feature:
      feature = euler_ops.get_dense_feature(
          inputs, [self.feature_idx], [self.feature_dim])[0]
      embeddings.append(self.dense(feature))
    return tf.add_n(embeddings)


class SageEncoder(layers.Layer):
  """
  GraphSage style node encoder sampling multi-hop neighbors and
  performing graph convolution (https://arxiv.org/abs/1706.02216).
  """

  @staticmethod
  def create_aggregators(dim, num_layers, aggregator, **kwargs):
    new_aggregators = []
    aggregator_class = aggregators.get(aggregator)
    for layer in range(num_layers):
      activation = tf.nn.relu if layer < num_layers - 1 else None
      new_aggregators.append(
          aggregator_class(dim, activation=activation, **kwargs))
    return new_aggregators

  def __init__(self, metapath, fanouts, dim,
               aggregator='mean', concat=False, shared_aggregators=None,
               feature_idx=-1, feature_dim=0, max_id=-1,
               use_feature=True, use_id=False, **kwargs):
    super(SageEncoder, self).__init__(**kwargs)
    if len(metapath) != len(fanouts):
      raise ValueError('Len of metapath must be the same as fanouts.')
    self.metapath = metapath
    self.fanouts = fanouts
    self.num_layers = len(metapath)
    self.concat = concat

    layer0_dim = (feature_dim if use_feature else 0) + (dim if use_id else 0)
    self.dims = [layer0_dim] + [dim] * self.num_layers

    self.use_id = use_id
    self.use_feature = use_feature
    if use_id:
      self.embedding = layers.Embedding(max_id, dim)
    self.feature_idx = feature_idx
    self.feature_dim = feature_dim

    if shared_aggregators is not None:
      self.aggregators = shared_aggregators
    else:
      self.aggregators = self.create_aggregators(
          dim, self.num_layers, aggregator, concat=concat)

  def node_encoder(self, inputs):
    if self.use_id:
      id_embedding = self.embedding(inputs)
    if not self.use_feature:
      return id_embedding

    feature = euler_ops.get_dense_feature(inputs, [self.feature_idx],
                                          [self.feature_dim])[0]
    if self.use_id:
      feature = tf.concat([feature, id_embedding], 1)
    return feature

  def call(self, inputs):
    samples = euler_ops.sample_fanout(inputs, self.metapath, self.fanouts, default_node=0)[0]
    hidden = [self.node_encoder(sample) for sample in samples]
    for layer in range(self.num_layers):
      aggregator = self.aggregators[layer]
      next_hidden = []
      for hop in range(self.num_layers - layer):
        neigh_shape = [-1, self.fanouts[hop], self.dims[layer]]
        h = aggregator((hidden[hop], tf.reshape(hidden[hop + 1], neigh_shape)))
        next_hidden.append(h)
      hidden = next_hidden

    output_shape = inputs.shape.concatenate(self.dims[-1])
    output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
    return tf.reshape(hidden[0], output_shape)


class ScalableSageEncoder(SageEncoder):
  """
  GraphSage style node encoder using store to accelerate training.
  """

  def __init__(self, edge_type, fanout, num_layers, dim,
               aggregator='mean', concat=False, shared_aggregators=None,
               feature_idx=-1, feature_dim=0, max_id=-1,
               use_feature=True, use_id=False, **kwargs):
    metapath = [edge_type] * num_layers
    fanouts = [fanout] * num_layers
    super(ScalableSageEncoder, self).__init__(
        metapath, fanouts, dim, aggregator, concat, shared_aggregators,
        feature_idx, feature_dim, max_id, use_feature, use_id, **kwargs)
    self.edge_type = edge_type
    self.fanout = fanout
    self.max_id = max_id

  def build(self, input_shape):
    self.stores = [
        tf.get_variable('store_layer_{}'.format(i),
                        [self.max_id + 2, dim],
                        initializer=tf.glorot_uniform_initializer(),
                        trainable=False)
        for i, dim in enumerate(self.dims[1:-1], 1)]
    self.gradient_stores = [
        tf.get_variable('gradient_store_layer_{}'.format(i),
                        [self.max_id + 2, dim],
                        initializer=tf.zeros_initializer(),
                        trainable=False)
        for i, dim in enumerate(self.dims[1:-1], 1)]
    self.store_optimizer = tf.train.AdamOptimizer(0.002)

  def call(self, inputs, training=None):
    if training is None:
      training = utils_context.training
    if not training:
      return super(ScalableSageEncoder, self).call(inputs)

    node, neighbor = samples = euler_ops.sample_fanout(
        inputs, [self.edge_type], [self.fanout],
        default_node=self.max_id + 1)[0]
    node_embedding, neigh_embedding = [self.node_encoder(sample)
                                       for sample in samples]

    node_embeddings = []
    neigh_embeddings = []
    for layer in range(self.num_layers):
      aggregator = self.aggregators[layer]

      neigh_shape = [-1, self.fanout, self.dims[layer]]
      neigh_embedding = tf.reshape(neigh_embedding, neigh_shape)
      node_embedding = aggregator((node_embedding, neigh_embedding))
      node_embeddings.append(node_embedding)

      if layer < self.num_layers - 1:
        neigh_embedding = tf.nn.embedding_lookup(self.stores[layer], neighbor)
        neigh_embeddings.append(neigh_embedding)

    self.update_store_op = self._update_store(node, node_embeddings)
    self.get_update_gradient_op = \
        lambda loss: self._update_gradient(loss, neighbor, neigh_embeddings)
    self.optimize_store_op = self._optimize_store(node, node_embeddings)

    output_shape = tf.concat([tf.shape(inputs), [node_embedding.shape[-1]]], 0)
    return tf.reshape(node_embedding, output_shape)

  def _update_store(self, node, node_embeddings):
    update_ops = []
    for store, node_embedding in zip(self.stores, node_embeddings):
      update_ops.append(
          utils_embedding.embedding_update(store, node, node_embedding))
    return tf.group(*update_ops)

  def _update_gradient(self, loss, neighbor, neigh_embeddings):
    update_ops = []
    for gradient_store, neigh_embedding in zip(self.gradient_stores,
                                               neigh_embeddings):
      embedding_gradient = tf.gradients(loss, neigh_embedding)[0]
      update_ops.append(
          utils_embedding.embedding_add(gradient_store,
                                        neighbor, embedding_gradient))
    return tf.group(*update_ops)

  def _optimize_store(self, node, node_embeddings):
    if not self.gradient_stores:
      return tf.no_op()

    losses = []
    for gradient_store, node_embedding in zip(self.gradient_stores,
                                              node_embeddings):
      embedding_gradient = tf.nn.embedding_lookup(gradient_store, node)
      with tf.control_dependencies([embedding_gradient]):
        clear_gradient_op = \
            utils_embedding.embedding_update(gradient_store, node, 0)
      with tf.control_dependencies([clear_gradient_op]):
        losses.append(
            tf.reduce_sum(tf.multiply(node_embedding, embedding_gradient)))
    return self.store_optimizer.minimize(tf.add_n(losses))


class SparseSageEncoder(SageEncoder):
  """
  """

  @staticmethod
  def create_sparse_embeddings(feature_dims):
    sparse_embeddings = [
        layers.SparseEmbedding(feature_dim + 1, 16)
        for feature_dim in feature_dims
    ]
    return sparse_embeddings

  def __init__(self, metapath, fanouts, dim,
               feature_ixs, feature_dims, shared_embeddings=None,
               aggregator='mean', concat=False, shared_aggregators=None,
               **kwargs):
    super(SparseSageEncoder, self).__init__(
        metapath, fanouts, dim,
        aggregator=aggregator, concat=concat,
        shared_aggregators=shared_aggregators,
        use_feature=False, use_id=False)
    self.feature_ixs = feature_ixs
    self.feature_dims = feature_dims
    self.dims[0] = 16 * len(feature_ixs)

    if shared_embeddings is not None:
      self.sparse_embeddings = shared_embeddings
    else:
      self.sparse_embeddings = self.create_sparse_embeddings(feature_dims)

  def node_encoder(self, inputs):
    default_values = [feature_dim + 1 for feature_dim in self.feature_dims]
    features = euler_ops.get_sparse_feature(
        inputs, self.feature_ixs, default_values)
    embeddings = [
        sparse_embedding(feature)
        for sparse_embedding, feature in zip(self.sparse_embeddings, features)
    ]
    return tf.concat(embeddings, 1)


class AttEncoder(layers.Layer):
  """
  Attention Encoder with neighbor sampling (https://arxiv.org/abs/1710.10903)
  """

  def __init__(self,
               edge_type=0,
               feature_idx=-1,
               feature_dim=0,
               max_id=-1,
               head_num=1,
               hidden_dim=256,
               nb_num=5,
               out_dim=1,
               **kwargs):
    super(AttEncoder, self).__init__(**kwargs)
    self.feature_idx = feature_idx
    self.feature_dim = feature_dim
    self.hidden_dim = hidden_dim
    self.out_dim = out_dim
    self.head_num = head_num
    self.nb_num = nb_num
    self.edge_type = edge_type

  def att_head(self, seq, out_size, activation):
    seq_fts = tf.layers.conv1d(seq, out_size, 1, use_bias=False)

    f_1 = tf.layers.conv1d(seq_fts, 1, 1)
    f_2 = tf.layers.conv1d(seq_fts, 1, 1)
    logits = f_1 + tf.transpose(f_2, [0, 2, 1])
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
    vals = tf.matmul(coefs, seq_fts)
    print('shape', f_1.shape, f_2.shape, logits.shape, seq_fts.shape,
          vals.shape)
    ret = tf.contrib.layers.bias_add(vals)
    return ret

  def call(self, inputs):
    batch_size = inputs.shape[0]
    neighbors = euler_ops.sample_neighbor(inputs, [self.edge_type],
                                          self.nb_num)[0]
    node_feats = euler_ops.get_dense_feature(
        tf.reshape(inputs, [-1]), [self.feature_idx], [self.feature_dim])[0]
    neighbor_feats = euler_ops.get_dense_feature(
        tf.reshape(neighbors, [-1]), [self.feature_idx], [self.feature_dim])[0]
    node_feats = tf.reshape(node_feats, [batch_size, 1, self.feature_dim])
    neighbor_feats = tf.reshape(neighbor_feats,
                                [batch_size, self.nb_num, self.feature_dim])
    seq = tf.concat([node_feats, neighbor_feats], 1)  #[bz,nb+1,fdim]

    hidden = []
    for i in range(0, self.head_num):
      #hidden_val = self.att_head_v2(tf.reshape(inputs,[batch_size,1]),neighbors)
      hidden_val = self.att_head(seq, self.hidden_dim, tf.nn.elu)
      print('hidden shape', hidden_val.shape)
      hidden_val = tf.reshape(hidden_val,
                              [batch_size, self.nb_num + 1, self.hidden_dim])
      hidden.append(hidden_val)
    h_1 = tf.concat(hidden, -1)
    out = []
    for i in range(0, self.head_num):
      out_val = self.att_head(h_1, self.out_dim, tf.nn.elu)
      out_val = tf.reshape(out_val,
                           [batch_size, self.nb_num + 1, self.out_dim])
      out.append(out_val)
    out = tf.add_n(out) / self.head_num
    out = tf.reshape(out, [batch_size, self.nb_num + 1, self.out_dim])
    out = tf.slice(out, [0, 0, 0], [batch_size, 1, self.out_dim])
    print('out shape', out.shape)
    return tf.reshape(out, [batch_size, self.out_dim])
