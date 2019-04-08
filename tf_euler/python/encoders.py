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
from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python import sparse_aggregators
from tf_euler.python.utils import embedding as utils_embedding
from tf_euler.python.utils import context as utils_context


class ShallowEncoder(layers.Layer):
  """
  Basic encoder combining embedding of node id and dense feature.
  """

  def __init__(self, dim=None, feature_idx=-1, feature_dim=0, max_id=-1,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, combiner='concat',
               **kwargs):
    super(ShallowEncoder, self).__init__(**kwargs)

    if combiner not in ['add', 'concat']:
      raise ValueError('combiner must be \'add\' or \'concat\'.')
    if combiner == 'add' and dim is None:
      raise ValueError('add must be used with dim provided.')

    use_feature = feature_idx != -1
    use_id = max_id != -1
    use_sparse_feature = sparse_feature_idx != -1

    if isinstance(feature_idx, int) and use_feature:
      feature_idx = [feature_idx]
    if isinstance(feature_dim, int) and use_feature:
      feature_dim = [feature_dim]
    if use_feature and len(feature_idx) != len(feature_dim):
      raise ValueError('feature_dim must be the same length as feature_idx.')

    if isinstance(sparse_feature_idx, int) and use_sparse_feature:
      sparse_feature_idx = [sparse_feature_idx]
    if isinstance(sparse_feature_max_id, int) and use_sparse_feature:
      sparse_feature_max_id = [sparse_feature_max_id]
    if use_sparse_feature and \
       len(sparse_feature_idx) != len(sparse_feature_max_id):
      raise ValueError('sparse_feature_idx must be the same length as'
                       'sparse_feature_max_id.')

    embedding_num = (1 if use_id else 0) + \
                    (len(sparse_feature_idx) if use_sparse_feature else 0)

    if combiner == 'add':
      embedding_dim = dim
    if isinstance(embedding_dim, int) and embedding_num:
      embedding_dim = [embedding_dim] * embedding_num
    if embedding_num and len(embedding_dim) != embedding_num:
      raise ValueError('length of embedding_num must be int(use_id) + '
                       'len(sparse_feature_idx)')

    if isinstance(use_hash_embedding, bool) and embedding_num:
      use_hash_embedding = [use_hash_embedding] * embedding_num
    if embedding_num and len(use_hash_embedding) != embedding_num:
      raise ValueError('length of use_hash_embedding must be int(use_id) + '
                       'len(sparse_feature_idx)')

    # model architechture
    self.dim = dim
    self.use_id = use_id
    self.use_feature = use_feature
    self.use_sparse_feature = use_sparse_feature
    self.combiner = combiner

    # feature fetching parameters
    self.feature_idx = feature_idx
    self.feature_dim = feature_dim
    self.sparse_feature_idx = sparse_feature_idx
    self.sparse_feature_max_id = sparse_feature_max_id
    self.embedding_dim = embedding_dim

    # sub-layers
    if dim:
      self.dense = layers.Dense(self.dim, use_bias=False)

    if use_id:
      embedding_class = \
          layers.HashEmbedding if use_hash_embedding[0] else layers.Embedding
      self.embedding = embedding_class(max_id + 1, embedding_dim[0])
      embedding_dim = embedding_dim[1:]
      use_hash_embedding = use_hash_embedding[1:]
    if use_sparse_feature:
      self.sparse_embeddings = []
      for max_id, dim, use_hash in zip(
          sparse_feature_max_id, embedding_dim, use_hash_embedding):
        sparse_embedding_class = \
            layers.HashSparseEmbedding if use_hash else layers.SparseEmbedding
        self.sparse_embeddings.append(
            sparse_embedding_class(max_id + 1, dim))

  @property
  def output_dim(self):
    if self.dim is not None:
      return self.dim

    output_dim = 0
    if self.use_feature:
      output_dim += sum(self.feature_dim)
    if self.use_id or self.use_sparse_feature:
      output_dim += sum(self.embedding_dim)
    return output_dim

  def call(self, inputs):
    input_shape = inputs.shape
    inputs = tf.reshape(inputs, [-1])
    embeddings = []

    if self.use_id:
      embeddings.append(self.embedding(inputs))

    if self.use_feature:
      features = euler_ops.get_dense_feature(
          inputs, self.feature_idx, self.feature_dim)
      features = tf.concat(features, -1)
      if self.combiner == 'add':
        features = self.dense(features)
      embeddings.append(features)

    if self.use_sparse_feature:
      default_values = [max_id + 1 for max_id in self.sparse_feature_max_id]
      sparse_features = euler_ops.get_sparse_feature(
          inputs, self.sparse_feature_idx, default_values=default_values)
      embeddings.extend([
        sparse_embedding(sparse_feature)
        for sparse_embedding, sparse_feature
        in zip(self.sparse_embeddings, sparse_features)
      ])

    if self.combiner == 'add':
      embedding = tf.add_n(embeddings)
    else:
      embedding = tf.concat(embeddings, -1)
      if self.dim:
        embedding = self.dense(embedding)
    output_shape = input_shape.concatenate(self.output_dim)
    output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
    return tf.reshape(embedding, output_shape)


class GCNEncoder(layers.Layer):
  """
  GCN node encoder aggregating multi-hop neighbor information.
  """

  def __init__(self, metapath, dim, aggregator='mean',
               feature_idx=-1, feature_dim=0, max_id=-1, use_id=False,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, use_residual=False,
               **kwargs):
    super(GCNEncoder, self).__init__(**kwargs)
    self.metapath = metapath
    self.num_layers = len(metapath)

    self.use_residual = use_residual
    self._node_encoder = ShallowEncoder(
        dim=dim if use_residual else None,
        feature_idx=feature_idx, feature_dim=feature_dim,
        max_id=max_id if use_id else -1,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id,
        embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
        combiner='add' if use_residual else 'concat')

    self.aggregators = []
    aggregator_class = sparse_aggregators.get(aggregator)
    for layer in range(self.num_layers):
      activation = tf.nn.relu if layer < self.num_layers - 1 else None
      self.aggregators.append(aggregator_class(dim, activation=activation))

  def node_encoder(self, inputs):
    return self._node_encoder(inputs)

  def call(self, inputs):
    nodes, adjs = euler_ops.get_multi_hop_neighbor(inputs, self.metapath)
    hidden = [self.node_encoder(node) for node in nodes]
    for layer in range(self.num_layers):
      aggregator = self.aggregators[layer]
      next_hidden = []
      for hop in range(self.num_layers - layer):
        if self.use_residual:
          h = hidden[hop] + \
              aggregator((hidden[hop], hidden[hop + 1], adjs[hop]))
        else:
          h = aggregator((hidden[hop], hidden[hop + 1], adjs[hop]))
        next_hidden.append(h)
      hidden = next_hidden

    output_shape = inputs.shape.concatenate(hidden[0].shape[-1])
    output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
    return tf.reshape(hidden[0], output_shape)


class ScalableGCNEncoder(GCNEncoder):
  def __init__(self, edge_type, num_layers, dim, aggregator='mean',
               feature_idx=-1, feature_dim=0, max_id=-1, use_id=False,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, use_residual=False,
               store_learning_rate=0.001, store_init_maxval=0.05, **kwargs):
    metapath = [edge_type] * num_layers
    super(ScalableGCNEncoder, self).__init__(
        metapath, dim, aggregator,
        feature_idx, feature_dim, max_id, use_id,
        sparse_feature_idx, sparse_feature_max_id,
        embedding_dim, use_hash_embedding, use_residual, **kwargs)
    self.dim = dim
    self.edge_type = edge_type
    self.max_id = max_id
    self.store_learning_rate = store_learning_rate
    self.store_init_maxval = store_init_maxval

  def build(self, input_shape):
    self.stores = [
        tf.get_variable('store_layer_{}'.format(i),
                        [self.max_id + 2, self.dim],
                        initializer=tf.random_uniform_initializer(
                            maxval=self.store_init_maxval, seed=1),
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES])
        for i in range(1, self.num_layers)]
    self.gradient_stores = [
        tf.get_variable('gradient_store_layer_{}'.format(i),
                        [self.max_id + 2, self.dim],
                        initializer=tf.zeros_initializer(),
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES])
        for i in range(1, self.num_layers)]
    self.store_optimizer = tf.train.AdamOptimizer(self.store_learning_rate)

  def call(self, inputs, training=None):
    if training is None:
      training = utils_context.training
    if not training:
      return super(ScalableGCNEncoder, self).call(inputs)

    (node, neighbor), (adj,) = \
        euler_ops.get_multi_hop_neighbor(inputs, [self.edge_type])
    node_embedding = self.node_encoder(node)
    neigh_embedding = self.node_encoder(neighbor)

    node_embeddings = []
    neigh_embeddings = []
    for layer in range(self.num_layers):
      aggregator = self.aggregators[layer]

      if self.use_residual:
        node_embedding += aggregator((node_embedding, neigh_embedding, adj))
      else:
        node_embedding = aggregator((node_embedding, neigh_embedding, adj))
      node_embeddings.append(node_embedding)

      if layer < self.num_layers - 1:
        neigh_embedding = tf.nn.embedding_lookup(self.stores[layer], neighbor)
        neigh_embeddings.append(neigh_embedding)

    self.update_store_op = self._update_store(node, node_embeddings)
    store_loss, self.optimize_store_op = \
        self._optimize_store(node, node_embeddings)
    self.get_update_gradient_op = lambda loss: \
        self._update_gradient(loss + store_loss, neighbor, neigh_embeddings)

    output_shape = inputs.shape.concatenate(node_embedding.shape[-1])
    output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
    return tf.reshape(node_embedding, output_shape)

  def _update_store(self, node, node_embeddings):
    update_ops = []
    for store, node_embedding in zip(self.stores, node_embeddings):
      update_ops.append(
          utils_embedding.embedding_update(store, node, node_embedding))
    return tf.group(*update_ops)

  def _update_gradient(self, loss, neighbor, neigh_embeddings):
    update_ops = []
    for gradient_store, neigh_embedding in zip(
        self.gradient_stores, neigh_embeddings):
      embedding_gradient = tf.gradients(loss, neigh_embedding)[0]
      update_ops.append(
          utils_embedding.embedding_add(gradient_store,
                                        neighbor, embedding_gradient))
    return tf.group(*update_ops)

  def _optimize_store(self, node, node_embeddings):
    if not self.gradient_stores:
      return tf.zeros([]), tf.no_op()

    losses = []
    clear_ops = []
    for gradient_store, node_embedding in zip(
        self.gradient_stores, node_embeddings):
      embedding_gradient = tf.nn.embedding_lookup(gradient_store, node)
      with tf.control_dependencies([embedding_gradient]):
        clear_ops.append(
            utils_embedding.embedding_update(gradient_store, node,
                                             tf.zeros_like(embedding_gradient)))
      losses.append(tf.reduce_sum(node_embedding * embedding_gradient))

    store_loss = tf.add_n(losses)
    with tf.control_dependencies(clear_ops):
      return store_loss, self.store_optimizer.minimize(store_loss)


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
               use_feature=None, use_id=False,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False,
               shared_node_encoder=None, use_residual=False, **kwargs):
    super(SageEncoder, self).__init__(**kwargs)
    if len(metapath) != len(fanouts):
      raise ValueError('Len of metapath must be the same as fanouts.')
    if use_feature is not None or use_id is not None:
      tf.logging.warning('use_feature is deprecated '
                         'and would not have any effect.')

    self.metapath = metapath
    self.fanouts = fanouts
    self.num_layers = len(metapath)
    self.concat = concat

    if shared_node_encoder:
      self._node_encoder = shared_node_encoder
    else:
      self._node_encoder = ShallowEncoder(
          feature_idx=feature_idx, feature_dim=feature_dim,
          max_id=max_id if use_id else -1,
          sparse_feature_idx=sparse_feature_idx,
          sparse_feature_max_id=sparse_feature_max_id,
          embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding)

    layer0_dim = self._node_encoder.output_dim
    self.dims = [layer0_dim] + [dim] * self.num_layers

    if shared_aggregators is not None:
      self.aggregators = shared_aggregators
    else:
      self.aggregators = self.create_aggregators(
          dim, self.num_layers, aggregator, concat=concat)

    self._max_id = max_id

  def node_encoder(self, inputs):
    return self._node_encoder(inputs)

  def call(self, inputs):
    samples = euler_ops.sample_fanout(
        inputs, self.metapath, self.fanouts, default_node=self._max_id + 1)[0]
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
               use_feature=True, use_id=False,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False,
               shared_node_encoder=None, use_residual=False,
               store_learning_rate=0.001, store_init_maxval=0.05, **kwargs):
    metapath = [edge_type] * num_layers
    fanouts = [fanout] * num_layers
    super(ScalableSageEncoder, self).__init__(
        metapath, fanouts, dim, aggregator, concat, shared_aggregators,
        feature_idx, feature_dim, max_id, use_feature, use_id,
        sparse_feature_idx, sparse_feature_max_id,
        embedding_dim, use_hash_embedding, shared_node_encoder, use_residual,
        **kwargs)
    self.edge_type = edge_type
    self.fanout = fanout
    self.max_id = max_id
    self.store_learning_rate = store_learning_rate
    self.store_init_maxval = store_init_maxval

  def build(self, input_shape):
    self.stores = [
        tf.get_variable('store_layer_{}'.format(i),
                        [self.max_id + 2, dim],
                        initializer=tf.random_uniform_initializer(
                            maxval=self.store_init_maxval, seed=1),
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES])
        for i, dim in enumerate(self.dims[1:-1], 1)]
    self.gradient_stores = [
        tf.get_variable('gradient_store_layer_{}'.format(i),
                        [self.max_id + 2, dim],
                        initializer=tf.zeros_initializer(),
                        trainable=False,
                        collections=[tf.GraphKeys.LOCAL_VARIABLES])
        for i, dim in enumerate(self.dims[1:-1], 1)]
    self.store_optimizer = tf.train.AdamOptimizer(self.store_learning_rate)

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
    store_loss, self.optimize_store_op = \
        self._optimize_store(node, node_embeddings)
    self.get_update_gradient_op = lambda loss: \
        self._update_gradient(loss + store_loss, neighbor, neigh_embeddings)

    output_shape = inputs.shape.concatenate(node_embedding.shape[-1])
    output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
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
      return tf.zeros([]), tf.no_op()

    losses = []
    clear_ops = []
    for gradient_store, node_embedding in zip(self.gradient_stores,
                                              node_embeddings):
      embedding_gradient = tf.nn.embedding_lookup(gradient_store, node)
      with tf.control_dependencies([embedding_gradient]):
        clear_ops.append(
            utils_embedding.embedding_update(gradient_store, node,
                                             tf.zeros_like(embedding_gradient)))
      losses.append(tf.reduce_sum(node_embedding * embedding_gradient))

    store_loss = tf.add_n(losses)
    with tf.control_dependencies(clear_ops):
      return store_loss, self.store_optimizer.minimize(store_loss)


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
