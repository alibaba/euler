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

import numpy as np
import tensorflow as tf
from operator import mul
from functools import reduce

from tf_euler.python.utils import aggregators
from tf_euler.python import euler_ops
from tf_euler.python.utils import layers
from tf_euler.python.utils import sparse_aggregators
from tf_euler.python.utils import embedding as utils_embedding


class ShallowEncoder(layers.Layer):
    """
    Basic encoder combining embedding of node id and dense feature.
    """

    def __init__(self, dim=None, feature_idx='f1', feature_dim=0, max_id=-1,
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

        if not isinstance(feature_idx, list) and use_feature:
            feature_idx = [feature_idx]
        if isinstance(feature_dim, int) and use_feature:
            feature_dim = [feature_dim]
        if use_feature and len(feature_idx) != len(feature_dim):
            raise ValueError('feature_dim must be the same length as feature'
                             '_idx.idx:%s, dim:%s' % (str(feature_idx),
                                                      str(feature_dim)))

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
            raise ValueError('length of use_hash_embedding must be int(use_id)'
                             ' + len(sparse_feature_idx)')

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
                layers.HashEmbedding if use_hash_embedding[0] \
                else layers.Embedding
            self.embedding = embedding_class(max_id + 1, embedding_dim[0])
            embedding_dim = embedding_dim[1:]
            use_hash_embedding = use_hash_embedding[1:]
        if use_sparse_feature:
            self.sparse_embeddings = []
            for max_id, dim, use_hash in zip(
                  sparse_feature_max_id, embedding_dim, use_hash_embedding):
                sparse_embedding_class = \
                    layers.HashSparseEmbedding if use_hash \
                    else layers.SparseEmbedding
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
            default_values = [max_id + 1
                              for max_id in self.sparse_feature_max_id]
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
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
        return tf.reshape(embedding, output_shape)


class GCNEncoder(layers.Layer):
    """
    GCN node encoder aggregating multi-hop neighbor information.
    """

    def __init__(self, metapath, dim, aggregator='mean',
                 feature_idx=-1, feature_dim=0, max_id=-1, use_id=False,
                 sparse_feature_idx=-1, sparse_feature_max_id=-1,
                 embedding_dim=16, use_hash_embedding=False,
                 use_residual=False, head_num=4,
                 **kwargs):
        super(GCNEncoder, self).__init__(**kwargs)
        self.metapath = metapath
        self.num_layers = len(metapath)
        if isinstance(head_num, int):
            self.head_num = [head_num] * self.num_layers
        elif isinstance(head_num, list):
            assert len(head_num) == self.num_layers
            self.head_num = head_num
        else:
            raise ValueError('head_num error: expect int or'
                             ' list, got {}'.format(str(head_num)))

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
            self.aggregators.append(aggregator_class(
                dim, activation=activation, head_num=self.head_num[layer]))

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
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
        return tf.reshape(hidden[0], output_shape)


class GenieEncoder(GCNEncoder):
    def __init__(self, metapath, dim, aggregator='attention',
                 feature_idx=-1, feature_dim=0, max_id=-1, use_id=False,
                 sparse_feature_idx=-1, sparse_feature_max_id=-1,
                 embedding_dim=16, use_hash_embedding=False,
                 use_residual=False, head_num=4,
                 **kwargs):
        super(GenieEncoder, self).__init__(metapath, dim,
                                           aggregator,
                                           feature_idx,
                                           feature_dim,
                                           max_id,
                                           use_id,
                                           sparse_feature_idx,
                                           sparse_feature_max_id,
                                           embedding_dim,
                                           use_hash_embedding,
                                           use_residual,
                                           head_num, **kwargs)
        self.dim = dim
        self.depth_fc = []
        for layer in range(self.num_layers + 1):
            self.depth_fc.append(layers.Dense(dim))

    def call(self, inputs):
        nodes, adjs = euler_ops.get_multi_hop_neighbor(inputs, self.metapath)
        hidden = [self.node_encoder(node) for node in nodes]
        h_t = [self.depth_fc[0](hidden[0])]
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
            h_t.append(self.depth_fc[layer+1](hidden[0]))

        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.dim)
        initial_state = \
            lstm_cell.zero_state(tf.shape(inputs)[0], dtype=tf.float32)
        h_t = tf.concat([tf.reshape(i, [tf.shape(i)[0], 1, self.dim])
                         for i in h_t], 1)
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, h_t,
                                       initial_state=initial_state,
                                       dtype=tf.float32)
        outputs = tf.reshape(outputs[:, 0, :], [-1, outputs.shape[2]])
        output_shape = inputs.shape.concatenate(outputs.shape[-1])
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
        return tf.reshape(outputs, output_shape)


class ScalableGCNEncoder(GCNEncoder):
    def __init__(self, edge_type, num_layers, dim, aggregator='mean',
                 feature_idx=-1, feature_dim=0, max_id=-1, use_id=False,
                 sparse_feature_idx=-1, sparse_feature_max_id=-1,
                 embedding_dim=16, use_hash_embedding=False,
                 use_residual=False,
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
                node_embedding += aggregator((node_embedding,
                                              neigh_embedding,
                                              adj))
            else:
                node_embedding = aggregator((node_embedding,
                                             neigh_embedding,
                                             adj))
            node_embeddings.append(node_embedding)

            if layer < self.num_layers - 1:
                neigh_embedding = \
                    tf.nn.embedding_lookup(self.stores[layer], neighbor)
                neigh_embeddings.append(neigh_embedding)

        self.update_store_op = self._update_store(node, node_embeddings)
        store_loss, self.optimize_store_op = \
            self._optimize_store(node, node_embeddings)
        self.get_update_gradient_op = lambda loss: \
            self._update_gradient(loss + store_loss,
                                  neighbor,
                                  neigh_embeddings)

        output_shape = inputs.shape.concatenate(node_embedding.shape[-1])
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
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
                    utils_embedding.embedding_update(
                        gradient_store, node,
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
                 use_feature=None, use_id=None,
                 sparse_feature_idx=-1, sparse_feature_max_id=-1,
                 embedding_dim=16, use_hash_embedding=False,
                 use_residual=False, shared_node_encoder=None, **kwargs):
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
        self.feature_dim = feature_dim
        self.sparse_feature_idx = sparse_feature_idx
        self.sparse_feature_max_id = sparse_feature_max_id
        self.use_hash_embedding = use_hash_embedding
        self.embedding_dim = embedding_dim

        if shared_node_encoder:
            self._node_encoder = shared_node_encoder
        else:
            self._node_encoder = ShallowEncoder(
                feature_idx=feature_idx, feature_dim=feature_dim,
                max_id=max_id if use_id else -1,
                sparse_feature_idx=sparse_feature_idx,
                sparse_feature_max_id=sparse_feature_max_id,
                embedding_dim=embedding_dim,
                use_hash_embedding=use_hash_embedding)

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
            inputs, self.metapath, self.fanouts,
            default_node=self._max_id + 1)[0]
        hidden = [self.node_encoder(sample) for sample in samples]
        for layer in range(self.num_layers):
            aggregator = self.aggregators[layer]
            next_hidden = []
            for hop in range(self.num_layers - layer):
                neigh_shape = [-1, self.fanouts[hop], self.dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1],
                                neigh_shape)))
                next_hidden.append(h)
            hidden = next_hidden
        output_shape = inputs.shape.concatenate(self.dims[-1])
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
        return tf.reshape(hidden[0], output_shape)


class ShuffleSageEncoder(SageEncoder):
    '''
    Shuffle node features before aggregating
    Used in DGI models
    '''

    def shuffle_tensors(self, inputs):
        '''
        inputs : A list of Tensors of 2-rank. [[n1,d], [n2,d], ...]
        return the shuffle res, with the same shape with inputs.
        '''
        real_batch_size = tf.shape(inputs[0])[0]
        real_dim = tf.shape(inputs[0])[1]
        shuffle_res = tf.reshape(tf.transpose(tf.random_shuffle(
            tf.transpose(tf.concat([tf.reshape(v,
                                               [real_batch_size, -1, real_dim])
                                    for v in inputs], 1),
                         perm=[1, 0, 2])), perm=[1, 0, 2]), [-1, real_dim])
        return tf.split(shuffle_res, [tf.shape(v)[0] for v in inputs], 0)

    def agg(self, inputs, samples, shuffle):
        hidden = [self.node_encoder(sample) for sample in samples]
        if shuffle:
            hidden = self.shuffle_tensors(hidden)
        for layer in range(self.num_layers):
            aggregator = self.aggregators[layer]
            next_hidden = []
            for hop in range(self.num_layers - layer):
                neigh_shape = [-1, self.fanouts[hop], self.dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1],
                                neigh_shape)))
                next_hidden.append(h)
            hidden = next_hidden
        output_shape = inputs.shape.concatenate(self.dims[-1])
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
        return tf.reshape(hidden[0], output_shape)

    def call(self, inputs):
        samples = euler_ops.sample_fanout(
            inputs, self.metapath, self.fanouts,
            default_node=self._max_id + 1)[0]
        h = self.agg(inputs, samples, False)
        h_neg = self.agg(inputs, samples, True)
        return [h, h_neg]


class SageEncoderNew(SageEncoder):
    def __init__(self, metapath, fanouts, dim,
                 aggregator='mean', concat=False, shared_aggregators=None,
                 feature_idx=-1, feature_dim=0, max_id=-1,
                 use_feature=None, use_id=None,
                 sparse_feature_idx=-1, sparse_feature_max_id=-1,
                 embedding_dim=16, use_hash_embedding=False,
                 shared_node_encoder=None, use_residual=False,
                 shared_embedding_layers=None, **kwargs):
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
        self.feature_dim = feature_dim
        self.sparse_feature_idx = sparse_feature_idx
        self.sparse_feature_max_id = sparse_feature_max_id
        self.use_hash_embedding = use_hash_embedding
        self.embedding_dim = embedding_dim

        if shared_node_encoder:
            self._node_encoder = shared_node_encoder
        else:
            self._node_encoder = ShallowEncoder(
                feature_idx=feature_idx, feature_dim=feature_dim,
                max_id=max_id if use_id else -1,
                sparse_feature_idx=sparse_feature_idx,
                sparse_feature_max_id=sparse_feature_max_id,
                embedding_dim=embedding_dim,
                use_hash_embedding=use_hash_embedding)

        layer0_dim = self._node_encoder.output_dim
        self.dims = [layer0_dim] + [dim] * self.num_layers
        if shared_aggregators is not None:
            self.aggregators = shared_aggregators
        else:
            self.aggregators = self.create_aggregators(
                dim, self.num_layers, aggregator, concat=concat)
        self._max_id = max_id
        self.sparse_embeddings = shared_embedding_layers

    def call(self, inputs):
        default_values = [feature_dim + 1
                          for feature_dim in self.sparse_feature_max_id]
        samples, _, _, _, features = euler_ops.sample_fanout_with_feature(
            inputs, self.metapath, self.fanouts,
            default_node=self._max_id + 1,
            dense_feature_names=[],
            dense_dimensions=[],
            sparse_feature_names=self.sparse_feature_idx,
            sparse_default_values=default_values)

        f_num = len(self.sparse_feature_idx)
        hidden = []
        for layer in range(self.num_layers+1):
            embeddings = [
                sparse_embedding(sparse_feature)
                for sparse_embedding, sparse_feature, s_max_id
                in zip(self.sparse_embeddings,
                       features[layer * f_num: (layer + 1) * f_num],
                       self.sparse_feature_max_id)]
            embedding = tf.concat(embeddings, -1)
            emb_shape = [-1, self.embedding_dim * f_num]
            hidden.append(tf.reshape(embedding, emb_shape))
        for layer in range(self.num_layers):
            aggregator = self.aggregators[layer]
            next_hidden = []
            for hop in range(self.num_layers - layer):
                neigh_shape = [-1, self.fanouts[hop], self.dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1],
                                neigh_shape)))
                next_hidden.append(h)
            hidden = next_hidden
        output_shape = inputs.shape.concatenate(self.dims[-1])
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
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
            embedding_dim, use_hash_embedding,
            shared_node_encoder, use_residual,
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
            for i, dim in enumerate(self.dims[1: -1], 1)]
        self.gradient_stores = [
            tf.get_variable('gradient_store_layer_{}'.format(i),
                            [self.max_id + 2, dim],
                            initializer=tf.zeros_initializer(),
                            trainable=False,
                            collections=[tf.GraphKeys.LOCAL_VARIABLES])
            for i, dim in enumerate(self.dims[1: -1], 1)]
        self.store_optimizer = tf.train.AdamOptimizer(self.store_learning_rate)

    def call(self, inputs, training=None):
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
                neigh_embedding = \
                    tf.nn.embedding_lookup(self.stores[layer], neighbor)
                neigh_embeddings.append(neigh_embedding)

        self.update_store_op = self._update_store(node, node_embeddings)
        store_loss, self.optimize_store_op = \
            self._optimize_store(node, node_embeddings)
        self.get_update_gradient_op = lambda loss: \
            self._update_gradient(loss + store_loss,
                                  neighbor,
                                  neigh_embeddings)

        output_shape = inputs.shape.concatenate(node_embedding.shape[-1])
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
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
                    utils_embedding.embedding_update(
                        gradient_store, node,
                        tf.zeros_like(embedding_gradient)))
            losses.append(tf.reduce_sum(node_embedding * embedding_gradient))

        store_loss = tf.add_n(losses)
        with tf.control_dependencies(clear_ops):
            return store_loss, self.store_optimizer.minimize(store_loss)


class LayerEncoder(SageEncoder):
    def agg(self, inputs):
        use_att = True
        use_group = False
        group_num = 5
        seq_len = tf.shape(inputs)[1]
        split_size = [seq_len // group_num] * (group_num - 1) + [-1]
        group_values = tf.split(inputs, split_size, 1)
        group_hidden = tf.concat([tf.expand_dims(tf.reduce_sum(i, 1), 1)
                                  for i in group_values], 1)
        rank = inputs.shape.ndims
        if rank == 2:
            return inputs
        if use_att:
            att_layer = layers.AttLayer(
                self.feature_dim, hidden_dim=[128], head_num=[2, 2])
            if use_group:
                return att_layer(group_hidden)
            else:
                return att_layer(inputs)
        return tf.reduce_mean(inputs, axis=1)

    def call(self, inputs):
        samples = euler_ops.sample_fanout(
            inputs, self.metapath, self.fanouts, default_node=0)[0]
        hidden = [self.node_encoder(sample) for sample in samples]
        hidden = self.layerwise_embed(hidden)
        output = self.fm(hidden)

        output_shape = inputs.shape.concatenate(self.dims[-1])
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
        return tf.reshape(output, output_shape)

    def layerwise_embed(self, hidden):
        fanouts = [self.fanouts[0]]
        for factor in self.fanouts[1:]:
            val = fanouts[-1] * factor
            fanouts += [val]

        for i in range(1, len(hidden)):
            node = hidden[i]
            agg_dim = fanouts[i-1]
            shape = [-1, agg_dim, self.feature_dim]
            hidden[i] = self.agg(tf.reshape(node, shape))
        return hidden

    def fc(self, hidden):
        fc = tf.concat(hidden, 1)
        ld = layers.Dense(self.dims[-1], activation=tf.nn.relu, use_bias=True)
        return ld(fc)

    def fm(self, hidden):
        o = len(hidden)
        for i in range(1, o):
            comb = tf.multiply(hidden[0], hidden[i])
            hidden += [comb]
        fc = tf.concat(hidden, 1)
        ld = layers.Dense(self.dims[-1], activation=tf.nn.relu, use_bias=True)
        return ld(fc)

    def att(self, hidden):
        hidden = [tf.expand_dims(h, 1) for h in hidden]
        seq = tf.concat(hidden, 1)
        print ('seq shape:', seq.shape)
        att_layer = layers.AttLayer(self.dims[-1],
                                    hidden_dim=[128],
                                    head_num=[2, 1])
        return att_layer(seq)

    def sage(self, hidden):
        for layer in reversed(range(self.num_layers)):
            aggregator = self.aggregators[layer]
            h = aggregator((hidden[layer], hidden[layer + 1]))
            hidden[layer] = h
        return hidden[0]


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
            self.sparse_embeddings = \
                self.create_sparse_embeddings(feature_dims)

    def node_encoder(self, inputs):
        default_values = [feature_dim + 1 for feature_dim in self.feature_dims]
        features = euler_ops.get_sparse_feature(
            inputs, self.feature_ixs, default_values)
        embeddings = [
            sparse_embedding(feature)
            for sparse_embedding, feature in zip(self.sparse_embeddings,
                                                 features)
        ]
        return tf.concat(embeddings, 1)


class LGCEncoder(layers.Layer):
    """
    Large-Scale Learnable Graph Convolutional Networks
    (https://arxiv.org/pdf/1808.03965.pdf)
    """
    def __init__(self,
                 edge_type=[0],
                 feature_idx=-1,
                 feature_dim=0,
                 k=3,
                 hidden_dim=128,
                 nb_num=10,
                 out_dim=64,
                 **kwargs):
        super(LGCEncoder, self).__init__(**kwargs)
        self.edge_type = edge_type
        self.feature_idx = feature_idx
        self.feature_dim = feature_dim
        self.k = k
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.nb_num = nb_num

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        neighbors = euler_ops.sample_neighbor(
            inputs, self.edge_type, self.nb_num)[0]
        node_feats = euler_ops.get_dense_feature(
            tf.reshape(inputs, [-1]),
            [self.feature_idx],
            [self.feature_dim])[0]
        neighbor_feats = euler_ops.get_dense_feature(
            tf.reshape(neighbors, [-1]),
            [self.feature_idx],
            [self.feature_dim])[0]
        node_feats = tf.reshape(node_feats, [batch_size, 1, self.feature_dim])
        neighbor_feats = tf.reshape(
            neighbor_feats, [batch_size, self.nb_num, self.feature_dim])
        nbs = tf.concat([node_feats, neighbor_feats], 1)
        topk, _ = tf.nn.top_k(tf.transpose(neighbor_feats, [0, 2, 1]),
                              k=self.k)
        topk = tf.transpose(topk, [0, 2, 1])
        topk = tf.concat([node_feats, topk], 1)
        hidden = tf.layers.conv1d(topk,
                                  self.hidden_dim,
                                  self.k // 2 + 1, use_bias=True)
        out = tf.layers.conv1d(hidden,
                               self.out_dim,
                               self.k // 2 + 1, use_bias=True)
        out = tf.slice(out, [0, 0, 0], [batch_size, 1, self.out_dim])
        return tf.reshape(out, [batch_size, self.out_dim])
