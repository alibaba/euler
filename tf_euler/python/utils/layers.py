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

import collections

import tensorflow as tf

from tensorflow.python.util import nest


_LAYER_UIDS = collections.defaultdict(lambda: 0)


def get_layer_uid(layer_name=''):
    _LAYER_UIDS[layer_name] += 1
    return _LAYER_UIDS[layer_name]


class Layer(object):
    """
    Layer class modeled after Keras (http://keras.io).
    """

    def __init__(self, name=None, **kwargs):
        self.built = False

        if name is None:
            layer_name = self.__class__.__name__.lower()
            name = layer_name + '_' + str(get_layer_uid(layer_name))

        self._name = name

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        return inputs

    def __call__(self, inputs):
        input_shapes = None
        if all(hasattr(x, 'shape') for x in nest.flatten(inputs)):
            input_shapes = nest.map_structure(lambda x: x.shape, inputs)

        with tf.variable_scope(self._name):
            if not self.built:
                self.build(input_shapes)
            outputs = self.call(inputs)
            return outputs

    def compute_output_shape(self, input_shape):
        raise NotImplementedError()


class Dense(Layer):
    """
    Basic full-connected layer.
    """

    def __init__(self,
                 dim,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=lambda:
                     tf.uniform_unit_scaling_initializer(factor=0.36),
                 bias_initializer=lambda:
                     tf.constant_initializer(value=0.0002),
                 **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.dim = dim
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        self.kernel = tf.get_variable(
            'kernel',
            shape=[input_shape[-1].value, self.dim],
            initializer=self.kernel_initializer())
        if self.use_bias:
            self.bias = tf.get_variable(
                'bias',
                shape=[self.dim],
                initializer=self.bias_initializer())
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        rank = inputs.shape.ndims
        if rank > 2:
            outputs = tf.tensordot(inputs, self.kernel, [[rank - 1], [0]])
        else:
            outputs = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation:
            outputs = self.activation(outputs)
        return outputs


class Embedding(Layer):
    """
    Id to dense vector embedding.
    """

    def __init__(self,
                 max_id,
                 dim,
                 initializer=lambda:
                     tf.truncated_normal_initializer(stddev=0.1),
                 **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.max_id = max_id
        self.dim = dim
        self.initializer = initializer

    def build(self, input_shape):
        self.embeddings = tf.get_variable(
            'embeddings',
            shape=[self.max_id + 1, self.dim],
            initializer=self.initializer())
        self.built = True

    def call(self, inputs):
        shape = inputs.shape
        inputs = tf.reshape(inputs, [-1])
        output_shape = shape.concatenate(self.dim)
        output_shape = [d if d is not None else -1
                        for d in output_shape.as_list()]
        return tf.reshape(tf.nn.embedding_lookup(self.embeddings, inputs),
                          output_shape)


class SparseEmbedding(Embedding):
    """
    Sparse id to dense vector embedding.
    """
    def __init__(self,
                 max_id,
                 dim,
                 initializer=lambda: tf.truncated_normal_initializer(
                     stddev=0.0002),
                 combiner='sum',
                 **kwargs):
        super(SparseEmbedding, self).__init__(
            max_id=max_id, dim=dim, initializer=initializer, **kwargs)
        self.combiner = combiner

    def call(self, inputs):
        return tf.nn.embedding_lookup_sparse(self.embeddings, inputs, None,
                                             combiner=self.combiner)


class AttLayer(Layer):
    """
    Attention Layer: input shape should be[batch_size, seq_length, feature_dim]
    """
    def __init__(self,
                 out_dim,
                 activation=tf.nn.elu,
                 activation_out=lambda x: x,
                 hidden_dim=[],
                 head_num=[1],
                 **kwargs):
        super(AttLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.head_num = head_num
        self.activation = activation
        self.activation_out = activation_out
        if len(head_num) < 1 or len(head_num) != len(hidden_dim) + 1:
            raise ValueError('head_num must be greater than 1 and greater'
                             ' than hidden, got {},{}'.format(str(head_num),
                                                              str(hidden_dim)))

    def call(self, inputs):
        rank = inputs.shape.ndims
        if rank != 3:
            raise ValueError('inputs rank must be 3 for AttLayer'
                             ', got{}, shape:{}'.format(rank, inputs.shape))
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        if len(self.hidden_dim) > 0:
            hidden = []
            for i in range(0, self.head_num[0]):
                hidden_val = self.att_head(inputs,
                                           self.hidden_dim[0],
                                           self.activation)
                hidden_val = tf.reshape(
                    hidden_val, [batch_size, seq_len, self.hidden_dim[0]])
                hidden.append(hidden_val)
            h_1 = tf.concat(hidden, -1)
            for i in range(1, len(self.hidden_dim)):
                hidden = []
                for j in range(0, self.head_num[i]):
                    hidden_val = self.att_head(h_1,
                                               self.hidden_dim[i],
                                               self.activation)
                    hidden_val = tf.reshape(
                        hidden_val, [batch_size, seq_len, self.hidden_dim[i]])
                    hidden.append(hidden_val)
                h_1 = tf.concat(hidden, -1)
        else:
            h1 = inputs
        out = []
        for i in range(0, self.head_num[-1]):
            out_val = self.att_head(h_1, self.out_dim, self.activation_out)
            out_val = tf.reshape(out_val,
                                 [batch_size, seq_len, self.out_dim])
            out.append(out_val)
        out = tf.add_n(out) / self.head_num[-1]
        out = tf.reshape(out, [batch_size, seq_len, self.out_dim])
        out = tf.slice(out, [0, 0, 0], [batch_size, 1, self.out_dim])
        return tf.reshape(out, [batch_size, self.out_dim])

    def att_head(self, seq, out_size, activation):
        seq_fts = tf.layers.conv1d(seq, out_size, 1, use_bias=False)
        f_1 = tf.layers.conv1d(seq_fts, 1, 1, use_bias=False)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1, use_bias=False)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))
        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)
        return activation(ret)


class LSTMLayer(Layer):
    def __init__(self,
                 out_dim,
                 activation=None,
                 initializer=lambda: tf.uniform_unit_scaling_initializer(
                     factor=0.36),
                 **kwargs):
        super(LSTMLayer, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.activation = activation
        self.initializer = initializer

    def build(self, input_shape):
        self.lstm_cell = tf.nn.rnn_cell.LSTMCell(
            self.out_dim, initializer=self.initializer)
        self.initial_state = lstm_cell.zero_state(input_shape[0],
                                                  dtype=tf.float32)
        self.built = True

    def call(self, inputs):
        outputs, state = tf.nn.dynamic_rnn(
            self.lstm_cell,
            inputs,
            initial_state=self.initial_state,
            dtype=tf.float32)
        return outputs, final_state
