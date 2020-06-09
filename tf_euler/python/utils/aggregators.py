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

from tf_euler.python.utils import layers


class GCNAggregator(layers.Layer):
    def __init__(self, dim, activation=tf.nn.relu, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)
        self.dense = layers.Dense(dim, activation=activation, use_bias=False)

    def call(self, inputs):
        self_embedding, neigh_embedding = inputs
        self_embedding = tf.expand_dims(self_embedding, 1)
        all_embedding = tf.concat([self_embedding, neigh_embedding], axis=1)
        agg_embedding = tf.reduce_mean(all_embedding, axis=1)
        return self.dense(agg_embedding)


class BaseAggregator(layers.Layer):
    def __init__(self, dim, activation=tf.nn.relu, concat=False, **kwargs):
        super(BaseAggregator, self).__init__(**kwargs)
        if concat:
            if dim % 2:
                raise ValueError('dim must be divided exactly '
                                 'by 2 if concat is True.')
            dim //= 2
        self.concat = concat
        self.self_layer = layers.Dense(dim,
                                       activation=activation,
                                       use_bias=False)
        self.neigh_layer = layers.Dense(dim,
                                        activation=activation,
                                        use_bias=False)

    def call(self, inputs):
        self_embedding, neigh_embedding = inputs

        agg_embedding = self.aggregate(neigh_embedding)
        from_self = self.self_layer(self_embedding)
        from_neighs = self.neigh_layer(agg_embedding)

        if self.concat:
            return tf.concat([from_self, from_neighs], 1)
        else:
            return tf.add(from_self, from_neighs)

    def aggregate(self, inputs):
        raise NotImplementedError()


class MeanAggregator(BaseAggregator):
    def aggregate(self, inputs):
        return tf.reduce_mean(inputs, axis=1)


class BasePoolAggregator(BaseAggregator):

    def __init__(self, dim, *args, **kwargs):
        super(BasePoolAggregator, self).__init__(dim, *args, **kwargs)
        self.layers = [layers.Dense(dim, activation=tf.nn.relu)]

    def aggregate(self, inputs):
        embedding = inputs
        for layer in self.layers:
            embedding = layer(embedding)
        return self.pool(embedding)

    def pool(self, inputs):
        raise NotImplementedError()


class MeanPoolAggregator(BasePoolAggregator):

    def __init__(self, dim, *args, **kwargs):
        super(MeanPoolAggregator, self).__init__(dim, *args, **kwargs)

    def pool(self, inputs):
        return tf.reduce_mean(inputs, axis=1)


class MaxPoolAggregator(BasePoolAggregator):

    def __init__(self, dim, *args, **kwargs):
        super(MaxPoolAggregator, self).__init__(dim, *args, **kwargs)

    def pool(self, inputs):
        return tf.reduce_max(inputs, axis=1)


aggregators = {
    'gcn': GCNAggregator,
    'mean': MeanAggregator,
    'meanpool': MeanPoolAggregator,
    'maxpool': MaxPoolAggregator
}


def get(aggregator):
    return aggregators.get(aggregator)
