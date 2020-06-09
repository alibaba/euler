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


class GraphModel(object):
    def __init__(self, label_dim):
        self.out_fc = tf.layers.Dense(label_dim, use_bias=False)

    def embed(self, n_id):
        raise NotImplementedError

    def __call__(self, inputs, label=None, graph_index=None):
        if isinstance(inputs, dict):
            label = inputs['graph_label']
            graph_index = inputs['node_graph_idx']
            inputs = inputs['node_idx']
        assert (label is not None or graph_index is not None)
        graph_index = tf.cast(graph_index, tf.int32)
        embedding = self.embed(inputs, graph_index)
        logit = self.out_fc(embedding)
        label = tf.cast(label, tf.float32)

        _, acc = tf.metrics.accuracy(
            label, tf.floor(tf.nn.sigmoid(logit) + 0.5))
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label, logits=logit)
        loss = tf.reduce_mean(loss)
        return (embedding, loss, 'accuracy', acc)
