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
import numpy as np

from euler_estimator.python.base_estimator import BaseEstimator


class GraphEstimator(BaseEstimator):

    def __init__(self, model_class, params, run_config, **kv):
        super(GraphEstimator, self).__init__(model_class, params, run_config, **kv)

    def train_input_fn(self):
        return self.params['batch_size']

    def get_graph_label(self, sample_graph):
        graph_idx = sample_graph.indices[:, 1]
        graph_mask = tf.equal(graph_idx, 0)
        graph_mask = tf.cast(graph_mask, dtype=tf.int32)
        graph_mask = tf.reshape(graph_mask, [-1])
        graph_node = tf.boolean_mask(sample_graph.values, graph_mask)
        graph_label, = tf_euler.get_dense_feature(graph_node, self.params['label'], [1])
        graph_label = tf.reshape(tf.cast(graph_label, dtype=tf.int32), [-1])
        graph_label = tf.one_hot(graph_label, self.params['num_classes'])
        return graph_label

    def get_train_from_input(self, inputs, params):
        inputs = tf_euler.sample_graph_label(inputs)
        sample_graph = tf_euler.get_graph_by_label(inputs)
        node_idx = sample_graph.values
        node_graph_idx = sample_graph.indices[:, 0]
        graph_label = self.get_graph_label(sample_graph)
        graph_idx = inputs
        return {'node_idx': node_idx,
                'graph_label': graph_label,
                'node_graph_idx': node_graph_idx,
                'graph_idx': graph_idx}

    def get_input_from_id_file(self):
        dataset = tf.data.TextLineDataset(self.params['id_file'])
        dataset = dataset.batch(self.params['batch_size'])
        source = dataset.make_one_shot_iterator().get_next()
        return source

    def get_evaluate_from_input(self, inputs, params):
        sample_graph = tf_euler.get_graph_by_label(inputs)
        node_idx = sample_graph.values
        node_graph_idx = sample_graph.indices[:, 0]
        graph_label = self.get_graph_label(sample_graph)
        graph_idx = inputs
        return {'node_idx': node_idx,
                'graph_label': graph_label,
                'node_graph_idx': node_graph_idx,
                'graph_idx': graph_idx}

    def get_infer_from_input(self, inputs, params):
        return self.get_evaluate_from_input(inputs, params)

    def transfer_embedding(self, source, emb):
        return source['graph_idx'], emb

    def evaluate_input_fn(self):
        return self.get_input_from_id_file()

    def infer_input_fn(self):
        return self.get_input_from_id_file()
