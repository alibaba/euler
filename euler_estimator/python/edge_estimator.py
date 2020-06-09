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
import numpy as np
import tf_euler

from euler_estimator.python.base_estimator import BaseEstimator


class EdgeEstimator(BaseEstimator):

    def __init__(self, model_class, params, run_config):
        super(EdgeEstimator, self).__init__(model_class, params, run_config)

    def get_train_from_input(self, inputs, params):
        source = tf_euler.sample_edge(inputs, params['train_edge_type'])
        source.set_shape([inputs, 3])
        return source

    def train_input_fn(self):
        return self.params['batch_size']

    def transfer_embedding(self, source, emb):
        if self.params['infer_type'] == 'node_src':
            return source[:, 0], emb[0]
        elif self.params['infer_type'] == 'edge':
            return source[:, :2], emb[1]
        elif self.params['infer_type'] == 'node_dst':
            return source[:, 1], emb[2]
        else:
            raise ValueError('infer_type must be node_src/node_dst/edge.')

    def get_input_from_id_file(self):
        def _parse_line_py(line):
            triples = line.strip().split()
            ent1 = int(triples[0])
            ent2 = int(triples[1])
            rel_type = int(triples[2])
            out = np.asarray([ent1, ent2, rel_type])
            return out

        def _parse_function(line):
            return tf.py_func(_parse_line_py, [line], tf.int64)

        dataset = tf.data.TextLineDataset(self.params['id_file'])
        dataset = dataset.map(_parse_function)
        dataset = dataset.batch(self.params['batch_size'])
        source = dataset.make_one_shot_iterator().get_next()
        return source

    def evaluate_input_fn(self):
        return self.get_input_from_id_file()

    def infer_input_fn(self):
        return self.get_input_from_id_file()
