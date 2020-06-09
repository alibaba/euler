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

from euler_estimator.python.base_estimator import BaseEstimator

class SampleEstimator(BaseEstimator):

    def __init__(self, model_class, params, run_config):
        super(SampleEstimator, self).__init__(model_class, params, run_config)

    def get_input_from_sample(self, epochs):
        def _parse_func(line):
            data = tf.string_split([line], ",")
            return data

        dataset = tf.data.TextLineDataset(self.params['sample_dir'])
        dataset = dataset.map(_parse_func)
        dataset = dataset.batch(self.params['batch_size']).repeat(epochs)
        source = dataset.make_one_shot_iterator().get_next()
        return source

    def train_input_fn(self):
        return self.get_input_from_sample(self.params['epoch'])

    def evaluate_input_fn(self):
        return self.get_input_from_sample(1)

    def infer_input_fn(self):
        return self.get_input_from_sample(1)

    def transfer_embedding(self, source, emb):
        target_node = tf.string_to_number(source.values[1], out_type=tf.int64)
        target_node = tf.reshape(target_node, [-1, 1])
        return target_node, emb
