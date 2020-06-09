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


class GaeEstimator(BaseEstimator):

    def __init__(self, model_class, params, run_config):
        super(GaeEstimator, self).__init__(model_class, params, run_config)

    def get_train_from_input(self, inputs, params):
        result = tf_euler.sample_node(inputs, params['train_node_type'])
        result.set_shape([self.params['batch_size']])
        return result

    def train_input_fn(self):
        return self.params['batch_size']

    def get_input_from_id_file(self):
        dataset = tf.data.TextLineDataset(self.params['id_file'])
        dataset = dataset.map(
            lambda id_str: tf.string_to_number(id_str, out_type=tf.int64))
        dataset = dataset.batch(self.params['batch_size'])
        source = dataset.make_one_shot_iterator().get_next()
        return source

    def evaluate_input_fn(self):
        return self.get_input_from_id_file()

    def infer_input_fn(self):
        return self.get_input_from_id_file()
