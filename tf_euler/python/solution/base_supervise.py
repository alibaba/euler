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

from tf_euler.python.solution.losses import sigmoid_loss


class SuperviseSolution(object):
    def __init__(self,
                 get_label_fn,
                 encoder_fn,
                 logit_fn,
                 metric_name='f1',
                 loss_fn=sigmoid_loss):
        self.get_label_fn = get_label_fn
        self.metric_name = metric_name
        self.metric_class = tf_euler.utils.metrics.get(metric_name)
        self.encoder = encoder_fn
        self.logit_fn = logit_fn
        self.loss_fn = loss_fn

    def embed(self, n_id):
        return self.encoder(n_id)

    def __call__(self, inputs):
        label = self.get_label_fn(inputs)
        embedding = self.embed(inputs)
        logit = self.logit_fn(embedding)

        metric = self.metric_class(label, tf.nn.sigmoid(logit))
        loss = self.loss_fn(label, logit)
        return (embedding, loss, self.metric_name, metric)
