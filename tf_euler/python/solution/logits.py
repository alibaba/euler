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
import tf_euler

class DenseLogits(object):
    def __init__(self, logits_dim):
        self.out_fc = tf.layers.Dense(logits_dim, use_bias=False)

    def __call__(self, inputs, **kwargs):
        return self.out_fc(inputs)

class PosNegLogits(object):
    def __call__(self, emb, pos_emb, neg_emb):
        logit = tf.matmul(emb, pos_emb, transpose_b=True)
        neg_logit = tf.matmul(emb, neg_emb, transpose_b=True)
        return logit, neg_logit

class CosineLogits(object):
    def __call__(self, target_emb, context_emb):
        normalized_x = tf.nn.l2_normalize(target_emb, axis=-1)
        normalized_y = tf.nn.l2_normalize(context_emb, axis=-1)
        logits = tf.reduce_sum(normalized_x * normalized_y, -1, True)
        logits = logits * 5.0
        return logits


