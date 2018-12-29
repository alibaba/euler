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


def f1_score(labels, predictions, name=None):
  """Streaming f1 score.
  """
  with tf.variable_scope(name, 'f1', (labels, predictions)):
    epsilon = 1e-7
    _, tp = tf.metrics.true_positives(labels, predictions)
    _, fn = tf.metrics.false_negatives(labels, predictions)
    _, fp = tf.metrics.false_positives(labels, predictions)
    precision = tf.div(tp, epsilon + tp + fp, name='precision')
    recall = tf.div(tp, epsilon + tp + fn, name='recall')
    f1 = 2.0 * precision * recall / (precision + recall + epsilon)
  return f1

def mrr_score(logits, negative_logits, name=None):
  """Mean reciprocal rank score.
  """
  with tf.variable_scope(name, 'mrr', (logits, negative_logits)):
    all_logits = tf.concat([negative_logits, logits], axis=2)
    size = tf.shape(all_logits)[2]
    _, indices_of_ranks = tf.nn.top_k(all_logits, k=size)
    _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
    return tf.reduce_mean(tf.reciprocal(tf.to_float(ranks[:, :, -1] + 1)))
