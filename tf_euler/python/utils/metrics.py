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


def acc_score(labels, predict, name=None):
    predictions = tf.floor(predict + 0.5)
    _, acc = tf.metrics.accuracy(labels, predictions)
    return acc


def auc_score(labels, predict, num_thresholds=5000):
    predictions = tf.nn.sigmoid(predict)
    _, auc = tf.metrics.auc(labels, predictions, num_thresholds=num_thresholds)
    return auc


def f1_score(labels, predict, name=None):
    """
    Streaming f1 score.
    """
    predictions = tf.floor(predict + 0.5)
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
    """
    Mean reciprocal rank score.
    """
    with tf.variable_scope(name, 'mrr', (logits, negative_logits)):
        all_logits = tf.concat([negative_logits, logits], axis=2)
        size = tf.shape(all_logits)[2]
        _, indices_of_ranks = tf.nn.top_k(all_logits, k=size)
        _, ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        return tf.reduce_mean(tf.reciprocal(tf.to_float(ranks[:, :, -1] + 1)))

def hitk_score(k, pos_scores, neg_scores):
    scores_all = tf.concat([neg_scores, pos_scores], axis=2)
    size = tf.shape(scores_all)[2]
    _, indices_of_ranks = tf.nn.top_k(scores_all, k=size)
    ranks = tf.argmax(indices_of_ranks, -1)
    return tf.reduce_mean(tf.cast(tf.less(ranks, k), tf.float32))

def hit1_score(pos_scores, neg_scores):
    return hitk_score(1, pos_scores, neg_scores)

def hit3_score(pos_scores, neg_scores):
    return hitk_score(3, pos_scores, neg_scores)

def hit10_score(pos_scores, neg_scores):
    return hitk_score(10, pos_scores, neg_scores)

def mr_score(pos_scores, neg_scores):
    scores_all = tf.concat([neg_scores, pos_scores], axis=2)
    size = tf.shape(scores_all)[2]
    _, indices_of_ranks = tf.nn.top_k(scores_all, k=size)
    ranks = tf.argmax(indices_of_ranks, -1)
    return tf.reduce_mean(ranks)

metrics = {
    'acc': acc_score,
    'auc': auc_score,
    'f1': f1_score,
    'mrr': mrr_score,
    'hit1': hit1_score,
    'hit3': hit3_score,
    'hit10': hit10_score,
    'mr': mr_score
}


def get(name):
    return metrics.get(name)
