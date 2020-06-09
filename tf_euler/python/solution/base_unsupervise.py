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

from tf_euler.python.solution.logits import PosNegLogits
from tf_euler.python.solution.losses import xent_loss


class UnsuperviseSolution(object):
    def __init__(self, target_encoder_fn, context_encoder_fn,
                 pos_sample_fn, neg_sample_fn,
                 metric_name='mrr',
                 logit_fn=PosNegLogits(),
                 loss_fn=xent_loss):
        self.metric_name = metric_name
        self.metric_class = tf_euler.utils.metrics.get(metric_name)
        self.target_encoder = target_encoder_fn
        self.context_encoder = context_encoder_fn
        self.pos_sample_fn = pos_sample_fn
        self.neg_sample_fn = neg_sample_fn
        self.logit_fn = logit_fn
        self.loss_fn = loss_fn

    def target_embed(self, n_id):
        batch_size = tf.shape(n_id)[0]
        n_id = tf.reshape(n_id, [-1])
        emb = self.target_encoder(n_id)
        emb = tf.reshape(emb, [batch_size, -1, tf.shape(emb)[-1]])
        return emb

    def context_embed(self, n_id):
        batch_size = tf.shape(n_id)[0]
        n_id = tf.reshape(n_id, [-1])
        emb = self.context_encoder(n_id)
        emb = tf.reshape(emb, [batch_size, -1, tf.shape(emb)[-1]])
        return emb

    def to_sample(self, inputs):
        src = tf.expand_dims(inputs, -1)
        pos = self.pos_sample_fn(inputs)
        negs = self.neg_sample_fn(inputs)
        assert len(negs.get_shape().as_list()) == 2 and len(pos.get_shape().as_list()) == 2
        return src, pos, negs

    def __call__(self, inputs):
        src, pos, negs = self.to_sample(inputs)
        embedding = self.target_embed(src)
        embedding_pos = self.context_embed(pos)
        embedding_negs = self.context_embed(negs)

        logits, neg_logits = self.logit_fn(embedding, embedding_pos, embedding_negs)
        loss = self.loss_fn(logits, neg_logits)
        metric = self.metric_class(logits, neg_logits)
        embedding = self.target_embed(inputs)
        return (embedding, loss, self.metric_name, metric)
