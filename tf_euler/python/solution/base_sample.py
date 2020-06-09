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
import six

from tf_euler.python.solution.logits import CosineLogits, PosNegLogits
from tf_euler.python.solution.losses import sigmoid_loss, xent_loss
from tf_euler.python.solution.base_unsupervise import UnsuperviseSolution


class SuperviseSampleSolution(object):
    def __init__(self,
                 parse_input_fn,
                 encoder_fn,
                 parse_group_emb_fn,
                 logit_fn=CosineLogits(),
                 metric_name='auc',
                 neg_sample_fn=None,
                 loss_fn=sigmoid_loss):
        self.metric_name = metric_name
        self.metric_class = tf_euler.utils.metrics.get(metric_name)
        self.parse_input_fn = parse_input_fn
        self.parse_group_emb_fn = parse_group_emb_fn
        self.encoder = encoder_fn
        self.neg_sample_fn = neg_sample_fn
        self.logit_fn = logit_fn
        self.loss_fn = loss_fn

    def embed(self, n_id):
        return self.encoder(n_id)

    def __call__(self, inputs):
        inputs = self.parse_input_fn(inputs)
        label = inputs[0]
        if len(inputs) == 2:
            node_groups = inputs[1]
        else:
            node_groups = inputs[1:]

        if self.neg_sample_fn is not None:
            neg_node_groups = self.neg_sample_fn(node_groups)
            neg_labels = tf.zeros_like(neg_node_groups[0], dtype=tf.float32)
            label = tf.concat([label, neg_labels], axis=0)
            pos_node_groups = node_groups
            node_groups = []
            for pos_nodes, neg_nodes in zip(pos_node_groups, neg_node_groups):
                node_groups.append(tf.concat([pos_node_groups, neg_node_groups], axis=0))

        node_groups_embedding = self.embed(node_groups)
        target_embedding, context_embedding, output_embedding = self.parse_group_emb_fn(node_groups_embedding)
        logit = self.logit_fn(target_embedding, context_emb=context_embedding)
        loss = self.loss_fn(label, logit)
        metric = self.metric_class(label, logit)
        embedding = output_embedding
        return (embedding, loss, self.metric_name, metric)

class UnsuperviseSampleSolution(object):
    def __init__(self,
                 parse_input_fn,
                 target_encoder_fn,
                 context_encoder_fn,
                 pos_sample_fn,
                 neg_sample_fn,
                 logit_fn=PosNegLogits(),
                 metric_name='auc',
                 metric_fn=None,
                 loss_fn=xent_loss):
        self.parse_input_fn = parse_input_fn
        self.metric_name = metric_name
        if metric_fn is None:
            self.metric_fn = tf_euler.utils.metrics.get(metric_name)
        else:
            self.metric_fn = metric_fn
        self.target_encoder = target_encoder_fn
        self.context_encoder = context_encoder_fn
        self.pos_sample_fn = pos_sample_fn
        self.neg_sample_fn = neg_sample_fn
        self.logit_fn = logit_fn
        self.loss_fn = loss_fn

    def target_embed(self, n_id):
        emb = self.target_encoder(n_id)
        return emb

    def context_embed(self, n_id):
        emb = self.context_encoder(n_id)
        return emb

    def to_sample(self, inputs):
        inputs = self.parse_input_fn(inputs)
        neg = []
        pos = []
        if len(inputs) == 2 and inputs[1] is not None:
            neg.append(inputs[1])
        if len(inputs) == 3:
            pos.append(inputs[2])

        if self.pos_sample_fn is not None:
            src, pos_from_fn = self.pos_sample_fn(inputs[0])
            pos.append(pos_from_fn)
        else:
            src = inputs[0]
        if self.neg_sample_fn is not None:
            neg.append(self.neg_sample_fn(inputs[0]))
        pos = tf.concat(pos, axis=-1)
        neg = tf.concat(neg, axis=-1)
        return src, pos, neg

    def __call__(self, inputs):
        src, pos, negs = self.to_sample(inputs)
        embedding = self.target_embed(src)
        embedding_pos = self.context_embed(pos)
        embedding_negs = self.context_embed(negs)

        logits, neg_logits = self.logit_fn(embedding, embedding_pos, embedding_negs)
        loss = self.loss_fn(logits, neg_logits)
        metric = self.metric_fn(logits, neg_logits)
        return (embedding, loss, self.metric_name, metric)
