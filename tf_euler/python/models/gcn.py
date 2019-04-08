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

from tf_euler.python import encoders
from tf_euler.python.models import base


class SupervisedGCN(base.SupervisedModel):

  def __init__(self, label_idx, label_dim, metapath, dim, aggregator='mean',
               feature_idx=-1, feature_dim=0, max_id=-1, use_id=False,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, use_residual=False,
               *args, **kwargs):
    super(SupervisedGCN, self).__init__(label_idx, label_dim, *args, **kwargs)
    self._encoder = encoders.GCNEncoder(
        metapath, dim, aggregator,
        feature_idx=feature_idx, feature_dim=feature_dim,
        max_id=max_id, use_id=use_id,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id,
        embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
        use_residual=use_residual)

  def encoder(self, inputs):
    return self._encoder(inputs)


class ScalableGCN(base.SupervisedModel):

  def __init__(self, label_idx, label_dim, edge_type, num_layers, dim,
               aggregator='mean', feature_idx=-1, feature_dim=0,
               max_id=-1, use_id=False,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, use_residual=False,
               store_learning_rate=0.001, store_init_maxval=0.05,
               *args, **kwargs):
    super(ScalableGCN, self).__init__(label_idx, label_dim, *args, **kwargs)
    self._encoder = encoders.ScalableGCNEncoder(
        edge_type, num_layers, dim, aggregator,
        feature_idx=feature_idx, feature_dim=feature_dim,
        max_id=max_id, use_id=use_id,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id,
        embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
        use_residual=use_residual,
        store_learning_rate=store_learning_rate,
        store_init_maxval=store_init_maxval)

  def encoder(self, inputs):
    return self._encoder(inputs)

  def call(self, inputs):
    model_output = super(ScalableGCN, self).call(inputs)
    self._loss = model_output.loss
    return model_output

  def make_session_run_hook(self):
    return _ScalableGCNHook(self._encoder, self._loss)


class _ScalableGCNHook(tf.train.SessionRunHook):

  def __init__(self, scalable_sage_encoder, loss):
    self._scalable_gcn_encoder = scalable_sage_encoder
    self._loss = loss

  def begin(self):
    self._update_store_op = self._scalable_gcn_encoder.update_store_op
    self._update_gradient_op = \
        self._scalable_gcn_encoder.get_update_gradient_op(self._loss)
    self._optimize_store_op = self._scalable_gcn_encoder.optimize_store_op

  def before_run(self, run_context):
    return tf.train.SessionRunArgs([self._update_store_op,
                                    self._update_gradient_op,
                                    self._optimize_store_op])
