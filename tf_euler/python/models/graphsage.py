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


class GraphSage(base.UnsupervisedModel):
  def __init__(self,
               node_type,
               edge_type,
               max_id,
               metapath,
               fanouts,
               dim,
               aggregator='mean',
               concat=False,
               feature_idx=-1,
               feature_dim=0,
               use_feature=True,
               use_id=False,
               *args,
               **kwargs):
    super(GraphSage, self).__init__(node_type, edge_type, max_id, *args,
                                    **kwargs)
    self._target_encoder = encoders.SageEncoder(
        metapath,
        fanouts,
        dim=dim,
        aggregator=aggregator,
        concat=concat,
        feature_idx=feature_idx,
        feature_dim=feature_dim,use_feature=use_feature,max_id=max_id+1,use_id=use_id)
    self._context_encoder = encoders.SageEncoder(
        metapath, fanouts, dim=dim, aggregator=aggregator, concat=concat,
        feature_idx=feature_idx, feature_dim=feature_dim,max_id=max_id+1, use_feature=use_feature,
        use_id=use_id)

  def target_encoder(self, inputs):
    return self._target_encoder(inputs)

  def context_encoder(self, inputs):
    return self._context_encoder(inputs)


class SupervisedGraphSage(base.SupervisedModel):
  def __init__(self,
               label_idx,
               label_dim,
               metapath,
               fanouts,
               dim,
               aggregator='mean',
               concat=False,
               feature_idx=-1,
               feature_dim=0,
               *args,
               **kwargs):
    super(SupervisedGraphSage, self).__init__(label_idx, label_dim, *args,
                                              **kwargs)
    self._encoder = encoders.SageEncoder(
        metapath,
        fanouts,
        dim=dim,
        aggregator=aggregator,
        concat=concat,
        feature_idx=feature_idx,
        feature_dim=feature_dim)

  def encoder(self, inputs):
    return self._encoder(inputs)


class ScalableGCN(base.SupervisedModel):
  def __init__(self, label_idx, label_dim, edge_type, fanout, num_layers, dim,
               aggregator='mean', concat=False,
               feature_idx=-1, feature_dim=0, max_id=-1,
               *args, **kwargs):
    super(ScalableGCN, self).__init__(label_idx, label_dim, *args, **kwargs)
    self._encoder = encoders.ScalableSageEncoder(
        edge_type, fanout, num_layers, dim=dim,
        aggregator=aggregator, concat=concat,
        feature_idx=feature_idx, feature_dim=feature_dim, max_id=max_id)

  def encoder(self, inputs):
    return self._encoder(inputs)

  def call(self, inputs):
    model_output = super(ScalableGCN, self).call(inputs)
    self._loss = model_output.loss
    return model_output

  def make_session_run_hook(self):
    return _ScalableGCNHook(self._encoder, self._loss)

  def get_train_op(self):
    return tf.group(
        self._encoder.update_store_op,
        self._encoder.get_update_gradient_op(self._loss),
        self._encoder.optimize_store_op)


class _ScalableGCNHook(tf.train.SessionRunHook):
  def __init__(self, scalable_sage_encoder, loss):
    self._scalable_sage_encoder = scalable_sage_encoder
    self._loss = loss

  def begin(self):
    self._update_store_op = self._scalable_sage_encoder.update_store_op
    self._update_gradient_op = \
        self._scalable_sage_encoder.get_update_gradient_op(self._loss)
    self._optimize_store_op = self._scalable_sage_encoder.optimize_store_op

  def before_run(self, run_context):
    return tf.train.SessionRunArgs([self._update_store_op,
                                    self._update_gradient_op,
                                    self._optimize_store_op])
