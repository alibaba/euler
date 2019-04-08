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
from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python.models import base


class LINE(base.UnsupervisedModel):
  """
  Implementation of LINE model.
  """

  def __init__(self, node_type, edge_type, max_id, dim, order=1,
               feature_idx=-1, feature_dim=0, use_id=True,
               sparse_feature_idx=-1, sparse_feature_max_id=-1,
               embedding_dim=16, use_hash_embedding=False, combiner='add',
               *args, **kwargs):
    super(LINE, self).__init__(node_type, edge_type, max_id, *args, **kwargs)

    if order == 1:
      order = 'first'
    if order == 2:
      order = 'second'

    self._target_encoder = encoders.ShallowEncoder(
        dim=dim, feature_idx=feature_idx, feature_dim=feature_dim,
        max_id=max_id if use_id else -1,
        sparse_feature_idx=sparse_feature_idx,
        sparse_feature_max_id=sparse_feature_max_id,
        embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
        combiner=combiner)
    if order == 'first':
      self._context_encoder = self._target_encoder
    elif order == 'second':
      self._context_encoder = encoders.ShallowEncoder(
          dim=dim, feature_idx=feature_idx, feature_dim=feature_dim,
          max_id=max_id if use_id else -1,
          sparse_feature_idx=sparse_feature_idx,
          sparse_feature_max_id=sparse_feature_max_id,
          embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
          combiner=combiner)
    else:
      raise ValueError('LINE order must be one of 1, 2, "first", or "second"'
                       'got {}:'.format(order))

  def target_encoder(self, inputs):
    return self._target_encoder(inputs)

  def context_encoder(self, inputs):
    return self._context_encoder(inputs)
