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

from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python.models import base


class LINE(base.UnsupervisedModel):
  """
  Implementation of LINE model.
  """

  def __init__(self,
               node_type,
               edge_type,
               max_id,
               dim,
               order=1,
               *args,
               **kwargs):
    super(LINE, self).__init__(node_type, edge_type, max_id, *args, **kwargs)

    self.target_embedding = layers.Embedding(
        name='target_embedding', max_id=max_id + 1, dim=dim)
    if order == 1:
      self.context_embedding = self.target_embedding
    elif order == 2:
      self.context_embedding = layers.Embedding(
          name='context_embedding', max_id=max_id + 1, dim=dim)
    else:
      raise ValueError('LINE order must be 1 or 2, got {}:'.format(order))

  def target_encoder(self, inputs):
    return self.target_embedding(inputs)

  def context_encoder(self, inputs):
    return self.context_embedding(inputs)
