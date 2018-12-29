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


class GAT(base.SupervisedModel):
  def __init__(self,
               label_idx,
               label_dim,
               feature_idx=-1,
               feature_dim=0,
               max_id=-1,
               head_num=1,
               hidden_dim=128,
               nb_num=5,
               edge_type=0,
               *args,
               **kwargs):
    super(GAT, self).__init__(label_idx, label_dim, *args, **kwargs)
    self._encoder = encoders.AttEncoder(edge_type, feature_idx, feature_dim,
                                        max_id, head_num, hidden_dim, nb_num,
                                        self.num_classes)
    print('head_num', head_num)

  def encoder(self, inputs):
    return self._encoder(inputs)
