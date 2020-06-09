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

from tf_euler.python.utils import encoders
from tf_euler.python.mp_utils.base import SuperviseModel


class LGCN(SuperviseModel):
    def __init__(self, dim, metapath,
                 label_idx, label_dim,
                 feature_idx=-1, feature_dim=0,
                 k=3, nb_num=10, out_dim=64,
                 *args, **kwargs):
        super(LGCN, self).__init__(label_idx, label_dim, *args, **kwargs)
        self._encoder = encoders.LGCEncoder(metapath, feature_idx, feature_dim,
                                            k, dim, nb_num, out_dim)

    def embed(self, n_id):
        return self._encoder(n_id)
