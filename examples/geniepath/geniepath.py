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


class GeniePath(SuperviseModel):
    def __init__(self, dim, metapath,
                 label_idx, label_dim, max_id=-1,
                 feature_idx=-1, feature_dim=0,
                 use_id=False,
                 sparse_feature_idx=-1,
                 sparse_feature_max_id=-1,
                 embedding_dim=16,
                 use_hash_embedding=False,
                 use_residual=False,
                 head_num=4, *args, **kwargs):
        super(GeniePath, self).__init__(label_idx, label_dim, *args, **kwargs)
        self._encoder = encoders.GenieEncoder(
            metapath, dim, 'attention',
            feature_idx=feature_idx, feature_dim=feature_dim,
            max_id=max_id, use_id=use_id,
            sparse_feature_idx=sparse_feature_idx,
            sparse_feature_max_id=sparse_feature_max_id,
            embedding_dim=embedding_dim, use_hash_embedding=use_hash_embedding,
            use_residual=use_residual,
            head_num=head_num)

    def embed(self, n_id):
        return self._encoder(n_id)
