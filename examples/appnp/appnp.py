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

from tf_euler.python.mp_utils.base_gnn import BaseGNNNet
from tf_euler.python.mp_utils.base import SuperviseModel


class GNN(BaseGNNNet):

    def __init__(self, conv, flow,
                 dims, fanouts, metapath,
                 feature_idx, feature_dim,
                 K=10, alpha=0.1,
                 add_self_loops=True):
        self.K = K
        self.alpha = alpha
        super(GNN, self).__init__(conv=conv,
                                  flow=flow,
                                  dims=dims,
                                  fanouts=fanouts,
                                  metapath=metapath,
                                  add_self_loops=add_self_loops)
        if not isinstance(feature_idx, list):
            feature_idx = [feature_idx]
        if not isinstance(feature_dim, list):
            feature_dim = [feature_dim]
        self.feature_idx = feature_idx
        self.feature_dim = feature_dim

    def to_x(self, n_id):
        x, = tf_euler.get_dense_feature(n_id,
                                        self.feature_idx,
                                        self.feature_dim)
        return x

    def get_conv(self, conv_class, dim):
        return conv_class(dim, self.K, self.alpha)


class APPNP(SuperviseModel):

    def __init__(self, dims, metapath,
                 feature_idx, feature_dim,
                 label_idx, label_dim,
                 K=10, alpha=0.1,
                 metric_name='f1'):
        super(APPNP, self).__init__(label_idx,
                                    label_dim,
                                    metric_name=metric_name)
        self.gnn = GNN('appnp', 'full', dims, None, metapath,
                       feature_idx, feature_dim)

    def embed(self, n_id):
        return self.gnn(n_id)
