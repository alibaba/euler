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
from tf_euler.python.convolution import GATConv
from tf_euler.python.mp_utils.base_gnn import BaseGNNNet
from tf_euler.python.mp_utils.base import SuperviseModel


class GNN(BaseGNNNet):

    def __init__(self, conv, flow,
                 dims, fanouts, metapath,
                 feature_idx, feature_dim,
                 head_num=4, concat=True,
                 improved=True, add_self_loops=False):
        self.head_num = head_num
        self.concat = concat
        self.improved = improved
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
        if self.concat:
            assert dim % self.head_num == 0
            dim = dim // self.head_num
        return [conv_class(dim, improved=self.improved)
                for _ in range(self.head_num)]

    def calculate_conv(self, conv, inputs, edge_index, size=None, **kwargs):
        out = tf.concat([single_conv(inputs, edge_index, size=size)
                        for single_conv in conv], axis=1)
        if self.concat:
            return out
        else:
            out = tf.reshape(out, [-1, self.dim, self.head_num])
            return tf.reduce_mean(out, 1)


class GAT(SuperviseModel):
    def __init__(self, dims, metapath,
                 feature_idx, feature_dim,
                 label_idx, label_dim,
                 head_num=1, concat=True,
                 improved=False):
        super(GAT, self).__init__(label_idx,
                                  label_dim)
        self.gnn = GNN('gat', 'full', dims, None, metapath,
                       feature_idx, feature_dim,
                       head_num=head_num, concat=concat, improved=improved)

    def embed(self, n_id):
        return self.gnn(n_id)
