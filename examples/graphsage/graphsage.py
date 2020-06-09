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
from tf_euler.python.mp_utils.base import SuperviseModel, UnsuperviseModel


class GNN(BaseGNNNet):

    def __init__(self, conv, flow,
                 dims, fanouts, metapath,
                 feature_idx, feature_dim,
                 add_self_loops=False,
                 max_id=-1, **kwargs):
        super(GNN, self).__init__(conv=conv,
                                  flow=flow,
                                  dims=dims,
                                  fanouts=fanouts,
                                  metapath=metapath,
                                  add_self_loops=add_self_loops,
                                  max_id=max_id,
                                  **kwargs)
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


class SupervisedGraphSage(SuperviseModel):

    def __init__(self, dims, fanouts, metapath,
                 feature_idx, feature_dim,
                 label_idx, label_dim, max_id=-1):
        super(SupervisedGraphSage, self).__init__(label_idx,
                                                  label_dim)
        self.gnn = GNN('sage', 'sage', dims, fanouts, metapath,
                       feature_idx, feature_dim, max_id=max_id)

    def embed(self, n_id):
        return self.gnn(n_id)


class UnsupervisedGraphSage(UnsuperviseModel):

    def __init__(self, dims, fanouts, metapath,
                 feature_idx, feature_dim,
                 node_type, edge_type, max_id,
                 num_negs=5):
        super(UnsupervisedGraphSage, self).__init__(node_type,
                                                    edge_type,
                                                    max_id,
                                                    num_negs=num_negs)
        self.gnn = GNN('sage', 'sage', dims, fanouts, metapath,
                       feature_idx, feature_dim, max_id=max_id)
        self.context_gnn = GNN('sage', 'sage', dims, fanouts, metapath,
                               feature_idx, feature_dim, max_id=max_id)
        self.dim = dims[-1]

    def embed(self, n_id):
        batch_size = tf.shape(n_id)[0]
        n_id = tf.reshape(n_id, [-1])
        emb = self.gnn(n_id)
        emb = tf.reshape(emb, [batch_size, -1, self.dim])
        return emb

    def embed_context(self, n_id):
        batch_size = tf.shape(n_id)[0]
        n_id = tf.reshape(n_id, [-1])
        emb = self.context_gnn(n_id)
        emb = tf.reshape(emb, [batch_size, -1, self.dim])
        return emb
