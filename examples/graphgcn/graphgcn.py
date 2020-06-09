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
from tf_euler.python.mp_utils.base_graph import GraphModel


class GNN(BaseGNNNet):

    def __init__(self, conv, flow,
                 dims, fanouts, metapath,
                 feature_idx, feature_max_id,
                 add_self_loops=True):
        super(GNN, self).__init__(conv=conv,
                                  flow=flow,
                                  dims=dims,
                                  fanouts=fanouts,
                                  metapath=metapath,
                                  add_self_loops=add_self_loops)
        if not isinstance(feature_idx, list):
            feature_idx = [feature_idx]
        self.feature_idx = feature_idx
        self.encoder = \
            tf_euler.utils.layers.SparseEmbedding(feature_max_id, dims[0])

    def to_x(self, n_id):
        x, = tf_euler.get_sparse_feature(n_id, self.feature_idx)
        x = self.encoder(x)
        return x


class GraphGCN(GraphModel):
    def __init__(self, dims, metapath, label_dim,
                 feature_idx, feature_max_id):
        super(GraphGCN, self).__init__(label_dim)
        self.gnn = GNN('graphgcn', 'full', dims, None, metapath,
                       feature_idx, feature_max_id)
        self.pool = tf_euler.graph_pool.Pooling('add')

    def embed(self, n_id, graph_index):
        node_emb = self.gnn(n_id)
        graph_emb = self.pool(node_emb, graph_index)
        return graph_emb
