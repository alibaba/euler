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

from tf_euler.python.mp_utils.base import UnsuperviseModel
from tf_euler.python.euler_ops import mp_ops
from tf_euler.python.mp_utils.base_gnn import BaseGNNNet
from tf_euler.python.dataflow import RelationDataFlow
from tf_euler.python.convolution import RelationConv


class GNN(BaseGNNNet):

    def __init__(self, conv, flow,
                 dims, fanouts, metapath,
                 rel_num, node_max_id,
                 feature_idx, feature_dim,
                 embedding_dim,
                 add_self_loops=False):
        self.fea_dim = embedding_dim
        self.relation_num = rel_num 
        super(GNN, self).__init__(conv=conv,
                                  flow=flow,
                                  dims=dims,
                                  fanouts=fanouts,
                                  metapath=metapath,
                                  add_self_loops=add_self_loops)

        self._encoder = tf_euler.utils.encoders.ShallowEncoder(
            dim=embedding_dim, feature_idx=-1,
            max_id=node_max_id,
            embedding_dim=embedding_dim)
        if not isinstance(feature_idx, list):
            feature_idx = [feature_idx]
        if not isinstance(feature_dim, list):
            feature_dim = [feature_dim]
        self.feature_idx = feature_idx
        self.feature_dim = feature_dim


    def to_x(self, n_id):
        x = self._encoder(n_id)
        return x

    def to_edge(self, n_id_src, n_id_dst, e_id):
        a = tf.expand_dims(n_id_src, -1)
        b = tf.expand_dims(n_id_dst, -1)
        c = tf.expand_dims(e_id, -1)
        c = tf.cast(c, dtype=tf.int64)
        edges = tf.concat([a,b,c], axis=1)
        rel = tf_euler.get_edge_dense_feature(edges, self.feature_idx, self.feature_dim)
        return tf.cast(tf.reshape(rel,[-1]), dtype=tf.int32)

    def get_conv(self, conv_class, dim):
        conv = conv_class(self.fea_dim, dim, None, self.relation_num)
        self.fea_dim = dim
        return conv


class UnsupervisedRGCN(UnsuperviseModel):
    def __init__(self, node_type, edge_type, max_id,
                       dims, metapath, relation_num,
                       feature_idx, feature_dim,
                       embedding_dim, num_negs, metric):
        super(UnsupervisedRGCN, self).__init__(node_type, edge_type, max_id, num_negs, metric)

        self.dim = dims[-1]

        self.gnn = GNN('relation', 'relation', dims, None,
                       metapath, relation_num, max_id,
                       feature_idx, feature_dim,
                       embedding_dim)

        self._target_encoder = tf_euler.utils.encoders.ShallowEncoder(
            dim=self.dim, feature_idx=-1,
            max_id=max_id,
            embedding_dim=embedding_dim)

    def embed(self, n_id):
        shape = n_id.shape
        output_shape = shape.concatenate(self.dim)
        output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
        res = self.gnn(tf.reshape(n_id, [-1]))
        return tf.reshape(res, output_shape)

    def embed_context(self, n_id):
        return self.embed(n_id)
