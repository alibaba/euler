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

from tf_euler.python.solution.base_supervise import SuperviseSolution
from tf_euler.python.mp_utils.base_gnn import BaseGNNNet


class GNN(BaseGNNNet):

    def __init__(self, conv, flow,
                 dims, fanouts, metapath,
                 feature_idx, feature_dim,
                 add_self_loops=False):
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


def get_solution_model(conv, dataflow, dims, fanouts, metapath,
                       feature_idx, feature_dim, label_idx, label_dim):
    my_encoder = GNN(conv, dataflow, dims, fanouts, metapath,
                     feature_idx, feature_dim)

    my_label_fn = tf_euler.solution.GetLabelFromFea(label_idx, label_dim)
    my_logit_fn = tf_euler.solution.DenseLogits(label_dim)

    return SuperviseSolution(my_label_fn,
                             my_encoder,
                             my_logit_fn)

def get_unsupervise_solution_model(conv, dataflow, dims, fanouts, metapath,
                                   feature_idx, feature_dim,
                                   neg_node_types, pos_edge_types):
    target_encoder = GNN(conv, dataflow, dims, fanouts, metapath,
                         feature_idx, feature_dim)
    context_encoder = GNN(conv, dataflow, dims, fanouts, metapath,
                          feature_idx, feature_dim)

    my_neg_sample_fn = tf_euler.solution.SampleNegWithTypes(neg_node_types)
    my_pos_sample_fn = tf_euler.solution.SamplePosWithTypes(pos_edge_types)

    return UnsuperviseSolution(target_encoder, context_encoder,
                               my_pos_sample_fn,
                               my_neg_sample_fn)
