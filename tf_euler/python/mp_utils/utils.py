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

from tf_euler.python.convolution import GATConv, GCNConv, SAGEConv, TAGConv, \
                                        AGNNConv, SGCNConv, GINConv, \
                                        GraphConv, APPNPConv, GatedConv, \
                                        ARMAConv, DNAConv, RelationConv
from tf_euler.python.dataflow import SageDataFlow, GCNDataFlow, \
                                     FastGCNDataFlow, LayerwiseDataFlow, \
                                     LayerwiseEachDataFlow, WholeDataFlow, \
                                     RelationDataFlow

conv_classes = {
    'sage': SAGEConv,
    'gcn': GCNConv,
    'gat': GATConv,
    'tag': TAGConv,
    'agnn': AGNNConv,
    'sgcn': SGCNConv,
    'graphgcn': GraphConv,
    'appnp': APPNPConv,
    'arma': ARMAConv,
    'dna': DNAConv,
    'gin': GINConv,
    'gated': GatedConv,
    'relation': RelationConv
}


def get_conv_class(conv):
    return conv_classes.get(conv) if isinstance(conv, str) else conv


class WrappedGCNDataFlow(GCNDataFlow):
    def __init__(self, fanouts, metapath, add_self_loops=True, **kwargs):
        super(WrappedGCNDataFlow, self).__init__(
            metapath, add_self_loops=add_self_loops, **kwargs)


class WrappedWholeDataFlow(WholeDataFlow):
    def __init__(self, fanouts, metapath, add_self_loops=True, **kwargs):
        super(WrappedWholeDataFlow, self).__init__(
            metapath, add_self_loops=add_self_loops, **kwargs)


flow_classes = {
    'full': WrappedGCNDataFlow,
    'sage': SageDataFlow,
    'fast': FastGCNDataFlow,
    'adapt': LayerwiseDataFlow,
    'layerwise': LayerwiseEachDataFlow,
    'whole': WrappedWholeDataFlow,
    'relation': RelationDataFlow,
}


def get_flow_class(flow):
    return flow_classes.get(flow) if isinstance(flow, str) else flow
