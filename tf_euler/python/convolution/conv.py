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

import sys
import inspect

import tensorflow as tf
from tf_euler.python.euler_ops import mp_ops


class Conv(object):

    def __init__(self, aggr='add'):
        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

    def gather_feature(self, features, edge_index):
        convert_features = []

        for feature in features:
            convert_feature = []
            assert isinstance(feature, tuple) or isinstance(feature, list)
            assert len(feature) == 2
            if feature[1] is None:
                feature[1] = feature[0]
            for idx, tmp in enumerate(feature):
                if tmp is not None:
                    tmp = mp_ops.gather(tmp, edge_index[idx])
                convert_feature.append(tmp)
            convert_features.append(convert_feature)
        return convert_features

    def apply_edge(self, x_j):
        return x_j

    def apply_node(self, aggr_out):
        return aggr_out
