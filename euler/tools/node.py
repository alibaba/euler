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

import struct
from euler.tools.util import *


class Node(object):
    def __init__(self, node_id, weight, node_type, gmeta):
        self.id = node_id
        self.weight = weight
        self.type = node_type
        self.dense = []
        self.sparse = []
        self.binary = []
        self.neighbor = []
        self.in_neighbor = []
        self.preprocess_feature_idx(gmeta)

    def preprocess_feature_idx(self, gmeta):
        expend_list(self.dense, gmeta.node_feature_maxnum['dense'])
        expend_list(self.sparse, gmeta.node_feature_maxnum['sparse'])
        expend_list(self.binary, gmeta.node_feature_maxnum['binary'])

    def set_feature(self, feature_type, value, feature_idx):
        if feature_type == "dense":
            self.dense[feature_idx] = value
        elif feature_type == "sparse":
            self.sparse[feature_idx] = value
        elif feature_type == "binary":
            self.binary[feature_idx] = value
        else:
            print('error type is not support ' + feature_type)
            exit(1)

    def set_neighbor(self, dst, weight, type, type_count):
        if type_count > 0:
            expend_list(self.neighbor, type_count - 1)
        self.neighbor[type].append((dst, weight))

    def set_in_neighbor(self, src, weight, type, type_count):
        if type_count > 0:
            expend_list(self.in_neighbor, type_count - 1)
        self.in_neighbor[type].append((src, weight))

    def convert_neighbor(self, neighbor_index):
        groups_idx = []
        neighbor = []
        weight = []
        type_ids = []
        type_weights = []
        idx = 0
        sum_weight = 0
        i = 0
        for t_neighbors in neighbor_index:
            idx += len(t_neighbors)
            groups_idx.append(idx)
            type_weight = 0
            for n in t_neighbors:
                neighbor.append(n[0])
                sum_weight += n[1]
                type_weight += n[1]
                weight.append(sum_weight)

            type_ids.append(i)
            type_weights.append(type_weight)
            i += 1
        return groups_idx, neighbor, weight, type_ids, type_weights

    def Serialize(self):
        s = b''
        s += write_correct_data('uint64_t', self.id)
        s += write_correct_data('int32_t', self.type)
        s += write_correct_data('float', self.weight)

        g_idx, n, w, t_ids, t_wei = self.convert_neighbor(self.neighbor)
        s += write_list(t_ids, 'int32_t')
        s += write_list(t_wei, 'float')
        s += write_list(g_idx, 'int32_t')
        s += write_list(n, 'uint64_t')
        s += write_list(w, 'float')

        g_idx, n, w, t_ids, t_wei = self.convert_neighbor(self.in_neighbor)
        s += write_list(t_ids, 'int32_t')
        s += write_list(t_wei, 'float')
        s += write_list(g_idx, 'int32_t')
        s += write_list(n, 'uint64_t')
        s += write_list(w, 'float')

        f_idx, f = convert_feature(self.sparse)
        s += write_list(f_idx, 'int32_t')
        s += write_list(f, 'uint64_t')

        f_idx, f = convert_feature(self.dense)
        s += write_list(f_idx, 'int32_t')
        s += write_list(f, 'float')

        f_idx, f = convert_feature(self.binary)
        s += write_list(f_idx, 'int32_t')
        s += write_string(''.join(f))
        return s
