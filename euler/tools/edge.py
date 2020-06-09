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

from euler.tools.util import *


class Edge(object):
    def __init__(self, src_id, dst_id, type, weight, gmeta):
        self.src_id = src_id
        self.dst_id = dst_id
        self.type = type
        self.weight = weight
        self.dense = []
        self.sparse = []
        self.binary = []
        self.preprocess_feature_idx(gmeta)

    def preprocess_feature_idx(self, gmeta):
        expend_list(self.dense, gmeta.edge_feature_maxnum['dense'])
        expend_list(self.sparse, gmeta.edge_feature_maxnum['sparse'])
        expend_list(self.binary, gmeta.edge_feature_maxnum['binary'])


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

    def Serialize(self):
        s = b''
        s += write_correct_data('uint64_t', self.src_id)
        s += write_correct_data('uint64_t', self.dst_id)
        s += write_correct_data('int32_t', self.type)
        s += write_correct_data('float', self.weight)
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
