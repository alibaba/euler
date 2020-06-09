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

import json

from euler.tools.util import *

class GraphMeta(object):
    def __init__(self, name, version):
        self.name = name
        self.v = version
        self.node_count = 0
        self.edge_count = 0
        self.node_type_count = 0
        self.edge_type_count = 0
        self.partition_num = 1
        self.node_meta = {}
        self.edge_meta = {}
        self.node_feature_dim = {}
        self.edge_feature_dim = {}
        self.node_type_info = dict()
        self.edge_type_info = dict()
        self.node_feature_maxnum = {'dense':0, 'sparse':0, 'binary':0}
        self.edge_feature_maxnum = {'dense':0, 'sparse':0, 'binary':0}

    def gen_feature_maxnum(self):
        for name in self.node_meta:
            aa = name.split('_')
            self.node_feature_maxnum[aa[0]] += 1
        for name in self.edge_meta:
            aa = name.split('_')
            self.edge_feature_maxnum[aa[0]] += 1

    def debug_string(self):
        return """
name: {}
version: {}
node_count: {}
edge_count: {}
node_type_count: {}
edge_type_count: {}
partition_num: {}
node_meta: {}
edge_meta: {}
node_feature_dim: {}
edge_feature_dim: {}
node_type_info: {}
edge_type_info: {}
node_feature_maxnum: {}
edge_feature_maxnum: {}""".format(
    self.name, self.v, self.node_count, self.edge_count,
    self.node_type_count, self.edge_type_count, self.partition_num,
    json.dumps(self.node_meta), json.dumps(self.edge_meta),
    json.dumps(self.node_feature_dim), json.dumps(self.edge_feature_dim),
    json.dumps(self.node_type_info), json.dumps(self.edge_type_info),
    json.dumps(self.node_feature_maxnum),
    json.dumps(self.edge_feature_maxnum))

    def write(self):
        s = b''
        s += write_correct_data('string', self.name)
        s += write_correct_data('string', self.v)
        s += write_correct_data('uint64_t', self.node_count)
        s += write_correct_data('uint64_t', self.edge_count)
        s += write_correct_data('uint32_t', self.partition_num)
        s += write_correct_data('uint32_t', len(self.node_meta))
        for name in self.node_meta:
            s += write_correct_data('string', name)
            s += write_correct_data('featureType', self.node_meta[name][1])
            s += write_correct_data('int32_t', self.node_meta[name][2])
            s += write_correct_data('int64_t', self.node_feature_dim[name])
        s += write_correct_data('uint32_t', len(self.edge_meta))
        for name in self.edge_meta:
            s += write_correct_data('string', name)
            s += write_correct_data('featureType', self.edge_meta[name][1])
            s += write_correct_data('int32_t', self.edge_meta[name][2])
            s += write_correct_data('int64_t', self.edge_feature_dim[name])

        s += write_correct_data('uint32_t', len(self.node_type_info))
        for tname in self.node_type_info:
            s += write_correct_data('string', tname)
            s += write_correct_data('uint32_t', self.node_type_info[tname])

        s += write_correct_data('uint32_t', len(self.edge_type_info))
        for tname in self.edge_type_info:
            s += write_correct_data('string', tname)
            s += write_correct_data('uint32_t', self.edge_type_info[tname])

        return s

    def read(self, s):
        self.name, s = read_data('string', s)
        self.v, s = read_data('string', s)
        self.node_count, s = read_data('uint64_t', s)
        self.edge_count, s = read_data('uint64_t', s)
        self.partition_num, s = read_data('uint32_t', s)
        node_meta_count, s = read_data('uint32_t', s)
        for i in range(node_meta_count):
            name, s = read_data('string', s)
            t, s = read_data('featureType', s)
            idx, s = read_data('int32_t', s)
            feature_dim, s = read_data('int64_t', s)
            self.node_meta[name] = (name, t, idx)
            self.node_feature_dim[name] = feature_dim

        edge_meta_count, s = read_data('uint32_t', s)
        for i in range(edge_meta_count):
            name, s = read_data('string', s)
            t, s = read_data('featureType', s)
            idx, s = read_data('int32_t', s)
            feature_dim, s = read_data('int64_t', s)
            self.edge_meta[name] = (name, t, idx)
            self.edge_feature_dim[name] = feature_dim

        self.node_type_count, s = read_data('uint32_t', s)
        for i in range(self.node_type_count):
            tname, s = read_data('string', s)
            index, s = read_data('uint32_t', s)
            self.node_type_info[tname] = index

        self.edge_type_count, s = read_data('uint32_t', s)
        for i in range(self.edge_type_count):
            tname, s = read_data('string', s)
            index, s = read_data('uint32_t', s)
            self.edge_type_info[tname] = index

        self.gen_feature_maxnum()
