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
import json
import glob
import os
import shutil

from euler.tools.util import *
from euler.tools.graph_meta import GraphMeta


class Generator(object):
    def __init__(self, inputs, output, partition_num):
        self.inputs = inputs
        self.output = os.path.realpath(output)
        self.node_type_info = {'sparse': 0, 'dense': 0, 'binary': 0}
        self.edge_type_info = {'sparse': 0, 'dense': 0, 'binary': 0}
        self.gmeta = GraphMeta("graph", "2.0")
        self.gmeta.partition_num = int(partition_num)

    def do(self):
        path_list = os.path.split(self.output)
        if len(path_list) > 1:
            if os.path.exists(os.path.join(*path_list[0:-1])):
                shutil.rmtree(os.path.join(*path_list[0:-1]))
            os.makedirs(os.path.join(*path_list[0:-1]))

        print ("generate meta...")
        for inp in self.inputs:
            print('process ' + inp)
            self.parse(json.loads(open(inp).read()))

        print(self.gmeta.debug_string())
        with open(self.output, 'wb') as f:
            f.write(self.gmeta.write())

    def parse(self, data):
        self.gmeta.node_count = len(data['nodes'])
        self.gmeta.edge_count = len(data['edges'])
        for node in data['nodes']:
            self.parse_node(node)
        for edge in data['edges']:
            self.parse_edge(edge)

    def parse_node(self, node_json):
        node_t = str(node_json['type'])
        if node_t not in self.gmeta.node_type_info:
            self.gmeta.node_type_info[node_t] = len(self.gmeta.node_type_info)
            self.gmeta.node_type_count += 1

        for feature in node_json['features']:
            name = feature['type'] + '_' + feature['name']
            t = feature['type']
            v = feature['value']

            if name not in self.gmeta.node_feature_dim:
                self.gmeta.node_feature_dim[name] = 0
            if t == 'sparse':
                for i in v:
                    self.gmeta.node_feature_dim[name] = \
                        max(i, self.gmeta.node_feature_dim[name])
            elif t == 'dense':
                self.gmeta.node_feature_dim[name] = len(v)

            if name not in self.gmeta.node_meta:
                self.node_type_info[t] += 1
                idx = self.node_type_info[t] - 1
                self.gmeta.node_meta[name] = (name, t, idx)

    def parse_edge(self, edge_json):
        edge_t = str(edge_json['type'])
        if edge_t not in self.gmeta.edge_type_info:
            self.gmeta.edge_type_info[edge_t] = len(self.gmeta.edge_type_info)
            self.gmeta.edge_type_count += 1

        for feature in edge_json['features']:
            name = feature['type'] + '_' + feature['name']
            t = feature['type']
            v = feature['value']

            if name not in self.gmeta.edge_feature_dim:
                self.gmeta.edge_feature_dim[name] = 0
            if t == 'sparse':
                for i in v:
                    self.gmeta.edge_feature_dim[name] = \
                        max(i, self.gmeta.edge_feature_dim[name])
            elif t == 'dense':
                self.gmeta.edge_feature_dim[name] = len(v)

            if name not in self.gmeta.edge_meta:
                self.edge_type_info[t] += 1
                idx = self.edge_type_info[t] - 1
                self.gmeta.edge_meta[name] = (name, t, idx)


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("python json2meta.py input_files, output, partnum")
        exit(1)
    c = Generator(sys.argv[1:-2], sys.argv[-2], sys.argv[-1])
    c.do()
