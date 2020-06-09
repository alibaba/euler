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

import errno
import json
import sys
import struct

from euler.tools.util import *
from euler.tools.node import Node
from euler.tools.edge import Edge
from euler.tools.graph_meta import GraphMeta
import os


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


class Converter(object):
    def __init__(self, input_json, graph_meta, output_dir, partition_num,
                 pref='data'):
        output_dir = os.path.realpath(output_dir)
        self.input_json = os.path.realpath(input_json)
        self.partition_num = int(partition_num)
        self.node_out = []
        self.edge_out = []
        mkdirs(os.path.join(output_dir, 'Node'))
        mkdirs(os.path.join(output_dir, 'Edge'))
        for i in range(0, partition_num):
            self.node_out.append(open(os.path.join(
                output_dir, 'Node', pref + '_%d.dat' % (i)), 'wb'))
            self.edge_out.append(open(os.path.join(
                output_dir, 'Edge', pref + '_%d.dat' % (i)), 'wb'))
        self.gmeta = GraphMeta('', '')
        self.gmeta.read(open(graph_meta, 'rb').read())
        print(self.gmeta.debug_string())
        self.nodes = {}
        self.edges = []
        self.nodes_cnt = [0] * self.partition_num
        self.edges_cnt = [0] * self.partition_num
        self.node_type_weight = []
        self.edge_type_weight = []

    def do(self):
        r = open(self.input_json)
        print ("convert data:%s..." % self.input_json)
        data = json.loads(r.read())
        self.parse(data)
        for out in self.node_out:
            out.close()
        for out in self.edge_out:
            out.close()

    def parse(self, data):
        for node in data['nodes']:
            self.parse_node(node)
        for edge in data['edges']:
            self.parse_edge(edge)
        for idx in range(0, self.partition_num):
            print("convert data for partition:%d ..." % (idx))
            print("partition: %d, node cnt:%d, edge cnt:%d" % (
                idx, self.nodes_cnt[idx], self.edges_cnt[idx]))
            for n in self.nodes:
                if int(n) % self.partition_num == idx:
                    s = self.nodes[n].Serialize()
                    self.node_out[idx].write(write_correct_data('string', s))
            for e in self.edges:
                if int(e.src_id) % self.partition_num == idx:
                    s = e.Serialize()
                    self.edge_out[idx].write(write_correct_data('string', s))

    def parse_node(self, node_json):
        node_type = self.gmeta.node_type_info[str(node_json['type'])]
        node = Node(node_json['id'], node_json['weight'], node_type, self.gmeta)
        self.nodes_cnt[int(node_json['id']) % self.partition_num] += 1

        expend_type(self.node_type_weight, self.gmeta.node_type_count)
        self.node_type_weight[node_type] += float(node_json['weight'])

        for feature in node_json['features']:
            name = feature['type'] + '_' + feature['name']
            t = feature['type']
            v = feature['value']

            feature_idx = self.gmeta.node_meta[name][2]
            feature_type = self.gmeta.node_meta[name][1]
            node.set_feature(feature_type, v, feature_idx)

        self.nodes[node_json['id']] = node

    def parse_edge(self, edge_json):
        edge_type = self.gmeta.edge_type_info[str(edge_json['type'])]
        src_id = edge_json['src']
        self.edges_cnt[int(src_id) % self.partition_num] += 1
        self.nodes[src_id].set_neighbor(edge_json['dst'],
                                        edge_json['weight'],
                                        edge_type,
                                        self.gmeta.edge_type_count)
        self.nodes[edge_json['dst']].set_in_neighbor(
            src_id,
            edge_json['weight'],
            edge_type,
            self.gmeta.edge_type_count)

        edge = Edge(edge_json['src'],
                    edge_json['dst'],
                    edge_type,
                    edge_json['weight'],
                    self.gmeta)

        expend_type(self.edge_type_weight, edge_type)
        self.edge_type_weight[edge_type] += float(edge_json['weight'])
        for feature in edge_json['features']:
            name = feature['type'] + '_' + feature['name']
            t = feature['type']
            v = feature['value']

            feature_idx = self.gmeta.edge_meta[name][2]
            feature_type = self.gmeta.edge_meta[name][1]
            edge.set_feature(feature_type, v, feature_idx)
        self.edges.append(edge)


if __name__ == '__main__':
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("python json2partdat.py input_json graph_meta "
              "output partition_num [prefix]")
        exit(-1)
    if len(sys.argv) == 5:
        c = Converter(sys.argv[1], sys.argv[2], os.path.realpath(sys.argv[3]), int(sys.argv[4]))
    if len(sys.argv) == 6:
        c = Converter(sys.argv[1], sys.argv[2], os.path.realpath(sys.argv[3]), int(sys.argv[4]),
                      sys.argv[5])
    c.do()
