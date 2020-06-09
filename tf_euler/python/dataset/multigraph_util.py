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

import sys
import os
import json
import csv
import random as rand


def read_adj(adj):
    f = open(adj)
    lines = f.readlines()
    edge_list = []
    for line in lines:
        src_dst = line.split(',')
        pair = [int(src_dst[0].strip()) - 1, int(src_dst[1].strip()) - 1]
        edge_list.append(pair)
    f.close()
    return edge_list


def read_graph_indicator(graph_indicator):
    f = open(graph_indicator)
    lines = f.readlines()
    indicator_list = []
    for line in lines:
        one_indicator = int(line.strip()) - 1
        indicator_list.append(one_indicator)
    f.close()
    return indicator_list


def read_graph_label(graph_label):
    f = open(graph_label)
    lines = f.readlines()
    graph_labels = []
    for line in lines:
        label = max(int(line.strip()), 0)
        graph_labels.append(label)
    f.close()
    return graph_labels


def read_node_label(node_label):
    f = open(node_label)
    lines = f.readlines()
    node_labels = []
    for line in lines:
        label = int(line.strip())
        node_labels.append(label)
    f.close()
    return node_labels


def gen_graph_json(edge_list, indicator_list, graph_labels, node_labels):
    graph = {}
    graph['nodes'] = []
    graph['edges'] = []
    print("Total nodes: {}.".format(len(node_labels)))
    for i in range(len(node_labels)):
        node_id = i
        node_label = [node_labels[i]]
        node_graph_label = [graph_labels[indicator_list[i]]]
        graph_indicator = str(indicator_list[i])
        features = [{'name': 'f1',
                     'type': 'sparse',
                     'value': node_label},
                    {'name': 'label',
                     'type': 'dense',
                     'value': node_graph_label},
                    {'name': 'graph_label',
                     'type': 'binary',
                     'value': graph_indicator}]
        node = {'id': node_id,
                'type': node_labels[i],
                'weight': 1,
                'features': features}
        graph['nodes'].append(node)
    for one_edge in edge_list:
        src = one_edge[0]
        dst = one_edge[1]
        edge = {'src': src,
                'dst': dst,
                'type': 0,
                'weight': 1,
                'features': []}
        graph['edges'].append(edge)
    return json.dumps(graph)

