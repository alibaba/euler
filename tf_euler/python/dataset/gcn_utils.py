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

import os
import numpy as np

def gen_node_map(cite_file):
    node_map = {}
    node_index = 0
    with open(os.path.realpath(cite_file), 'r') as f:
        for line in f:
            nodes = line.strip().split("\t")
            if len(nodes) != 2:
                continue
            for one_node in nodes:
                if not node_map.has_key(one_node):
                    node_map[one_node] = node_index
                    node_index += 1
    return node_map

def gen_edge(cite_file, node_map, train_num):
    src = []
    dst = []
    type = []
    with open(os.path.realpath(cite_file), 'r') as f:
        for line in f:
            nodes = line.strip().split("\t")
            if len(nodes) != 2:
                continue
            src.append(node_map[nodes[0]])
            dst.append(node_map[nodes[1]])
            if node_map[nodes[0]] > train_num or node_map[nodes[1]] > train_num:
                type.append("train_removed")
            else:
                type.append("train")
    return src, dst, type

def int2onehot(label, label_num):
    out_label = np.zeros(label_num, dtype=int)
    out_label[label] = 1
    return out_label

def gen_node(content_file, node_map, total_label, fea_len, train_num):
    label_map = {}
    label_cnt = 0
    node_ids = []
    type = []
    label = []
    feature = []
    with open(os.path.realpath(content_file), 'r') as f:
        for line in f:
            features = line.strip().split('\t')
            if len(features) != fea_len:
                continue
            node_ids.append(node_map[features[0]])
            feature.append(np.asarray(features[1:-1], dtype=int))
            if (node_map[features[0]]) > train_num:
                type.append("test")
            else:
                type.append("train")
            if not label_map.has_key(features[-1]):
                label_map[features[-1]] = label_cnt
                label_cnt += 1
            label.append(int2onehot(label_map[features[-1]], total_label))
    return node_ids, type, label, feature

def parse_graph_file(file_dir, total_label, graph_name, fea_len, train_num):
    content_file = os.path.join(file_dir, graph_name + ".content")
    cite_file = os.path.join(file_dir, graph_name + ".cites")
    node_map = gen_node_map(cite_file)
    edge_src, edge_dst, edge_type = gen_edge(cite_file, node_map, train_num)
    node_ids, node_type, node_label, node_feature = gen_node(content_file, node_map, total_label, fea_len, train_num)
    return node_ids, node_type, node_label, node_feature, edge_src, edge_dst, edge_type




