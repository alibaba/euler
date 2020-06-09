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
            if len(nodes) != 4:
                continue
            src_node = nodes[1].split(":")[1]
            if not node_map.has_key(src_node):
                node_map[src_node] = node_index
                node_index += 1
            dst_node = nodes[3].split(":")[1]
            if not node_map.has_key(dst_node):
                node_map[dst_node] = node_index
                node_index += 1
    return node_map

def gen_edge(cite_file, node_map, train_num):
    src = []
    dst = []
    type = []
    with open(os.path.realpath(cite_file), 'r') as f:
        for line in f:
            nodes = line.strip().split("\t")
            if len(nodes) != 4:
                continue
            src.append(node_map[nodes[1].split(":")[1]])
            dst.append(node_map[nodes[3].split(":")[1]])
            if node_map[nodes[1].split(":")[1]] > train_num or node_map[nodes[3].split(":")[1]] > train_num:
                type.append("train_removed")
            else:
                type.append("train")
    return src, dst, type

def int2onehot(label, label_num):
    out_label = np.zeros(label_num, dtype=int)
    out_label[label] = 1
    return out_label

def gen_feature_map(feature_str):
    fea_map = {}
    fea_index = 0
    for one_fea in feature_str.split('\t')[1:-1]:
        fea_map[one_fea.split(":")[-2]] = fea_index
        fea_index += 1
    return fea_map

def gen_node(content_file, node_map, total_label, fea_len, train_num):
    node_ids = []
    type = []
    label = []
    feature = []
    with open(os.path.realpath(content_file), 'r') as f:
        for line in f:
            if len(line.strip().split('\t')) == 2:
                continue
            if line.strip().startswith('cat=1,2,3'):
                feature_map = gen_feature_map(line.strip())
            else:
                features = line.strip().split('\t')
                node_ids.append(node_map[features[0]])
                label.append(int2onehot(int(features[1].split('=')[1])-1, total_label))
                one_features = np.zeros(len(feature_map), dtype=float)
                for one_fea in features[2:-1]:
                    fea_name = one_fea.split('=')[0]
                    fea_val = one_fea.split('=')[1]
                    one_features[feature_map[fea_name]] = fea_val
                feature.append(one_features)
                if (node_map[features[0]]) > train_num:
                    type.append("test")
                else:
                    type.append("train")
    return node_ids, type, label, feature

def parse_graph_file(file_dir, total_label, graph_name, fea_len, train_num):
    content_file = os.path.join(file_dir, "data", graph_name + ".NODE.paper.tab")
    cite_file = os.path.join(file_dir, "data", graph_name + ".DIRECTED.cites.tab")
    node_map = gen_node_map(cite_file)
    edge_src, edge_dst, edge_type = gen_edge(cite_file, node_map, train_num)
    node_ids, node_type, node_label, node_feature = gen_node(content_file, node_map, total_label, fea_len, train_num)
    return node_ids, node_type, node_label, node_feature, edge_src, edge_dst, edge_type
