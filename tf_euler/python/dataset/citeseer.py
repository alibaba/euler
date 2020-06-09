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
import json
import tqdm
import numpy as np
import urllib
import tarfile

from tf_euler.python.dataset.gcn_utils import *
from tf_euler.python.dataset.base_dataset import DataSet

current_dir = os.path.dirname(os.path.realpath(__file__))
citeseer_dir = os.path.join(current_dir, 'citeseer')


class citeseer(DataSet):

    def __init__(self, data_dir=citeseer_dir, data_type='all'):
        super(citeseer, self).__init__(data_dir, data_type)
        self.source_url = 'http://www.cs.umd.edu/~sen/lbc-proj/data/citeseer.tgz'
        self.origin_files = ['citeseer.cites',
                             'citeseer.content']
        self.max_node_id = 3327
        self.train_node_type = ['train']
        self.train_edge_type = ['train']
        self.total_size = 3327
        self.all_node_type = -1
        self.all_edge_type = ['train', 'train_removed']
        self.id_file = os.path.join(data_dir, 'citeseer_test.id')
        self.feature_idx = 'feature'
        self.feature_dim = 3703
        self.label_idx = 'label'
        self.label_dim = 6
        self.num_classes = 6
        self.test_start_num = 2312

    def download_data(self, source_url, out_dir):
        citeseer_tgz_dir = os.path.join(out_dir, 'citeseer.tgz')
        out_dir = os.path.join(out_dir, "..")
        DataSet.download_file(source_url, citeseer_tgz_dir)
        with tarfile.open(citeseer_tgz_dir) as citeseer_file:
            print('unzip data..')
            citeseer_file.extractall(out_dir)

    def convert2json(self, convert_dir, out_dir):
        def add_node(id, type, weight, label, feature):
            node_buf = {}
            node_buf["id"] = id
            node_buf["type"] = type
            node_buf["weight"] = weight
            node_buf["features"] = [{}, {}]
            node_buf['features'][0]['name'] = 'label'
            node_buf['features'][0]['type'] = 'dense'
            node_buf['features'][0]['value'] = label.astype(
                                               float).tolist()
            node_buf['features'][1]['name'] = 'feature'
            node_buf['features'][1]['type'] = 'dense'
            feature = feature.astype(float)
            feature /= np.sum(feature) + 1e-7
            node_buf['features'][1]['value'] = feature.tolist()
            return node_buf

        def add_edge(src, dst, type, weight):
            edge_buf = {}
            edge_buf["src"] = src
            edge_buf["dst"] = dst
            edge_buf["type"] = type
            edge_buf["weight"] = weight
            edge_buf["features"] = []
            return edge_buf

        node_ids, node_type, node_label, node_feature, edge_src, edge_dst, edge_type = \
            parse_graph_file(convert_dir, self.num_classes, 'citeseer', self.feature_dim + 2, self.test_start_num)


        with open(out_dir, 'w') as out, open(self.id_file, 'w') as out_test:
            buf = {}
            buf["nodes"] = []
            buf["edges"] = []
            valid_node = {}
            for one_node, one_type, one_label, one_feature in zip(node_ids, node_type, node_label, node_feature):
                valid_node[one_node] = 1
                buf["nodes"].append(add_node(one_node, one_type, 1, one_label, one_feature))
                if one_type == "test":
                    out_test.write(str(one_node) + "\n")
            for one_src, one_dst, one_type in zip(edge_src, edge_dst, edge_type):
                if valid_node.has_key(one_src) and valid_node.has_key(one_dst):
                    buf["edges"].append(add_edge(one_src, one_dst, one_type, 1))
            out.write(json.dumps(buf))
