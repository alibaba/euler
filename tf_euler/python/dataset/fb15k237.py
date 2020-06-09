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
import urllib
import zipfile
import tqdm

from tf_euler.python.dataset.sage_util import *
from tf_euler.python.dataset.base_dataset import DataSet

current_dir = os.path.dirname(os.path.realpath(__file__))
fb_dir = os.path.join(current_dir, 'FB15k-237')


class FB15K237(DataSet):

    def __init__(self, data_dir=fb_dir, data_type='all'):
        super(FB15K237, self).__init__(data_dir, data_type)
        self.source_url = \
            'https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip'
        self.origin_files = ['Release/test.txt',
                             'Release/train.txt',
                             'Release/valid.txt']
        self.max_node_id = 14541
        self.max_edge_id = 237
        self.train_node_type = ['train']
        self.train_edge_type = ['train']
        self.total_size = 310116
        self.all_node_type = -1
        self.all_edge_type = ['train', 'valid', 'test']
        self.edge_id_file = os.path.join(data_dir, 'fb15k-237_test.edgeid')
        self.node_id_file = os.path.join(data_dir, 'fb15k-237_test.nodeid')
        self.edge_id_idx = 'id'
        self.edge_id_dim = 1

    def download_data(self, source_url, out_dir):
        fb_tgz_dir = os.path.join(out_dir, 'fb15k-237.zip')
        urllib.urlretrieve(source_url, fb_tgz_dir)
        with zipfile.ZipFile(fb_tgz_dir) as fb_file:
            print('unzip data..')
            fb_file.extractall(out_dir)

    def convert2json(self, convert_dir, out_dir):
        def add_node(id, type, weight):
            node_buf = {}
            node_buf["id"] = id
            node_buf["type"] = type
            node_buf["weight"] = weight
            node_buf["features"] = []
            return node_buf

        def add_edge(src, dst, id, type, weight):
            edge_buf = {}
            edge_buf["src"] = src
            edge_buf["dst"] = dst
            edge_buf["type"] = type
            edge_buf["weight"] = weight
            edge_buf["features"] = []
            lab_buf = {}
            lab_buf["name"] = "id"
            lab_buf["type"] = "dense"
            lab_buf["value"] = [id]
            edge_buf["features"].append(lab_buf)
            return edge_buf

        out_test = open(self.edge_id_file, 'w')
        node_out_test = open(self.node_id_file, 'w')
        with open(out_dir, 'w') as out:
            buf = {}
            buf["nodes"] = []
            buf["edges"] = []
            entity_map = {}
            relation_map = {}
            entity_id = 0
            relation_id = 0
            for file_type in ['train', 'test', 'valid']:
                in_file = open(convert_dir + '/Release/' + file_type + '.txt', 'r')
                for line in in_file.readlines():
                    triple = line.strip().split()
                    if not triple[0] in entity_map:
                        entity_map[triple[0]] = entity_id
                        entity_id += 1
                        buf["nodes"].append(add_node(entity_map[triple[0]],
                                                     file_type, 1))
                    if not triple[2] in entity_map:
                        entity_map[triple[2]] = entity_id
                        entity_id += 1
                        buf["nodes"].append(add_node(entity_map[triple[2]],
                                                     file_type, 1))
                    if not triple[1] in relation_map:
                        relation_map[triple[1]] = relation_id
                        relation_id += 1
                    buf["edges"].append(add_edge(entity_map[triple[0]],
                                                 entity_map[triple[2]],
                                                 relation_map[triple[1]],
                                                 file_type, 1))
                    if file_type == 'test':
                        edge_line = str(entity_map[triple[0]]) + " " + \
                                  str(entity_map[triple[2]]) + " " + "1" + "\n"
                        id_line = str(entity_map[triple[0]]) + "\n"
                        out_test.write(edge_line)
                        node_out_test.write(id_line)
                in_file.close()
            out.write(json.dumps(buf))
            print("Total Entity: {}, Relation: {}, Node: {}, Edge: {}."
                  .format(len(buf["nodes"]), relation_id,
                          entity_id, len(buf["edges"])))
        out_test.close()
        node_out_test.close()
