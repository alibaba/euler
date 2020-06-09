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

from tf_euler.python.dataset.multigraph_util import *
from tf_euler.python.dataset.base_dataset import DataSet

current_dir = os.path.dirname(os.path.realpath(__file__))
mutag_dir = os.path.join(current_dir, 'MUTAG')


class MUTAG(DataSet):

    def __init__(self, data_dir=mutag_dir, data_type='all', train_rate=0.9):
        super(MUTAG, self).__init__(data_dir, data_type)
        self.source_url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/' \
                          'graphkerneldatasets/MUTAG.zip'
        self.origin_files = ['MUTAG_A.txt',
                             'MUTAG_graph_indicator.txt',
                             'MUTAG_graph_labels.txt',
                             'MUTAG_node_labels.txt',
                             'MUTAG_edge_labels.txt']
        self.max_node_id = 3371
        self.partition_num = 3
        self.max_edge_id = 7442
        self.max_graph_id = 188
        self.train_node_type = -1
        self.train_edge_type = ['0']
        self.total_size = 188
        self.all_node_type = -1
        self.all_edge_type = ['0']
        self.train_rate = 0.9
        self.id_file = os.path.join(data_dir, 'mutag_test.id')
        self.sparse_fea_idx = 'f1'
        self.sparse_fea_max_id = 7
        self.num_classes = 2
        self.meta_file = os.path.join(data_dir, 'index.meta')

    def generate_index_meta(self):
        meta = {}
        meta['node'] = {}
        meta['node']['features'] = {}
        meta['node']['features']['graph_label'] = \
            "graph_label:string:uint64_t:hash_index"
        meta['edge'] = {}
        meta_json = json.dumps(meta)
        print(meta_json)
        with open(self.meta_file ,'w') as out:
            out.write(meta_json)

    def download_data(self, source_url, out_dir):
        mutag_zip_dir = os.path.join(out_dir, 'MUTAG.zip')
        out_dir = os.path.join(out_dir, '..')
        DataSet.download_file(source_url, mutag_zip_dir)
        with zipfile.ZipFile(mutag_zip_dir) as mutag_zip:
            print('unzip data..')
            mutag_zip.extractall(out_dir)

    def convert2json(self, convert_dir, out_dir):
        prefix = os.path.join(self.data_dir, 'MUTAG_')
        adj_list = read_adj(prefix + 'A.txt')
        graph_indicator = read_graph_indicator(prefix + 'graph_indicator.txt')
        graph_label = read_graph_label(prefix + 'graph_labels.txt')
        node_label = read_node_label(prefix + 'node_labels.txt')
        self.generate_index_meta()
        with open(out_dir, 'w') as out:
            out.write(gen_graph_json(adj_list, graph_indicator,
                                     graph_label, node_label))
        with open(self.id_file, 'w') as id_out:
            start_idx = int(self.total_size * self.train_rate)
            for i in range(start_idx, self.total_size):
                id_out.write(str(i) + '\n')
