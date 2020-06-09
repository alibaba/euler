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

from tf_euler.python.dataset.base_dataset import DataSet

current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'test_data')


class test_data(DataSet):

    def __init__(self, data_dir=data_dir, data_type='all'):
        super(test_data, self).__init__(data_dir, data_type)
        self.source_url = '/Euler-2.0/tools/test_data'
        self.meta_file = os.path.join(data_dir, 'meta')
        self.partition_num = 2
        self.origin_files = ['meta']
        self.max_node_id = 6
        self.train_node_type = [0, 1]
        self.train_edge_type = [0, 1]
        self.total_size = 6
        self.all_node_type = -1
        self.all_edge_type = [0, 1]
        self.id_file = ''

    def download_data(self, source_url, out_dir):
        shell_cmd = 'cp ' + self.source_url + '/meta ' + self.meta_file
        os.system(shell_cmd)

    def convert2json(self, convert_dir, out_dir):
        shell_cmd = 'cp ' + self.source_url + '/graph.json ' + out_dir
        os.system(shell_cmd)
