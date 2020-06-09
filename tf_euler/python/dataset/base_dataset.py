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
import shutil
import sys

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlretrieve

import tf_euler
from euler.tools.generate_euler_data import EulerGenerator

current_dir = os.path.dirname(os.path.realpath(__file__))


class DataSet(object):

    def __init__(self, data_dir=current_dir, data_type='all'):
        self.data_dir = data_dir
        self.data_type = data_type
        self.source_url = 'DEFALT URL'
        self.partition_num = 10
        self.meta_file = None
        self.origin_files = []
        self.test_start_num = None

    def load_graph(self):
        origin_file = self.maybe_download(self.data_dir, self.source_url)
        convert_file = self.maybe_convert2json(origin_file)
        euler_file = self.maybe_convert2euler(convert_file)
        if not tf_euler.initialize_embedded_graph(euler_file,
                                                  data_type=self.data_type):
            raise RuntimeError('Failed to initialize graph.')

    def get_data_dir(self):
        origin_file = self.maybe_download(self.data_dir, self.source_url)
        convert_file = self.maybe_convert2json(origin_file)
        euler_file = self.maybe_convert2euler(convert_file)
        return euler_file

    def check_file(self, data_files):
        for data_file in data_files:
            if not os.path.exists(data_file):
                return False
        return True

    def maybe_download(self, data_dir, source_url):
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        out_file_names = [os.path.join(data_dir, single_file)
                          for single_file in self.origin_files]
        if not self.check_file(out_file_names):
            print("Downloading data from: {}".format(source_url))
            self.download_data(source_url, data_dir)
        return data_dir

    def maybe_convert2json(self, origin_data_dir):
        out_dir = os.path.join(self.data_dir, 'convert_data.json')
        if not self.check_file([out_dir]):
            print("Converting data to json...")
            self.convert2json(origin_data_dir, out_dir)
        return out_dir

    def maybe_convert2euler(self, convert_data_dir):
        out_dir = os.path.join(self.data_dir, 'euler')
        out_file_names = [os.path.join(out_dir, single_file)
                          for single_file in ['Edge', 'Node', 'euler.meta']]
        if not self.check_file(out_file_names):
            if os.path.isdir(out_dir):
                shutil.rmtree(out_dir)
            print("Converting json data to Euler data...")
            self.convert2euler(convert_data_dir, out_dir)
        return out_dir

    @staticmethod
    def download_file(file_url, path):
        urlretrieve(file_url, path)

    def download_data(self, source_url, out_dir):
        raise NotImplementedError

    def convert2json(self, origin_dir, out_dir):
        raise NotImplementedError

    def convert2euler(self, convert_dir, out_dir):
        dir_name = os.path.dirname(os.path.realpath(__file__))
        convert_meta = self.meta_file
        g = EulerGenerator(convert_dir,
                           convert_meta,
                           out_dir,
                           self.partition_num)
        g.do()

    def remove_data(self):
        print("removing data from {}...".format(self.data_dir))
        shutil.rmtree(self.data_dir)
