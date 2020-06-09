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

from euler.tools.json2meta import Generator
from euler.tools.json2partdat import Converter as DatConverter
from euler.tools.json2partindex import Converter as IndexConverter

import os
import sys


class EulerGenerator(object):

    def __init__(self, graph_json, index_meta, output_dir, partition_num):
        self.graph_json = os.path.realpath(graph_json)
        self.index_meta = index_meta
        self.output_dir = os.path.realpath(output_dir)
        self.partition_num = partition_num

    def do(self):
        meta_dir = os.path.join(self.output_dir, 'euler.meta')
        g = Generator([self.graph_json], meta_dir, self.partition_num)
        g.do()
        d = DatConverter(self.graph_json,
                         meta_dir,
                         self.output_dir,
                         self.partition_num)
        d.do()
        if self.index_meta is not None:
            i = IndexConverter(self.index_meta,
                               self.graph_json,
                               self.output_dir,
                               self.partition_num)
            i.do()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("python generate_euler_data.py graph.json output_dir "
              "partition_num [index.meta]")
        exit(1)
    index_meta = None
    if len(sys.argv) == 5:
        index_meta = sys.argv[4]
    g = EulerGenerator(sys.argv[1], index_meta, sys.argv[2], int(sys.argv[3]))
    g.do()
