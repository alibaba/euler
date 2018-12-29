# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
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
"""Graph Initialize python api test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from tensorflow.python.platform import test

from tf_euler.python.euler_ops import base


class BaseTest(test.TestCase):
  """Graph Initialize test.

    Test python wrapper for Graph c++ api works as expected.
    """

  @classmethod
  def setUpClass(cls):
    """Build Graph data for test"""
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    meta_file = os.path.join(cur_dir, 'testdata/meta.json')
    graph_file = os.path.join(cur_dir, 'testdata/graph.json')
    output_file = os.path.join(cur_dir, 'testdata/graph.dat')
    builder = os.path.join(cur_dir, '../../../tools/bin/json2dat.py')

    command = "python {builder} -i {input} -c {meta} -o {output}".format(
        builder=builder, input=graph_file, meta=meta_file, output=output_file)

    try:
      ret_code = subprocess.call(command, shell=True)
    except:
      raise RuntimeError("Build Graph for test failed")

    cls._cur_dir = cur_dir

  def testInitializeGraph(self):
    self.assertTrue(
        base.initialize_graph({
            'mode':
            'Local',
            'directory':
            os.path.join(BaseTest._cur_dir, 'testdata'),
            'load_type':
            'compact'
        }))


if __name__ == "__main__":
  test.main()
