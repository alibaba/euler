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
"""Euler sample ops test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from tensorflow.python.platform import test
import tensorflow as tf

from tf_euler.python.euler_ops import base
from tf_euler.python.euler_ops import sample_ops as ops


class SampleOpsTest(test.TestCase):
  """Sample Ops Test.

    Test sample ops for euler graph
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
      subprocess.call(command, shell=True)
    except:
      raise RuntimeError("Build Graph for test failed")

    base.initialize_graph({
        'mode': 'Local',
        'directory': os.path.join(cur_dir, 'testdata'),
        'load_type': 'compact'
    })

  def testSampleNode(self):
    """Test euler graph sample node"""
    op = ops.sample_node(10, 1)
    with tf.Session() as sess:
      nodes = sess.run(op)
      self.assertEqual(10, len(nodes))
      [self.assertTrue(n in [1, 3, 5]) for n in nodes]

  def testSampleEdge(self):
    """Test euler graph sample edge"""
    op = ops.sample_edge(10, 1)
    with tf.Session() as sess:
      edges = sess.run(op)
      self.assertEqual(10, len(edges))
      [self.assertTrue(e[1] in [1, 3, 5]) for e in edges]


if __name__ == "__main__":
  test.main()
