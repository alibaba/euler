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
"""Euler walk ops test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from tensorflow.python.platform import test
import tensorflow as tf

from tf_euler.python.euler_ops import base
from tf_euler.python.euler_ops import walk_ops as ops


class WalkOpsTest(test.TestCase):
  """Walk Ops Test.

    Test random walk ops on euler graph
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

  def testGenPair(self):
    op = ops.gen_pair([[1, 2, 3, 4, 5, 6, 7, 8, 9]], 2, 2)
    with tf.Session() as sess:
      pairs = sess.run(op)
      self.assertAllEqual(
          [[[1, 2], [1, 3], [2, 1], [2, 3], [2, 4], [3, 2], [3, 1], [3, 4],
            [3, 5], [4, 3], [4, 2], [4, 5], [4, 6], [5, 4], [5, 3], [5, 6],
            [5, 7], [6, 5], [6, 4], [6, 7], [6, 8], [7, 6], [7, 5], [7, 8],
            [7, 9], [8, 7], [8, 6], [8, 9], [9, 8], [9, 7]]], pairs)

  def testRandomWalk(self):
    op = ops.random_walk([1, 2], [1 for _ in range(10)], 10, 1.0, 2.0)
    with tf.Session() as sess:
      paths = sess.run(op)
      self.assertAllEqual([2, 11], paths.shape)
      self.assertEqual(1, paths[0][0])
      self.assertEqual(2, paths[1][0])


if __name__ == "__main__":
  test.main()
