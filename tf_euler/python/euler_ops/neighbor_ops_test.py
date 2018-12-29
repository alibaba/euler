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
"""Euler neighbor ops test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from tensorflow.python.platform import test
import tensorflow as tf

from tf_euler.python.euler_ops import base
from tf_euler.python.euler_ops import neighbor_ops as ops


class NeighborOpsTest(test.TestCase):
  """Neighbor Ops Test.

    Test get neighbors ops for nodes
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

  def testGetFullNeighbor(self):
    """Test get full neighbors for nodes"""
    op = ops.get_full_neighbor([1, 2], [0, 1])
    with tf.Session() as sess:
      sparse_ids, sparse_weights, sparse_types = sess.run(op)
      ids = sess.run(tf.sparse_tensor_to_dense(sparse_ids))
      weights = sess.run(tf.sparse_tensor_to_dense(sparse_weights))
      types = sess.run(tf.sparse_tensor_to_dense(sparse_types))
      self.assertAllEqual([[2, 4, 3], [3, 5, 0]], ids)
      self.assertAllClose([[2.0, 4.0, 3.0], [3.0, 5.0, 0.0]], weights)
      self.assertAllEqual([[0, 0, 1], [1, 1, 0]], types)

  def testGetSortedFullNeighbor(self):
    """Test get sorted full neighbors for nodes"""
    op = ops.get_sorted_full_neighbor([1, 2], [0, 1])
    with tf.Session() as sess:
      sparse_ids, sparse_weights, sparse_types = sess.run(op)
      ids = sess.run(tf.sparse_tensor_to_dense(sparse_ids))
      weights = sess.run(tf.sparse_tensor_to_dense(sparse_weights))
      types = sess.run(tf.sparse_tensor_to_dense(sparse_types))
      self.assertAllEqual([[2, 3, 4], [3, 5, 0]], ids)
      self.assertAllClose([[2.0, 3.0, 4.0], [3.0, 5.0, 0.0]], weights)
      self.assertAllEqual([[0, 1, 0], [1, 1, 0]], types)

  def testGetTopKNeighbor(self):
    """Test get top k neighbor for nodes"""
    op = ops.get_top_k_neighbor([1, 2], [0, 1], 2)
    with tf.Session() as sess:
      ids, weights, types = sess.run(op)
      self.assertAllEqual([[4, 3], [5, 3]], ids)
      self.assertAllClose([[4.0, 3.0], [5.0, 3.0]], weights)
      self.assertAllEqual([[0, 1], [1, 1]], types)

  def testSampleNeighbor(self):
    """Test Sample Neighbor for nodes"""
    op = ops.sample_neighbor([1, 2], [0, 1], 10)
    with tf.Session() as sess:
      ids, weights, types = sess.run(op)
      self.assertEqual(10, len(ids[0]))
      self.assertEqual(10, len(weights[0]))
      self.assertEqual(10, len(types[0]))

      [self.assertTrue(n1 in [2, 3, 4]) for n1 in ids[0]]
      [self.assertTrue(int(w1) in [2, 3, 4]) for w1 in weights[0]]
      [self.assertTrue(t1 in [0, 1] for t1 in types[0])]

      [self.assertTrue(n2 in [3, 5]) for n2 in ids[1]]
      [self.assertTrue(int(w2) in [3, 5]) for w2 in weights[1]]
      [self.assertTrue(t2 in [0, 1] for t2 in types[1])]


if __name__ == "__main__":
  test.main()
