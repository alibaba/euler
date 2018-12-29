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
"""Euler feature ops test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from tensorflow.python.platform import test
import tensorflow as tf

from tf_euler.python.euler_ops import base
from tf_euler.python.euler_ops import feature_ops as ops


class FeatureOpsTest(test.TestCase):
  """Feature Ops Test.

    Test get feature(binary/sparse/dense) ops for nodes and edges
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

  def testGetNodeSparseFeature(self):
    """Test get sparse feature for nodes"""
    op = ops.get_sparse_feature([1, 2], [0, 1])
    with tf.Session() as sess:
      sparse_features = sess.run(op)
      features = [
          sess.run(tf.sparse_tensor_to_dense(sp)) for sp in sparse_features
      ]
      self.assertAllEqual(
          [[[1234, 5678], [1234, 5678]], [[8888, 9999], [8888, 9999]]],
          features)

  def testGetEdgeSparseFeature(self):
    """Test get sparse feature for edges"""
    op = ops.get_edge_sparse_feature([[1, 2, 0], [2, 3, 1]], [0, 1])
    with tf.Session() as sess:
      sparse_features = sess.run(op)
      features = [
          sess.run(tf.sparse_tensor_to_dense(sp)) for sp in sparse_features
      ]
      self.assertAllEqual(
          [[[1234, 5678], [1234, 5678]], [[8888, 9999], [8888, 9999]]],
          features)

  def testGetBinaryFeature(self):
    """Test get binaray feature for nodes"""
    op = ops.get_binary_feature([1, 2], [0, 1])
    with tf.Session() as sess:
      binary_features = sess.run(op)
      self.assertAllEqual([['aa', 'eaa'], ['bb', 'ebb']], binary_features)

  def testGetEdgeBinaryFeature(self):
    """Test get binary feature for edges"""
    op = ops.get_edge_binary_feature([[1, 2, 0], [2, 3, 1]], [0, 1])
    with tf.Session() as sess:
      binary_features = sess.run(op)
      self.assertAllEqual([['eaa', 'eaa'], ['ebb', 'ebb']], binary_features)

  def testGetDenseFeature(self):
    """Test get dense feature for nodes"""
    op = ops.get_dense_feature([1, 2], [0, 1], [2, 3])
    with tf.Session() as sess:
      dense_features = sess.run(op)
      self.assertAllClose([[2.4, 3.6], [2.4, 3.6]], dense_features[0])
      self.assertAllClose([[4.5, 6.7, 8.9], [4.5, 6.7, 8.9]],
                          dense_features[1])

  def testGetEdgeDenseFeature(self):
    """Test get dense feature for edges"""
    op = ops.get_edge_dense_feature([[1, 2, 0], [2, 3, 1]], [0, 1], [2, 3])
    with tf.Session() as sess:
      dense_features = sess.run(op)
      self.assertAllClose([[2.4, 3.6], [2.4, 3.6]], dense_features[0])
      self.assertAllClose([[4.5, 6.7, 8.9], [4.5, 6.7, 8.9]],
                          dense_features[1])


if __name__ == "__main__":
  test.main()
