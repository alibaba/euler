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
        base.initialize_graph({
            'mode': 'local',
            'data_path': '/tmp/euler',
            'sampler_type': 'all',
            'data_type': 'all'
        })

    def testGetNodeSparseFeature(self):
        """Test get sparse feature for nodes"""
        op = ops.get_sparse_feature(
            tf.constant([1, -1, 2, 3, 4], dtype=tf.int64),
            ['f1', 'f2'], None, 2)
        with tf.Session() as sess:
            sparse_features = sess.run(op)
            features = [sess.run(tf.sparse_tensor_to_dense(sp))
                        for sp in sparse_features]

            self.assertAllEqual(
                [[11, 12], [0, 0], [21, 22], [31, 32], [41, 42]],
                features[0])
            self.assertAllEqual(
                [[13, 14], [0, 0], [23, 24], [33, 34], [43, 44]],
                features[1])

    def testGetEdgeSparseFeature(self):
        """Test get sparse feature for edges"""
        op = ops.get_edge_sparse_feature(
            tf.constant([[1, 2, 0], [2, 3, 1]], dtype=tf.int64),
            ['f1', 'f2'], None)
        with tf.Session() as sess:
            sparse_features = sess.run(op)
            features = [sess.run(tf.sparse_tensor_to_dense(sp))
                        for sp in sparse_features]
            self.assertAllEqual(
                [[[121, 122], [231, 232]], [[123, 124], [233, 234]]],
                features)

    def testGetBinaryFeature(self):
        """Test get binaray feature for nodes"""
        op = ops.get_binary_feature(
            tf.constant([1, 2], dtype=tf.int64), ['f5', 'f6'], 3)
        with tf.Session() as sess:
            binary_features = sess.run(op)
            self.assertAllEqual([['1a', '2a'], ['1b', '2b']], binary_features)

    def testGetEdgeBinaryFeature(self):
        """Test get binary feature for edges"""
        op = ops.get_edge_binary_feature(
            tf.constant([[1, 2, 0], [2, 3, 1]], dtype=tf.int64), ['f5'])
        with tf.Session() as sess:
            binary_features = sess.run(op)
            self.assertAllEqual([['12a', '23a']], binary_features)

    def testGetDenseFeature(self):
        """Test get dense feature for nodes"""
        op = ops.get_dense_feature(
            tf.constant([1, 0, 2], dtype=tf.int64), ["f3", "f4"], [2, 3], 2)
        with tf.Session() as sess:
            dense_features = sess.run(op)
            self.assertAllClose([[1.1, 1.2], [0.0, 0.0], [2.1, 2.2]],
                                dense_features[0])
            self.assertAllClose([[1.3, 1.4, 1.5],
                                 [0.0, 0.0, 0.0],
                                 [2.3, 2.4, 2.5]],
                                dense_features[1])

    def testGetEdgeDenseFeature(self):
        """Test get dense feature for edges"""
        op = ops.get_edge_dense_feature(
            tf.constant([[1, 2, 0], [2, 3, 1]], dtype=tf.int64),
            ["f3", "f4"], [2, 3], 2)
        with tf.Session() as sess:
            dense_features = sess.run(op)
            self.assertAllClose([[12.1, 12.2], [23.1, 23.2]],
                                dense_features[0])
            self.assertAllClose([[12.3, 12.4, 12.5], [23.3, 23.4, 23.5]],
                                dense_features[1])


if __name__ == "__main__":
    test.main()
