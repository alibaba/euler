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
    """
    Neighbor Ops Test. Test get neighbors ops for nodes
    """

    @classmethod
    def setUpClass(cls):
        base.initialize_graph({
            'mode': 'local',
            'data_path': '/tmp/euler',
            'sampler_type': 'all',
            'data_type': 'all'
        })

    def testGetBatchAdj(self):
        op = ops.get_batch_adj([1, 2, 3, 4, 5, 6,
                                1, 2, 3, 4, -1, -1],
                               ["0", "1"], 2, 6, 0)
        with tf.Session() as sess:
            sparse_adj = sess.run(op)
            adj = sess.run(tf.sparse_tensor_to_dense(sparse_adj))
            print(adj)

    def testGetFullNeighbor(self):
        """Test get full neighbors for nodes"""

        op = ops.get_full_neighbor([1, 2], ["0", "1"])
        with tf.Session() as sess:
            sparse_ids, sparse_weights, sparse_types = sess.run(op)
            ids = sess.run(tf.sparse_tensor_to_dense(sparse_ids))
            weights = sess.run(tf.sparse_tensor_to_dense(sparse_weights))
            types = sess.run(tf.sparse_tensor_to_dense(sparse_types))
            self.assertAllEqual([[2, 4, 3], [3, 5, 0]], ids)
            self.assertAllClose([[2.0, 4.0, 3.0], [3.0, 5.0, 0.0]], weights)
            self.assertAllEqual([[0, 0, 1], [1, 1, 0]], types)

    def testGetFullNeighborWithCondition(self):
        """Test get full neighbors for nodes"""

        op = ops.get_full_neighbor([1, 2, 3, 4], ["0", "1"], 'price gt 3')
        with tf.Session() as sess:
            sparse_ids, sparse_weights, sparse_types = sess.run(op)
            ids = sess.run(tf.sparse_tensor_to_dense(sparse_ids))
            weights = sess.run(tf.sparse_tensor_to_dense(sparse_weights))
            types = sess.run(tf.sparse_tensor_to_dense(sparse_types))
            self.assertAllEqual([[4, 3], [3, 5], [4, 0], [5, 0]], ids)
            self.assertAllClose([[4.0, 3.0], [3.0, 5.0], [4.0, 0], [5.0, 0]],
                                weights)
            self.assertAllEqual([[0, 1], [1, 1], [0, 0], [1, 0]], types)

    def testGetSortedFullNeighbor(self):
        """Test get sorted full neighbors for nodes"""

        op = ops.get_sorted_full_neighbor([1, 2], ["0", "1"])
        with tf.Session() as sess:
            sparse_ids, sparse_weights, sparse_types = sess.run(op)
            ids = sess.run(tf.sparse_tensor_to_dense(sparse_ids))
            weights = sess.run(tf.sparse_tensor_to_dense(sparse_weights))
            types = sess.run(tf.sparse_tensor_to_dense(sparse_types))
            self.assertAllEqual([[2, 3, 4], [3, 5, 0]], ids)
            self.assertAllClose([[2.0, 3.0, 4.0], [3.0, 5.0, 0.0]], weights)
            self.assertAllEqual([[0, 1, 0], [1, 1, 0]], types)

    def testGetSortedFullNeighborWithCondition(self):
        """Test get full neighbors for nodes"""

        op = ops.get_sorted_full_neighbor([1, 2, 3, 4],
                                          ["0", "1"],
                                          'price gt 3')
        with tf.Session() as sess:
            sparse_ids, sparse_weights, sparse_types = sess.run(op)
            ids = sess.run(tf.sparse_tensor_to_dense(sparse_ids))
            weights = sess.run(tf.sparse_tensor_to_dense(sparse_weights))
            types = sess.run(tf.sparse_tensor_to_dense(sparse_types))
            self.assertAllEqual([[3, 4], [3, 5], [4, 0], [5, 0]], ids)
            self.assertAllClose([[3.0, 4.0], [3.0, 5.0], [4.0, 0], [5.0, 0]],
                                weights)
            self.assertAllEqual([[1, 0], [1, 1], [0, 0], [1, 0]], types)

    def testGetTopKNeighbor(self):
        """Test get top k neighbor for nodes"""

        op = ops.get_top_k_neighbor([1, 2], ["0", "1"], 2)
        with tf.Session() as sess:
            ids, weights, types = sess.run(op)
            self.assertAllEqual([[4, 3], [5, 3]], ids)
            self.assertAllClose([[4.0, 3.0], [5.0, 3.0]], weights)
            self.assertAllEqual([[0, 1], [1, 1]], types)

    def testGetTopKNeighborWithCondition(self):
        """Test get top k neighbor for nodes"""

        op = ops.get_top_k_neighbor([1, 2], ["0", "1"],
                                    2,
                                    condition='price gt 4')
        with tf.Session() as sess:
            ids, weights, types = sess.run(op)
            self.assertAllEqual([[4, -1], [5, -1]], ids)
            self.assertAllClose([[4.0, 0.0], [5.0, 0.0]], weights)
            self.assertAllEqual([[0, -1], [1, -1]], types)

    def testSampleNeighbor(self):
        """Test Sample Neighbor for nodes"""

        op = ops.sample_neighbor([1, 2], ["0", "1"], 10)
        with tf.Session() as sess:
            ids, weights, types = sess.run(op)
            self.assertEqual(10, len(ids[0]))
            self.assertEqual(10, len(weights[0]))
            self.assertEqual(10, len(types[0]))

            [self.assertTrue(n1 in [2, 3, 4]) for n1 in ids[0]]
            [self.assertTrue(int(w1) in [2, 3, 4]) for w1 in weights[0]]
            # [self.assertTrue(t1 in [0, 1] for t1 in types[0])]

            [self.assertTrue(n2 in [3, 5]) for n2 in ids[1]]
            [self.assertTrue(int(w2) in [3, 5]) for w2 in weights[1]]
            # [self.assertTrue(t2 in [0, 1] for t2 in types[1])]

    def testSampleNeighborLayerwise(self):
        """Test sample neighbor layerwise"""

        op = ops.sample_neighbor_layerwise([[1, 2, 3], [1, 2, 3],
                                            [2, 3, 4], [2, 2, 4]],
                                           ['0', '1'],
                                           10)
        with tf.Session() as sess:
            for i in range(10):
                ids, _ = sess.run(op)
                self.assertEqual(10, len(ids[0]))
                [self.assertTrue(n1 in [2, 3, 4, 5]) for n1 in ids[0]]
                [self.assertTrue(n2 in [3, 4, 5]) for n2 in ids[2]]
                [self.assertTrue(n3 in [3, 5]) for n3 in ids[3]]

    def testSampleNeighborLayerwiseWithAdj(self):
        """Test sample neighbor layerwise"""

        ans_adj = set([(1, 2), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (4, 5)])
        src_nodes = [[1, 2, 3], [1, 2, 3], [2, 3, 4], [2, 2, 4]]
        op = ops.sample_neighbor_layerwise(
            src_nodes, ["0", "1"], 10, -1, 'sqrt')
        with tf.Session() as sess:
            for i in range(0, 1):
                ids, adj = sess.run(op)
                adj = sess.run(tf.sparse_tensor_to_dense(adj))
                [self.assertTrue(n1 in [2, 3, 4, 5]) for n1 in ids[0]]
                [self.assertTrue(n2 in [3, 4, 5]) for n2 in ids[2]]
                [self.assertTrue(n3 in [3, 5]) for n3 in ids[3]]
                for bs in range(0, 4):
                    for src_idx in range(0, 3):
                        for dst_idx in range(0, 10):
                            pair = (src_nodes[bs][src_idx], ids[bs][dst_idx])
                            self.assertTrue(
                                (pair in ans_adj) == adj[bs][src_idx][dst_idx],
                                "{},{},{},pair:{}".format(bs, src_idx,
                                                          dst_idx, pair))

    def testSampleNeighborLayerwiseFanout(self):
        """Test sample fanout neighbor layerwise"""

        op = ops.sample_fanout_layerwise_each_node(
            tf.constant([1, 2, 3, 4, 5], dtype=tf.int64),
            [['0', '1'], ['0', '1']],
            [10, 20])
        with tf.Session() as sess:
            ids, adj = sess.run(op)
            self.assertEqual(50, len(ids[1]))
            self.assertEqual(100, len(ids[2]))

        op = ops.sample_fanout_layerwise(
            tf.constant([1, 2, 3, 4, 5], dtype=tf.int64),
            [['0', '1'], ['0', '1']],
            [10, 20])
        with tf.Session() as sess:
            ids, adj = sess.run(op)
            self.assertEqual(10, len(ids[1]))
            self.assertEqual(20, len(ids[2]))
            print (ids[1], sess.run(tf.sparse_tensor_to_dense(adj[0])),
                   ids[2], sess.run(tf.sparse_tensor_to_dense(adj[1])))

    def testSampleFanoutWithFeature(self):
        """test sample fanout with feature"""

        dense_feature_names = ["dense_f3", "dense_f4"]
        sparse_feature_names = ["sparse_f1", "sparse_f2"]
        fanout = [['0', '1'], ['0', '1']]
        op = ops.sample_fanout_with_feature(
            tf.constant([1, 2, 0, 3], dtype=tf.int64),
            fanout,
            count=[3, 3],
            default_node=-1,
            sparse_feature_names=sparse_feature_names,
            sparse_default_values=[0]*len(sparse_feature_names),
            dense_feature_names=dense_feature_names,
            dense_dimensions=[2, 3])
        with tf.Session() as sess:
            neighbors, weights, types, dense_features, sparse_features = \
                sess.run(op)
            s = [sess.run(tf.sparse_tensor_to_dense(sp))
                 for sp in sparse_features]

    def testSparseGetAdj(self):
        op = ops.sparse_get_adj([1, 2, 3], [4, 5, 6], ['0', '1'])
        with tf.Session() as sess:
            adj = sess.run(op)
            print(sess.run(tf.sparse_tensor_to_dense(adj)))


if __name__ == "__main__":
    test.main()
