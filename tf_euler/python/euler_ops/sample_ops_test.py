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

"""Euler sample ops test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from tensorflow.python.platform import test
import tensorflow as tf
import numpy as np

from tf_euler.python.euler_ops import base
from tf_euler.python.euler_ops import sample_ops as ops


class SampleOpsTest(test.TestCase):
    """Sample Ops Test.

    Test sample ops for euler graph
    """

    @classmethod
    def setUpClass(cls):
        base.initialize_graph({
            'mode': 'local',
            'data_path': '/tmp/euler',
            'sampler_type': 'all',
            'data_type': 'all'
        })

    def testSampleNode(self):
        """Test euler graph sample node"""

        op = ops.sample_node(10, '-1')
        with tf.Session() as sess:
            nodes = sess.run(op)
            self.assertEqual(10, len(nodes))
            [self.assertTrue(n in [1, 2, 3, 4, 5, 6]) for n in nodes]

        op = ops.sample_node(10, '0')
        with tf.Session() as sess:
            nodes = sess.run(op)
            self.assertEqual(10, len(nodes))
            [self.assertTrue(n in [2, 4, 6]) for n in nodes]

    def testSampleNodeWithCondition(self):
        """Test euler graph sample node"""

        c = 'price gt 4'
        op = ops.sample_node(10, '1', condition=c)
        with tf.Session() as sess:
            nodes = sess.run(op)
            self.assertEqual(10, len(nodes))
            [self.assertTrue(n in [5]) for n in nodes]

        op = ops.sample_node(10, '0', condition=c)
        with tf.Session() as sess:
            nodes = sess.run(op)
            self.assertEqual(10, len(nodes))
            [self.assertTrue(n in [4, 6]) for n in nodes]

    def testSampleEdge(self):
        """Test euler graph sample edge"""

        op = ops.sample_edge(10, '1')
        with tf.Session() as sess:
            edges = sess.run(op)
            self.assertEqual(10, len(edges))
            [self.assertTrue(e[1] in [1, 3, 5]) for e in edges]

    def testSampleNodeWithSrc(self):
        """Test euler graph sample edge"""

        op = ops.sample_node_with_src([1, 2, 3, 4, 5], 5)
        with tf.Session() as sess:
            samples = sess.run(op)
            samples = np.array(samples)
            self.assertEqual(samples.shape, (5, 5))
            for i in range(0, 5):
                if i % 2 == 0:
                    [self.assertTrue(n in [1, 3, 5]) for n in samples[i]]
                else:
                    [self.assertTrue(n in [2, 4, 6]) for n in samples[i]]

    def testSampleGraphLabel(self):
        op = ops.sample_graph_label(10)
        with tf.Session() as sess:
            graph_labels = sess.run(op)
            print(graph_labels)

    def testGetGraphByLabel(self):
        op = tf.sparse_tensor_to_dense(ops.get_graph_by_label(['1', '2', '3']))
        with tf.Session() as sess:
            graphs = sess.run(op)
            print(graphs)

if __name__ == "__main__":
    test.main()
