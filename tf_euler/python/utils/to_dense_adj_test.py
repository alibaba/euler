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

from tensorflow.python.platform import test
from tf_euler.python.utils import to_dense_adj

import tensorflow as tf


class ToDenseAdjTest(test.TestCase):
    def test_to_dense_adj(self):
        edge_index = tf.constant([
            [0, 0, 1, 2, 3, 4], [0, 1, 0, 3, 4, 2]
        ])
        batch = tf.constant([0, 0, 1, 1, 1])

        adj = to_dense_adj(edge_index, batch)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            ret = sess.run(adj)

            assert ret.shape == (2, 3, 3)
            assert ret[0].tolist() == [[1, 1, 0], [1, 0, 0], [0, 0, 0]]
            assert ret[1].tolist() == [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

    def test_to_dense_adj_with_attr(self):
        edge_index = tf.constant([
            [0, 0, 1, 2, 3, 4], [0, 1, 0, 3, 4, 2]
        ])
        batch = tf.constant([0, 0, 1, 1, 1])
        edge_attr = tf.constant([1, 2, 3, 4, 5, 6])

        adj = to_dense_adj(edge_index, batch, edge_attr)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            ret = sess.run(adj)

            assert ret.shape == (2, 3, 3)
            assert ret[0].tolist() == [[1, 2, 0], [3, 0, 0], [0, 0, 0]]
            assert ret[1].tolist() == [[0, 4, 0], [0, 0, 5], [6, 0, 0]]

    def test_to_dense_adj_with_attr2(self):
        edge_index = tf.constant([
            [0, 0, 1, 2, 3, 4], [0, 1, 0, 3, 4, 2]
        ])
        batch = tf.constant([0, 0, 1, 1, 1])
        batch_size = 2
        edge_attr = tf.constant([1, 2, 3, 4, 5, 6])
        edge_attr = tf.reshape(edge_attr, [-1, 1])

        adj = to_dense_adj(edge_index, batch, edge_attr)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            ret = sess.run(adj)

            assert ret.shape == (2, 3, 3, 1)
            assert ret[0, 0].tolist() == [[1], [2], [0]]
