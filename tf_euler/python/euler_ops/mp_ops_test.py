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

"""Message Passing ops test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tf_euler.python.euler_ops import mp_ops


class ScatterTest(tf.test.TestCase):

    def testScatterAdd(self):
        x = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
        idx = tf.constant([1, 0, 1])
        out = mp_ops.scatter_add(x, idx, size=2)

        with self.test_session():
            self.assertAllEqual([[3., 4.], [6., 8.]], out.eval())

    def testScatterAddGrad(self):
        x = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
        idx = tf.constant([1, 0, 1])
        out = mp_ops.scatter_add(x, idx, size=2)
        with self.test_session():
            diff = tf.test.compute_gradient_error(x, [3, 2], out, [2, 2])
            self.assertLess(diff, 1e-4)

    def testScatterMean(self):
        x = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
        idx = tf.constant([1, 0, 1])
        out = mp_ops.scatter_mean(x, idx, size=2)

        with self.test_session():
            self.assertAllEqual([[3., 4.], [3., 4.]], out.eval())

    def testScatterMeanGrad(self):
        x = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
        idx = tf.constant([1, 0, 1])
        out = mp_ops.scatter_mean(x, idx, size=2)
        with self.test_session():
            diff = tf.test.compute_gradient_error(x, [3, 2], out, [2, 2])
            self.assertLess(diff, 1e-4)

    def testScatterMax(self):
        x = tf.constant([[1., 6.], [3., 4.], [5., 2.]])
        idx = tf.constant([1, 0, 1])
        out = mp_ops.scatter_max(x, idx, size=2)

        with self.test_session():
            self.assertAllEqual([[3., 4.], [5., 6.]], out.eval())

    def testScatterMaxGrad(self):
        x = tf.constant([[1., 2., 7], [3., 4., 8.], [5., 6., 7.]])
        idx = tf.constant([1, 0, 1])
        out = mp_ops.scatter_max(x, idx, size=2)
        with self.test_session():
            diff = tf.test.compute_gradient_error(x, [3, 3], out, [2, 3])
            self.assertLess(diff, 1e-4)

    def testGather(self):
        x = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
        idx = tf.constant([1, 0, 1, 2])
        out = mp_ops.gather(x, idx)
        with self.test_session():
            self.assertAllEqual([[3., 4.], [1., 2.], [3., 4.], [5., 6]],
                                out.eval())

    def testGatherGrad(self):
        x = tf.constant([[1., 2.], [3., 4.], [5., 6.]])
        idx = tf.constant([1, 0, 1, 2])
        out = mp_ops.gather(x, idx)
        with self.test_session():
            diff = tf.test.compute_gradient_error(x, [3, 2], out, [4, 2])
            self.assertLess(diff, 1e-4)


if __name__ == '__main__':
    tf.test.main()
