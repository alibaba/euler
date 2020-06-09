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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tf_euler.python.convolution import *
from tf_euler.python.euler_ops import mp_ops


class SimpleConv(Conv):

    def __call__(self, x, edge_index, size=None):
        gather_x, = self.gather_feature([x], edge_index)
        out = self.apply_edge(gather_x[1])
        out = mp_ops.scatter_(self.aggr, out, edge_index[0], size=size[0])
        out = self.apply_node(out)
        return out


class ConvTest(tf.test.TestCase):

    def testConv(self):
        edge_index = tf.constant([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=tf.int32)
        x = tf.constant([[-1], [0], [1]], dtype=tf.float32)
        x = [x, x]
        conv = SimpleConv()
        x1 = conv(x, edge_index, size=[3, 3])

        with self.test_session():
            self.assertAllEqual([[0.], [0.], [0.]], x1.eval())


if __name__ == '__main__':
    tf.test.main()
