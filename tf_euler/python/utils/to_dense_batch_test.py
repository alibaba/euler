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
from tf_euler.python.utils import to_dense_batch

import tensorflow as tf


class ToDenseBatchTest(test.TestCase):
    def test_to_dense_batch(self):
        x = tf.constant([
            [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]
        ])
        batch = tf.constant([0, 0, 1, 2, 2, 2])

        out_op, mask_op = to_dense_batch(x, batch)

        init = tf.global_variables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            out, mask = sess.run([out_op, mask_op])

        expected = [
            [[1, 2], [3, 4], [0, 0]],
            [[5, 6], [0, 0], [0, 0]],
            [[7, 8], [9, 10], [11, 12]],
        ]

        assert out.shape == (3, 3, 2)
        assert out.tolist() == expected
        assert mask.tolist() == [1, 1, 0, 1, 0, 0, 1, 1, 1]


if __name__ == '__main__':
    test.main()
