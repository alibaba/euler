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

"""Graph Initialize python api test"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from tensorflow.python.platform import test
from tf_euler.python.euler_ops import base


class BaseTest(test.TestCase):
    """Graph Initialize test.

    Test python wrapper for Graph c++ api works as expected.
    """

    @classmethod
    def setUpClass(cls):
        """Build Graph data for test"""
        cls._cur_dir = os.path.dirname(os.path.realpath(__file__))

    def testInitializeGraph(self):
        self.assertTrue(
            base.initialize_graph({
                'mode': 'local',
                'data_path': '/tmp/euler',
                'sampler_type': u'all'
            }))


if __name__ == "__main__":
    test.main()
