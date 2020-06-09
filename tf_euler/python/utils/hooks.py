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

import time

import tensorflow as tf


class SyncExitHook(tf.train.SessionRunHook):
    def __init__(self, num_workers):
        self._num_workers = num_workers
        self._num_finished_workers = tf.Variable(
            0, name="num_finished_workers")
        self._finish_self = tf.assign_add(
              self._num_finished_workers, 1, use_locking=True)

    def end(self, session):
        session.run(self._finish_self)
        num_finished_workers = session.run(self._num_finished_workers)
        while num_finished_workers < self._num_workers:
            tf.logging.info("%d workers have finished ...",
                            num_finished_workers)
            time.sleep(1)
            num_finished_workers = session.run(self._num_finished_workers)
