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

import tensorflow as tf

from tf_euler.python.euler_ops.mp_ops import scatter_


class Pooling(object):
    def __init__(self, aggr='add'):
        assert aggr in ['add', 'mean', 'max']
        self.aggr = aggr

    def __call__(self, inputs, index, size=None):
        size = tf.reduce_max(index) + 1 if size is None else size
        out = scatter_(self.aggr, inputs, index, size)
        return out
