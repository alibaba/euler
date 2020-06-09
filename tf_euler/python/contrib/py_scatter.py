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

import numpy as np
import tensorflow as tf
import uuid


def scatter_add(src, indices, size=None, fill_value=0):

    def _scatter_add(src, indices, size, fill_value):
        assert len(indices.shape) == 1
        assert src.shape[0] == indices.shape[0]
        n = src.shape[0]

        out_shape = list(src.shape)
        out_shape[0] = size
        out = np.full(out_shape, fill_value, dtype=src.dtype)
        for i in range(n):
            out[indices[i]] += src[i]
        return out

    def _scatter_add_grad_op(op, grad):
        grads = [None for _ in op.inputs]
        src = op.inputs[0]
        indices = op.inputs[1]
        grads[0] = tf.gather(grad, indices)
        return grads

    grad_name = 'ScatterAddGrad_' + str(uuid.uuid4())
    tf.RegisterGradient(grad_name)(_scatter_add_grad_op)

    if size is None:
        size = tf.reduce_max(indices) + 1

    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        out = tf.py_func(_scatter_add,
                         [src, indices, size, fill_value],
                         src.dtype)
    out.set_shape([None, src.shape[1]])
    return out
