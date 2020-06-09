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

import ctypes
import os

import tensorflow as tf

from tf_euler.python.euler_ops import base

gather = base._LIB_OP.mp_gather
scatter_add = base._LIB_OP.mp_scatter_add
scatter_max = base._LIB_OP.mp_scatter_max


"""
def scatter_add(updates, indices, size=None):
    out = tf.zeros([size, updates.shape[1]], dtype=updates.dtype)
    return tf.tensor_scatter_add(out, tf.reshape(indices, [-1, 1]), updates)
"""


@tf.RegisterGradient('MPGather')
def _GatherGrad(op, grad):
    params = op.inputs[0]
    indices = op.inputs[1]
    return [scatter_add(grad, indices, tf.shape(params)[0]), None]


@tf.RegisterGradient('MPScatterAdd')
def _ScatterAddGrad(op, grad):
    indices = op.inputs[1]
    return [gather(grad, indices), None, None]


@tf.RegisterGradient('MPScatterMax')
def _ScatterMaxGrad(op, grad):
    updates = op.inputs[0]
    indices = op.inputs[1]
    size = op.inputs[2]
    out = op.outputs[0]
    indicators = tf.equal(updates, gather(out, indices))
    indicators = tf.cast(indicators, updates.dtype)
    num_selected = scatter_add(indicators, indices, size)
    indicators = indicators / gather(num_selected, indices)
    return [indicators * gather(grad, indices), None, None]


def scatter_mean(updates, indices, size=None):
    out = scatter_add(updates, indices, size)
    ep = 1e-7
    count = scatter_add(tf.ones([tf.shape(updates)[0], 1]), indices, size) + ep
    return out / count


def scatter_(op, updates, indices, size):
    return globals()['scatter_' + op](updates, indices, size)


def scatter_softmax(updates, indices, size=None):
    updates = updates - gather(scatter_max(updates, indices, size), indices)
    updates = tf.exp(updates)
    return updates / gather(scatter_add(updates, indices, size), indices)
