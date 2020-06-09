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

ALL_NODE_TYPE = -1

get_node_type = base._LIB_OP.get_node_type


def _get_type_id(OP, type_id_or_names):
    type_id_or_names = tf.convert_to_tensor(type_id_or_names)
    if type_id_or_names.dtype == tf.string:
        type_id_or_names = OP(type_id_or_names)
    return type_id_or_names


def get_node_type_id(type_id_or_names):
    """
    Get Node TypeId by type names.

    Args:
      type_id_or_names: A 1-d Tensor/List of node type names/ids(string/int)

    Return:
      A 1-d Tensor of Node TypeId
    """

    return _get_type_id(base._LIB_OP.get_node_type_id, type_id_or_names)


def get_edge_type_id(type_id_or_names):
    """
    Get Edge TypeId by type names.

    Args:
      type_id_or_names: A 1-d Tensor/List of edge type names/ids

    Return:
      A 1-d Tensor of Edge TypeIds
    """

    return _get_type_id(base._LIB_OP.get_edge_type_id, type_id_or_names)
