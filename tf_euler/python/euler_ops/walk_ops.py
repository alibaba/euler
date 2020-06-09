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

from tf_euler.python.euler_ops import base
from tf_euler.python.euler_ops import type_ops

gen_pair = base._LIB_OP.gen_pair
_random_walk = base._LIB_OP.random_walk


def random_walk(nodes, edge_types, p=1.0, q=1.0, default_node=-1):
    '''
    Random walk from a list of nodes.

    Args:
    nodes: start node ids, 1-d Tensor
    edge_types: list of 1-d Tensor of edge types
    p: back probality
    q: forward probality
    default_node: default fill nodes
    '''

    edge_types = [type_ops.get_edge_type_id(edge_type)
                  for edge_type in edge_types]
    return _random_walk(nodes, edge_types, p, q, default_node)
