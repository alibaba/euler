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

import euler
import tensorflow as tf
import logging

logger = logging.getLogger("simple_example")
logger.setLevel(logging.DEBUG)

_LIB_DIR = os.path.dirname(os.path.realpath(__file__))
_LIB_PATH = os.path.join(_LIB_DIR, 'libtf_euler.so')

_LIB_OP = tf.load_op_library(_LIB_PATH)
_LIB = ctypes.CDLL(_LIB_PATH)


def initialize_graph(config):
    """
    Initialize the Euler graph driver used in Tensorflow.

    Args:
      config: str or dict of Euler graph driver configuration.

    Return:
      A boolean indicate whether the graph driver is initialized successfully.

    Raises:
      TypeError: if config is neither str nor dict.
    """
    if isinstance(config, dict):
        config = ';'.join(
            '{}={}'.format(key, value) for key, value in config.items())
    if not isinstance(config, str):
        raise TypeError('Expect str or dict for graph config, '
                        'got {}.'.format(type(config).__name__))

    if not isinstance(config, bytes):
        config = config.encode()

    return _LIB.InitQueryProxy(config)


def initialize_embedded_graph(data_dir, sampler_type='all', data_type='all'):
    return initialize_graph({'mode': 'local',
                             'data_path': data_dir,
                             'data_type': data_type,
                             'sampler_type': sampler_type})


def initialize_shared_graph(data_dir, zk_addr, zk_path, shard_num):
    return initialize_graph({'mode': 'remote',
                             'zk_server': zk_addr,
                             'zk_path': zk_path,
                             'shard_num': shard_num,
                             'num_retries': 1})
