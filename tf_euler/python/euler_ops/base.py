# Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.
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
import euler

_LIB_DIR = os.path.dirname(os.path.realpath(__file__))
_LIB_CLIENT_PATH = os.path.join(_LIB_DIR, 'libclient.so')
_LIB_PATH = os.path.join(_LIB_DIR, 'libtf_euler.so')

tf.load_op_library(_LIB_CLIENT_PATH)
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
  return _LIB.CreateGraph(config)


def initialize_embedded_graph(directory, graph_type='compact'):
  return initialize_graph({'mode': 'Local',
                           'directory': directory,
                           'load_type': graph_type})


# TODO: Consider lower the concept of shared graph to euler client.
def initialize_shared_graph(directory, zk_addr, zk_path, shard_idx, shard_num,
                            global_sampler_type='node', graph_type='compact',
                            server_thread_num=4):
  hdfs_prefix = 'hdfs://'
  if not directory.startswith(hdfs_prefix):
    raise ValueError('Only hdfs graph data is support for shared graph.')
  directory = directory[len(hdfs_prefix):]

  hdfs_addr = directory[:directory.index(':')]
  directory = directory[len(hdfs_addr):]
  directory = directory[len(':'):]

  hdfs_port = directory[:directory.index('/')]
  directory = directory[len(hdfs_port):]

  euler.start(directory=directory,
              loader_type='hdfs',
              hdfs_addr=hdfs_addr,
              hdfs_port=hdfs_port,
              shard_idx=shard_idx,
              shard_num=shard_num,
              zk_addr=zk_addr,
              zk_path=zk_path,
              global_sampler_type=global_sampler_type,
              graph_type=graph_type,
              server_thread_num=server_thread_num)

  return initialize_graph({'mode': 'Remote',
                           'zk_server': zk_addr,
                           'zk_path': zk_path})
