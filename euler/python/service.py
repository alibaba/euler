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
import threading

_LIB_DIR = os.path.dirname(os.path.realpath(__file__))
_LIB_NAME = 'libeuler_service.so'
_LIB_PATH = os.path.join(_LIB_DIR, _LIB_NAME)
_LIB = ctypes.CDLL(_LIB_PATH)


def start(directory='',
          loader_type='local',
          hdfs_addr='',
          hdfs_port=0,
          shard_idx=0,
          shard_num=1,
          zk_addr='',
          zk_path='',
          global_sampler_type='node',
          graph_type='compact',
          server_thread_num=''):
  euler = threading.Thread(target=_LIB.StartService,
                           args=(str(directory), str(loader_type),
                                 str(hdfs_addr), str(hdfs_port),
                                 str(shard_idx), str(shard_num),
                                 str(zk_addr), str(zk_path),
                                 str(global_sampler_type), str(graph_type),
                                 str(server_thread_num)))
  euler.daemon = True
  euler.start()
  return euler

def start_and_wait(directory='',
                   loader_type='local',
                   hdfs_addr='',
                   hdfs_port=0,
                   shard_idx=0,
                   shard_num=1,
                   zk_addr='',
                   zk_path='',
                   global_sampler_type='node',
                   graph_type='compact',
                   server_thread_num=''):
  return _LIB.StartService(str(directory), str(loader_type),
                           str(hdfs_addr), str(hdfs_port),
                           str(shard_idx), str(shard_num),
                           str(zk_addr), str(zk_path),
                           str(global_sampler_type), str(graph_type),
                           str(server_thread_num))
