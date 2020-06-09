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
import threading
import sys
import time
import multiprocessing

_LIB_DIR = os.path.dirname(os.path.realpath(__file__))
_LIB_NAME = 'libeuler_core.so'
_LIB_PATH = os.path.join(_LIB_DIR, _LIB_NAME)
_LIB = ctypes.CDLL(_LIB_PATH)


class Module(object):
    NODE = 1
    EDGE = 2
    NODE_SAMPLER = 4
    EDGE_SAMPLER = 8
    DEFAULT_MODULE = NODE | NODE_SAMPLER

    @classmethod
    def to_load_data_type_string(cls, module):
        data_module = module & (cls.NODE | cls.EDGE)
        if data_module == 0:
            return 'none'
        elif data_module == cls.NODE:
            return 'node'
        elif data_module == cls.EDGE:
            return 'edge'
        elif data_module == cls.NODE | cls.EDGE:
            return 'all'
        else:
            assert False

    @classmethod
    def to_global_sampler_type_string(cls, module):
        data_module = module & (cls.NODE_SAMPLER | cls.EDGE_SAMPLER)
        if data_module == 0:
            return 'none'
        elif data_module == cls.NODE_SAMPLER:
            return 'node'
        elif data_module == cls.EDGE_SAMPLER:
            return 'edge'
        elif data_module == cls.NODE_SAMPLER | cls.EDGE_SAMPLER:
            return 'all'
        else:
            assert False


def start(directory='',
          shard_idx=0,
          shard_num=1,
          zk_addr='',
          zk_path='',
          module=Module.DEFAULT_MODULE,
          server_thread_num=multiprocessing.cpu_count()):
    return _LIB.StartService(str(directory), str(shard_idx),
                             str(shard_num), str(zk_addr), str(zk_path),
                             Module.to_load_data_type_string(module),
                             Module.to_global_sampler_type_string(module),
                             str(server_thread_num))
