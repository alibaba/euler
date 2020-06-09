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

_LIB_DIR = os.path.dirname(os.path.realpath(__file__))
_LIB_NAME = 'libeuler_util.so'
_LIB_PATH = os.path.join(_LIB_DIR, _LIB_NAME)
_LIB = ctypes.CDLL(_LIB_PATH)

print (_LIB.py_hash64('1234', len('1234')))
