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

import struct
import ctypes
import os
import threading

_LIB_DIR = os.path.dirname(os.path.realpath(__file__))
ctypes.CDLL(os.path.join(_LIB_DIR, 'libcommon.so'))
_LIB_NAME = 'libeuler_util.so'
_LIB_PATH = os.path.join(_LIB_DIR, _LIB_NAME)
_LIB = ctypes.CDLL(_LIB_PATH)
hash64 = _LIB.py_hash64
hash64.restype = ctypes.c_ulonglong


def edgeIdHash(src_id, dst_id, etype):
    s = struct.pack('L', src_id)
    s += struct.pack('L', dst_id)
    s += struct.pack('i', etype)
    r = hash64(s, len(s))
    return r


def expend_list(lis, idx):
    idx += 1
    if idx > len(lis):
        for i in range(len(lis), idx):
            lis.append([])


def expend_type(lis, idx):
    idx += 1
    if idx > len(lis):
        for i in range(len(lis), idx):
            lis.append(0)


def convert_feature(features):
    feature_idx = []
    feature = []
    idx = 0
    for t in features:
        idx += len(t)
        feature_idx.append(idx)
        for f in t:
            feature.append(f)
    return feature_idx, feature


def write_string(ss):
    if not isinstance(ss, bytes):
        ss = ss.encode()
    b = str(len(ss)) + 's'
    s = struct.pack('I', len(ss))
    return s + struct.pack(b, ss)


def read_string(s):
    slen = struct.unpack('I', s[:4])[0]
    s = s[4:]
    b = str(slen) + 's'
    ss = struct.unpack(b, s[:slen])[0]
    s = s[slen:]
    if isinstance(ss, bytes):
        ss = ss.decode('utf-8')
    return ss, s


def write_correct_data(valueType, value):
    if valueType == 'string':
        s = write_string(value)
    elif valueType == 'float':
        s = struct.pack('f', value)
    elif valueType == 'int32_t':
        s = struct.pack('i', value)
    elif valueType == 'uint32_t':
        s = struct.pack('I', value)
    elif valueType == 'int64_t':
        s = struct.pack('l', value)
    elif valueType == 'uint64_t':
        s = struct.pack('L', value)
    elif valueType == 'featureType':
        if value == 'sparse':
            s = struct.pack('i', 0)
        elif value == 'dense':
            s = struct.pack('i', 1)
        elif value == 'binary':
            s = struct.pack('i', 2)
        else:
            s = struct.pack('i', 3)
    else:
        print('not support this valueType ' + valueType)
        exit(1)
    return s


def read_data(valueType, value):
    if valueType == 'string':
        s, value = read_string(value)
    elif valueType == 'uint64_t':
        s = struct.unpack('L', value[:8])[0]
        value = value[8:]
    elif valueType == 'int64_t':
        s = struct.unpack('l', value[:8])[0]
        value = value[8:]
    elif valueType == 'uint32_t':
        s = struct.unpack('I', value[:4])[0]
        value = value[4:]
    elif valueType == 'int32_t':
        s = struct.unpack('i', value[:4])[0]
        value = value[4:]
    elif valueType == 'featureType':
        t = struct.unpack('i', value[:4])[0]
        value = value[4:]
        if t == 0:
            s = 'sparse'
        elif t == 1:
            s = 'dense'
        elif t == 2:
            s = 'binary'
        else:
            s = 'unk'
    return s, value


def write_list(lis, valueType):
    s = struct.pack('I', len(lis))
    for i in lis:
        s += write_correct_data(valueType, i)
    return s
