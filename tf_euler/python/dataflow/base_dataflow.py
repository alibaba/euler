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


class Block(object):
    def __init__(self, n_id, res_n_id, e_id, edge_index, size):
        self.n_id = n_id
        self.res_n_id = res_n_id
        self.e_id = e_id
        self.edge_index = edge_index
        self.size = size


class DataFlow(object):
    def __init__(self, n_id):
        self.n_id = n_id
        self.__last_n_id__ = n_id
        self.blocks = []

    def append(self, n_id, res_n_id, e_id, edge_index):
        size = [tf.shape(self.__last_n_id__)[0], tf.shape(n_id)[0]]
        block = Block(n_id, res_n_id, e_id, edge_index, size)
        self.blocks.append(block)
        self.__last_n_id__ = n_id

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return self.blocks[::-1][idx]

    def __iter__(self):
        for block in self.blocks[::-1]:
            yield block
