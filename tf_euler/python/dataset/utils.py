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

from tf_euler.python.dataset.ppi import ppi
from tf_euler.python.dataset.reddit import reddit
from tf_euler.python.dataset.test_data import test_data
from tf_euler.python.dataset.cora import cora
from tf_euler.python.dataset.pubmed import pubmed
from tf_euler.python.dataset.citeseer import citeseer
from tf_euler.python.dataset.fb15k import FB15K
from tf_euler.python.dataset.fb15k237 import FB15K237
from tf_euler.python.dataset.wn18 import WN18
from tf_euler.python.dataset.mutag import MUTAG
from tf_euler.python.dataset.ml_1m import MovieLens_1M


def get_dataset(data_name):
    data_name = data_name.lower()
    if data_name == 'ppi':
        print('dataset is ppi')
        return ppi()
    elif data_name == 'reddit':
        print('dataset is reddit')
        return reddit()
    elif data_name == 'test_data':
        print('dataset is test_data')
        return test_data()
    elif data_name == 'cora':
        print('dataset is cora')
        return cora()
    elif data_name == 'pubmed':
        print('dataset is pubmed')
        return pubmed()
    elif data_name == 'citeseer':
        print('dataset is citeseer')
        return citeseer()
    elif data_name == 'fb15k':
        print('dataset is fb15k')
        return FB15K()
    elif data_name == 'fb15k-237':
        print('dataset is fb15k-237')
        return FB15K237()
    elif data_name == 'wn18':
        print('dataset is wn18')
        return WN18()
    elif data_name == 'mutag':
        print('dataset is mutag')
        return MUTAG()
    elif data_name == 'movielens-1m':
        print('dataset is movielens-1m')
        return MovieLens_1M()
    else:
        raise RuntimeError('Failed to get dataset. \
              Dataset name must be one of \
              [ppi/reddit/cora/pubmed/citeseer/test_data/fb15k/MUTAG]')
