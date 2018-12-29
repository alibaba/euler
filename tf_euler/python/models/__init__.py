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

from tf_euler.python.models.base import SupervisedModel
from tf_euler.python.models.lasgnn import LasGNN
from tf_euler.python.models.line import LINE
from tf_euler.python.models.node2vec import Node2Vec
from tf_euler.python.models.graphsage import GraphSage, SupervisedGraphSage, \
                                             ScalableGCN
from tf_euler.python.models.gat import GAT
from tf_euler.python.models.lshne import LsHNE
