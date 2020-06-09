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

from tf_euler.python.dataflow.sage_dataflow import SageDataFlow
from tf_euler.python.dataflow.gcn_dataflow import GCNDataFlow
from tf_euler.python.dataflow.fast_dataflow import FastGCNDataFlow
from tf_euler.python.dataflow.layerwise_dataflow import \
    LayerwiseDataFlow, LayerwiseEachDataFlow
from tf_euler.python.dataflow.whole_dataflow import WholeDataFlow
from tf_euler.python.dataflow.relation_dataflow import RelationDataFlow
