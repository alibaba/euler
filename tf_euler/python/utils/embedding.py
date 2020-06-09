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
from tensorflow.python.ops import variables


def embedding_update(params,
                     ids,
                     values,
                     partition_strategy='mod',
                     func=tf.scatter_update,
                     name=None):
    if isinstance(params, variables.PartitionedVariable):
        params = list(params)
    if not isinstance(params, list):
        params = [params]

    params = list(params)
    np = len(params)

    if np == 1:
        with tf.colocate_with(params[0]):
            return func(params[0], ids, values)

    if partition_strategy == 'mod':
        p_assignments = ids % np
        new_ids = ids // np
    else:
        raise ValueError('Unrecognized partition strategy: ' +
                         partition_strategy)

    p_assignments = tf.to_int32(p_assignments)
    scatter_ids = tf.dynamic_partition(new_ids, p_assignments, np)
    scatter_values = tf.dynamic_partition(values, p_assignments, np)

    update_ops = []
    for param, pids, pvalues in zip(params, scatter_ids, scatter_values):
        with tf.colocate_with(param):
            update_ops.append(func(param, pids, pvalues))

    return tf.group(*update_ops)


def embedding_add(params,
                  ids,
                  values,
                  partition_strategy='mod',
                  name=None):
    return embedding_update(params, ids, values,
                            partition_strategy=partition_strategy,
                            func=tf.scatter_add, name=name)
