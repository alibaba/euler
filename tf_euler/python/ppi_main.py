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

import tensorflow as tf

from tf_euler.python import run_loop
from tf_euler.python.utils import flags as utils_flags


def define_ppi_flags():
  run_loop.define_network_embedding_flags()
  tf.flags.adopt_module_key_flags(run_loop)
  utils_flags.set_defaults(
      data_dir='examples/ppi',
      max_id=56944,
      feature_idx=1,
      feature_dim=50,
      label_idx=0,
      label_dim=121,
      num_classes=121,
      id_file=['examples/ppi/ppi_test.id'])


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_ppi_flags()
  tf.app.run(run_loop.main)
