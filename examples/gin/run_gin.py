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
import tf_euler

from gin import GIN
from euler_estimator import GraphEstimator

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def define_network_flags():
    tf.flags.DEFINE_string('dataset', 'mutag', 'Dataset name.')
    tf.flags.DEFINE_integer('hidden_dim', 32, 'Hidden dimension.')
    tf.flags.DEFINE_integer('layers', 5, 'GCN convolution layer number.')
    tf.flags.DEFINE_boolean('train_eps', False, 'If eps trainable.')
    tf.flags.DEFINE_float('eps', 0.0, 'GIN eps init.')
    tf.flags.DEFINE_integer('batch_size', 160, 'Mini-batch size')
    tf.flags.DEFINE_integer('num_epochs', 500, 'Epochs to train')
    tf.flags.DEFINE_integer('log_steps', 20, 'Number of steps to print log.')
    tf.flags.DEFINE_string('model_dir', 'ckpt', 'Model checkpoint.')
    tf.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
    tf.flags.DEFINE_enum('optimizer', 'adam', ['adam', 'adagrad', 'sgd', 'momentum'],
                         'Optimizer algorithm')
    tf.flags.DEFINE_enum('run_mode', 'train', ['train', 'evaluate', 'infer'],
                         'Run mode.')


def main(_):
    flags_obj = tf.flags.FLAGS
    euler_graph = tf_euler.dataset.get_dataset(flags_obj.dataset)
    euler_graph.load_graph()

    dims = [flags_obj.hidden_dim] * (flags_obj.layers + 1)
    if flags_obj.run_mode == 'train':
        metapath = [euler_graph.train_edge_type] * flags_obj.layers
    else:
        metapath = [euler_graph.all_edge_type] * flags_obj.layers
    num_steps = int((euler_graph.total_size + 1) // flags_obj.batch_size *
                    flags_obj.num_epochs)


    model = GIN(dims, metapath, euler_graph.num_classes,
                euler_graph.sparse_fea_idx,
                euler_graph.sparse_fea_max_id,
                eps=flags_obj.eps,
                train_eps=flags_obj.train_eps)

    params = {
              'num_classes': euler_graph.num_classes,
              'optimizer': flags_obj.optimizer,
              'learning_rate': flags_obj.learning_rate,
              'log_steps': flags_obj.log_steps,
              'train_rate': euler_graph.train_rate,
              'id_file': euler_graph.id_file,
              'label': ['label'],
              'model_dir': flags_obj.model_dir,
              'total_size': euler_graph.total_size,
              'infer_dir': flags_obj.model_dir,
              'batch_size': flags_obj.batch_size,
              'total_step': num_steps}

    config = tf.estimator.RunConfig(log_step_count_steps=None)
    model_estimator = GraphEstimator(model, params, config)

    if flags_obj.run_mode == 'train':
        model_estimator.train()
    elif flags_obj.run_mode == 'evaluate':
        model_estimator.evaluate()
    elif flags_obj.run_mode == 'infer':
        model_estimator.infer()
    else:
        raise ValueError('Run mode not exist!')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    define_network_flags()
    tf.app.run(main)
