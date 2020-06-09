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

import tensorflow as tf
import tf_euler

from euler_estimator import EdgeEstimator
from transR import TransR

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def define_network_flags():
    tf.flags.DEFINE_string('dataset', 'fb15k', 'Dataset name.')
    tf.flags.DEFINE_integer('entity_embedding_dim', 100,
                            'Entity embedding dimension.')
    tf.flags.DEFINE_integer('relation_embedding_dim', 100,
                            'Relation embedding dimension.')
    tf.flags.DEFINE_integer('num_negs', 1, 'Number of negative samplings.')
    tf.flags.DEFINE_enum('corrupt', 'both', ['both', 'front', 'tail'],
                         'Corrupt triplets: front/tail/both')
    tf.flags.DEFINE_float('margin', 1., 'Margin of loss.')
    tf.flags.DEFINE_boolean('L1', False, 'Use l1 distance for score.')
    tf.flags.DEFINE_enum('metric_name', 'mrr', ['mrr', 'mr', 'hit10'],
                         'Metric name for valid: mrr/mr/hit10.')
    tf.flags.DEFINE_integer('batch_size', 128, 'Mini-batch size')
    tf.flags.DEFINE_integer('num_epochs', 4000, 'Epochs to train')
    tf.flags.DEFINE_integer('log_steps', 100, 'Number of steps to print log.')
    tf.flags.DEFINE_enum('infer_type', 'edge', 
                         ['edge', 'node_src', 'node_dst'],
                         'Infer type.')
    tf.flags.DEFINE_string('model_dir', 'ckpt', 'Model checkpoint.')
    tf.flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
    tf.flags.DEFINE_enum('optimizer', 'adam', ['adam', 'adagrad', 'sgd', 'momentum'],
                         'Optimizer algorithm')
    tf.flags.DEFINE_enum('run_mode', 'train', ['train', 'evaluate', 'infer'],
                         'Run mode.')


def main(_):
    flags_obj = tf.flags.FLAGS

    euler_graph = tf_euler.dataset.get_dataset(flags_obj.dataset)
    euler_graph.load_graph()
    num_steps = int((euler_graph.total_size + 1) // flags_obj.batch_size *
                    flags_obj.num_epochs)
    if flags_obj.run_mode == 'train':
        num_negs = flags_obj.num_negs
    else:
        num_negs = euler_graph.max_node_id


    model = TransR(
        node_type=euler_graph.train_node_type[0],
        edge_type=euler_graph.train_edge_type,
        node_max_id=euler_graph.max_node_id,
        edge_max_id=euler_graph.max_edge_id,
        ent_dim=flags_obj.entity_embedding_dim,
        rel_dim=flags_obj.relation_embedding_dim,
        num_negs=num_negs,
        margin=flags_obj.margin,
        l1=flags_obj.L1,
        metric_name=flags_obj.metric_name,
        corrupt=flags_obj.corrupt)

    params = {'train_edge_type': euler_graph.train_edge_type[0],
              'batch_size': flags_obj.batch_size,
              'optimizer': flags_obj.optimizer,
              'learning_rate': flags_obj.learning_rate,
              'log_steps': flags_obj.log_steps,
              'model_dir': flags_obj.model_dir,
              'id_file': euler_graph.edge_id_file,
              'infer_dir': flags_obj.model_dir,
              'infer_type': flags_obj.infer_type,
              'total_step': num_steps}
    config = tf.estimator.RunConfig(log_step_count_steps=None)
    model_estimator = EdgeEstimator(model, params, config)

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
