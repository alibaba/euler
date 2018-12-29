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

import os

import numpy as np
import tensorflow as tf

from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python import models
from tf_euler.python import optimizers
from tf_euler.python.utils import context as utils_context
from tf_euler.python.utils import hooks as utils_hooks
from euler.python import service


def define_network_embedding_flags():
  tf.flags.DEFINE_enum('mode', 'train',
                       ['train', 'evaluate', 'save_embedding'], 'Run mode.')

  tf.flags.DEFINE_string('data_dir', '', 'Local Euler graph data.')
  tf.flags.DEFINE_integer('train_node_type', 0, 'Node type of training set.')
  tf.flags.DEFINE_integer('all_node_type', euler_ops.ALL_NODE_TYPE,
                          'Node type of the whole graph.')
  tf.flags.DEFINE_list('train_edge_type', [0], 'Edge type of training set.')
  tf.flags.DEFINE_list('all_edge_type', [0, 1],
                       'Edge type of the whole graph.')
  tf.flags.DEFINE_integer('max_id', -1, 'Max node id.')
  tf.flags.DEFINE_integer('feature_idx', -1, 'Feature index.')
  tf.flags.DEFINE_integer('feature_dim', 0, 'Feature dimension.')
  tf.flags.DEFINE_integer('label_idx', -1, 'Label index.')
  tf.flags.DEFINE_integer('label_dim', 0, 'Label dimension.')
  tf.flags.DEFINE_integer('num_classes', None, 'Number of classes.')
  tf.flags.DEFINE_list('id_file', [], 'Files containing ids to evaluate.')

  tf.flags.DEFINE_string('model', 'graphsage_supervised', 'Embedding model.')
  tf.flags.DEFINE_boolean('sigmoid_loss', True, 'Whether to use sigmoid loss.')
  tf.flags.DEFINE_boolean('xent_loss', True, 'Whether to use xent loss.')
  tf.flags.DEFINE_integer('dim', 256, 'Dimension of embedding.')
  tf.flags.DEFINE_integer('order', 1, 'LINE order.')
  tf.flags.DEFINE_list('fanouts', [10, 10], 'GCN fanouts.')
  tf.flags.DEFINE_enum('aggregator', 'mean',
                       ['gcn', 'mean', 'meanpool', 'maxpool'],
                       'Sage aggregator.')
  tf.flags.DEFINE_boolean('concat', True, 'Sage aggregator concat.')
  tf.flags.DEFINE_integer('head_num', 1, 'multi head attention num')

  tf.flags.DEFINE_string('model_dir', 'ckpt', 'Model checkpoint.')
  tf.flags.DEFINE_integer('batch_size', 512, 'Mini-batch size.')
  tf.flags.DEFINE_string('optimizer', 'adam', 'Optimizer to use.')
  tf.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
  tf.flags.DEFINE_integer('num_epochs', 20, 'Number of epochs for training.')
  tf.flags.DEFINE_integer('log_steps', 20, 'Number of steps to print log.')

  tf.flags.DEFINE_list('ps_hosts', [], 'Parameter servers.')
  tf.flags.DEFINE_list('worker_hosts', [], 'Training workers.')
  tf.flags.DEFINE_string('job_name', '', 'Cluster role.')
  tf.flags.DEFINE_integer('task_index', 0, 'Task index.')

  tf.flags.DEFINE_string('euler_zk_addr', '127.0.0.1:2181',
                         'Euler ZK registration service.')
  tf.flags.DEFINE_string('euler_zk_path', '/tf_euler',
                         'Euler ZK registration node.')


def run_train(model, flags_obj, master, is_chief):
  utils_context.training = True

  batch_size = flags_obj.batch_size // model.batch_size_ratio
  if flags_obj.model == 'line' or flags_obj.model == 'randomwalk':
    source = euler_ops.sample_node(
        count=batch_size, node_type=flags_obj.all_node_type)
  else:
    source = euler_ops.sample_node(
        count=batch_size, node_type=flags_obj.train_node_type)
  source.set_shape([batch_size])
  # dataset = tf.data.TextLineDataset(flags_obj.id_file)
  # dataset = dataset.map(
  #     lambda id_str: tf.string_to_number(id_str, out_type=tf.int64))
  # dataset = dataset.shuffle(buffer_size=20000)
  # dataset = dataset.batch(batch_size)
  # dataset = dataset.repeat(flags_obj.num_epochs)
  # source = dataset.make_one_shot_iterator().get_next()
  _, loss, metric_name, metric = model(source)

  optimizer_class = optimizers.get(flags_obj.optimizer)
  optimizer = optimizer_class(learning_rate=flags_obj.learning_rate)
  global_step = tf.train.get_or_create_global_step()
  train_op = optimizer.minimize(loss, global_step=global_step)

  hooks = []

  tensor_to_log = {'step': global_step, 'loss': loss, metric_name: metric}
  hooks.append(
      tf.train.LoggingTensorHook(
          tensor_to_log, every_n_iter=flags_obj.log_steps))

  num_steps = int((flags_obj.max_id + 1) // batch_size * flags_obj.num_epochs)
  hooks.append(tf.train.StopAtStepHook(last_step=num_steps))

  if len(flags_obj.worker_hosts) == 0 or flags_obj.task_index == 1:
    hooks.append(
        tf.train.ProfilerHook(save_secs=180, output_dir=flags_obj.model_dir))
  if len(flags_obj.worker_hosts):
    hooks.append(utils_hooks.SyncExitHook(len(flags_obj.worker_hosts)))
  if hasattr(model, 'make_session_run_hook'):
    hooks.append(model.make_session_run_hook())

  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=is_chief,
      checkpoint_dir=flags_obj.model_dir,
      log_step_count_steps=None,
      hooks=hooks) as sess:
    while not sess.should_stop():
      sess.run(train_op)


def run_evaluate(model, flags_obj):
  utils_context.training = False

  dataset = tf.data.TextLineDataset(flags_obj.id_file)
  dataset = dataset.map(
      lambda id_str: tf.string_to_number(id_str, out_type=tf.int64))
  dataset = dataset.batch(flags_obj.batch_size)
  source = dataset.make_one_shot_iterator().get_next()
  _, _, metric_name, metric = model(source)

  tf.train.get_or_create_global_step()

  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=flags_obj.model_dir,
      save_checkpoint_secs=None,
      log_step_count_steps=None) as sess:
    while not sess.should_stop():
      metric_val = sess.run(metric)

  print('{}: {}'.format(metric_name, metric_val))


def run_save_embedding(model, flags_obj):
  utils_context.training = False

  dataset = tf.data.Dataset.range(flags_obj.max_id + 1)
  dataset = dataset.batch(flags_obj.batch_size)
  source = dataset.make_one_shot_iterator().get_next()
  embedding, _, _, _ = model(source)

  tf.train.get_or_create_global_step()

  embedding_vals = []
  with tf.train.MonitoredTrainingSession(
      checkpoint_dir=flags_obj.model_dir,
      save_checkpoint_secs=None,
      log_step_count_steps=None) as sess:
    while not sess.should_stop():
      embedding_val = sess.run(embedding)
      embedding_vals.append(embedding_val)

  embedding_val = np.concatenate(embedding_vals)
  np.save(os.path.join(flags_obj.model_dir, 'embedding.npy'), embedding_val)
  with open(os.path.join(flags_obj.model_dir, 'id.txt'), 'w') as fp:
    fp.write('\n'.join(map(str, range(flags_obj.max_id + 1))))


def run_network_embedding(flags_obj, master, is_chief):
  fanouts = map(int, flags_obj.fanouts)
  if flags_obj.mode == 'train':
    metapath = [map(int, flags_obj.train_edge_type)] * len(fanouts)
  else:
    metapath = [map(int, flags_obj.all_edge_type)] * len(fanouts)

  if flags_obj.model == 'line':
    model = models.LINE(
        node_type=flags_obj.all_node_type,
        edge_type=flags_obj.all_edge_type,
        max_id=flags_obj.max_id,
        dim=flags_obj.dim,
        xent_loss=flags_obj.xent_loss,
        order=flags_obj.order)

  elif flags_obj.model == 'randomwalk':
    model = models.Node2Vec(
        node_type=flags_obj.all_node_type,
        edge_type=flags_obj.all_edge_type,
        max_id=flags_obj.max_id,
        dim=flags_obj.dim,
        xent_loss=flags_obj.xent_loss,
        num_negs=5,
        walk_len=3,
        walk_p=1,
        walk_q=1,
        left_win_size=1,
        right_win_size=1)

  elif flags_obj.model == 'graphsage':
    model = models.GraphSage(
        node_type=flags_obj.train_node_type,
        edge_type=flags_obj.train_edge_type,
        max_id=flags_obj.max_id,
        xent_loss=flags_obj.xent_loss,
        metapath=metapath,
        fanouts=fanouts,
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,
        concat=flags_obj.concat,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim)

  elif flags_obj.model == 'graphsage_supervised':
    model = models.SupervisedGraphSage(
        label_idx=flags_obj.label_idx,
        label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss,
        metapath=metapath,
        fanouts=fanouts,
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator,
        concat=flags_obj.concat,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim)

  elif flags_obj.model == 'scalable_gcn':
    model = models.ScalableGCN(
        label_idx=flags_obj.label_idx, label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes, sigmoid_loss=flags_obj.sigmoid_loss,
        edge_type=metapath[0], fanout=fanouts[0], num_layers=len(fanouts),
        dim=flags_obj.dim,
        aggregator=flags_obj.aggregator, concat=flags_obj.concat,
        feature_idx=flags_obj.feature_idx, feature_dim=flags_obj.feature_dim,
        max_id=flags_obj.max_id)

  elif flags_obj.model == 'gat':
    model = models.GAT(
        label_idx=flags_obj.label_idx,
        label_dim=flags_obj.label_dim,
        num_classes=flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss,
        feature_idx=flags_obj.feature_idx,
        feature_dim=flags_obj.feature_dim,
        max_id=flags_obj.max_id,
        head_num=flags_obj.head_num,
        hidden_dim=flags_obj.dim,
        nb_num=5)

  elif flags_obj.model == 'lshne':
    model = models.LsHNE(-1,[[0,0,0],[0,0,0]],-1,128,[1,1],[1,1])

  elif flags_obj.model == 'saved_embedding':
    embedding_val = np.load(os.path.join(flags_obj.model_dir, 'embedding.npy'))
    embedding = layers.Embedding(
        max_id=flags_obj.max_id,
        dim=flags_obj.dim,
        initializer=lambda: tf.constant_initializer(embedding_val))
    model = models.SupervisedModel(
        flags_obj.label_idx,
        flags_obj.label_dim,
        flags_obj.num_classes,
        sigmoid_loss=flags_obj.sigmoid_loss)
    model.encoder = lambda inputs: tf.stop_gradient(embedding(inputs))

  else:
    raise ValueError('Unsupported network embedding model.')

  if flags_obj.mode == 'train':
    run_train(model, flags_obj, master, is_chief)
  elif flags_obj.mode == 'evaluate':
    run_evaluate(model, flags_obj)
  elif flags_obj.mode == 'save_embedding':
    run_save_embedding(model, flags_obj)


def run_local(flags_obj, run):
  if not euler_ops.initialize_embedded_graph(flags_obj.data_dir):
    raise RuntimeError('Failed to initialize graph.')

  run(flags_obj, master='', is_chief=True)


def run_distributed(flags_obj, run):
  cluster = tf.train.ClusterSpec({
      'ps': flags_obj.ps_hosts,
      'worker': flags_obj.worker_hosts
  })
  server = tf.train.Server(
      cluster, job_name=flags_obj.job_name, task_index=flags_obj.task_index)

  if flags_obj.job_name == 'ps':
    server.join()
  elif flags_obj.job_name == 'worker':
    if not euler_ops.initialize_shared_graph(
        directory=flags_obj.data_dir,
        zk_addr=flags_obj.euler_zk_addr,
        zk_path=flags_obj.euler_zk_path,
        shard_idx=flags_obj.task_index,
        shard_num=len(flags_obj.worker_hosts),
        global_sampler_type='node'):
      raise RuntimeError('Failed to initialize graph.')

    with tf.device(
        tf.train.replica_device_setter(
            worker_device='/job:worker/task:%d' % flags_obj.task_index,
            cluster=cluster)):
      run(flags_obj, server.target, flags_obj.task_index == 0)
  else:
    raise ValueError('Unsupport role: {}'.format(flags_obj.job_name))


def main(_):
  flags_obj = tf.flags.FLAGS
  if flags_obj.worker_hosts:
    run_distributed(flags_obj, run_network_embedding)
  else:
    run_local(flags_obj, run_network_embedding)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_network_embedding_flags()
  tf.app.run(main)
