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

import os
import json
import numpy as np

import tensorflow as tf
import tf_euler


class BaseEstimator(object):

    def __init__(self, model_fn, params, run_config, profiling=False):
        self.model_function = model_fn
        self.params = params
        self.run_config = run_config
        self.evaluate_stop_onetime = False
        self.profiling = profiling

    def _model_fn(self, features, mode, params):
        model = self.model_function
        if mode == tf.estimator.ModeKeys.TRAIN:
            spec = self.train_model_init(model, features, mode, params)
        elif mode == tf.estimator.ModeKeys.EVAL:
            spec = self.evaluate_model_init(model, features, mode, params)
        else:
            spec = self.infer_model_init(model, features, mode, params)
        return spec

    def get_infer_from_input(self, inputs, params):
        return inputs

    def transfer_embedding(self, source, emb):
        return source, emb

    def infer_model_init(self, model, features, mode, params):
        source = self.get_infer_from_input(features, params)
        emb, loss, _, _ = model(source)
        source, emb = self.transfer_embedding(source, emb)
        prediction = {'idx': source, 'embedding': emb}
        spec = tf.estimator.EstimatorSpec(mode=mode, predictions=prediction)
        return spec

    def get_evaluate_from_input(self, inputs, params):
        return inputs

    def evaluate_model_init(self, model, features, mode, params):
        source = self.get_evaluate_from_input(features, params)
        _, loss, metric_name, metric = model(source)
        tf.train.get_or_create_global_step()
        tensor_to_log = {'loss': loss, metric_name: metric}
        hooks = []
        hooks.append(
                tf.train.LoggingTensorHook(
                    tensor_to_log, every_n_iter=1))
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          evaluation_hooks=hooks)
        return spec

    def get_train_from_input(self, inputs, params):
        return inputs

    def train_model_init(self, model, features, mode, params):
        source = self.get_train_from_input(features, params)
        _, loss, metric_name, metric = model(source)
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf_euler.utils.optimizers.get(
                             params.get('optimizer', 'adam'))(
                             params.get('learning_rate', 0.001))
        train_op = optimizer.minimize(loss, global_step)
        hooks = []
        tensor_to_log = {'step': global_step,
                         'loss': loss,
                         metric_name: metric}
        hooks.append(
                tf.train.LoggingTensorHook(
                    tensor_to_log, every_n_iter=params.get('log_steps', 100)))
        spec = tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=hooks)
        return spec

    def train_and_evaluate(self):
        estimator = tf.estimator.Estimator(
                model_fn=self._model_fn,
                params=self.params,
                config=self.run_config,
                model_dir=self.params['model_dir'])
        total_step = None
        try:
            total_step = self.params['total_step']
        except:
            total_step = None
        train_spec = tf.estimator.TrainSpec(
            input_fn=self.train_input_fn,
            max_steps=total_step)
        run_steps = None
        if self.evaluate_stop_onetime:
            run_steps = 1
        eval_spec = tf.estimator.EvalSpec(input_fn=self.evaluate_input_fn,
                                          steps=run_steps)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    def train(self):
        estimator = tf.estimator.Estimator(
                model_fn=self._model_fn,
                params=self.params,
                config=self.run_config,
                model_dir=self.params['model_dir'])

        if self.profiling:
            hooks = [tf.train.ProfilerHook(50, output_dir="prof_dir")]
        else:
            hooks = []
        print (self.profiling, hooks)
        total_step = None
        try:
            total_step = self.params['total_step']
        except:
            total_step = None
        estimator.train(input_fn=self.train_input_fn,
                        hooks=hooks,
                        #steps=self.params['total_step'])
                        steps=total_step)

    def evaluate(self):
        estimator = tf.estimator.Estimator(
                model_fn=self._model_fn,
                params=self.params,
                config=self.run_config,
                model_dir=self.params['model_dir'])
        run_steps = None
        if self.evaluate_stop_onetime:
            run_steps = 1
        estimator.evaluate(input_fn=self.evaluate_input_fn,
                           steps=run_steps)

    def infer(self):
        estimator = tf.estimator.Estimator(
                model_fn=self._model_fn,
                params=self.params,
                config=self.run_config,
                model_dir=self.params['model_dir'])
        try:
            worker_idx = json.loads(os.environ['TF_CONFIG']['task'])
        except Exception:
            worker_idx = 0
        out_idxs = []
        out_embeddings = []
        for output in estimator.predict(input_fn=self.evaluate_input_fn):
            out_idxs.append(output['idx'])
            out_embeddings.append(output['embedding'])
        out_embeddings = np.asarray(out_embeddings)
        out_idxs = np.asarray(out_idxs)
        with open(os.path.realpath(os.path.join(self.params['infer_dir'],
                  'embedding_{}.npy'.format(worker_idx))), 'wb') as emb_file:
            np.save(emb_file, out_embeddings)
        with open(os.path.realpath(os.path.join(self.params['infer_dir'],
                  'ids_{}.npy'.format(worker_idx))), 'wb') as ids_file:
            np.save(ids_file, out_idxs)

    def train_input_fn(self):
        raise NotImplementedError

    def evaluate_input_fn(self):
        raise NotImplementedError

    def infer_input_fn(self):
        raise NotImplementedError
