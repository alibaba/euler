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

from tf_euler.python.solution.base_sample import SuperviseSampleSolution
from tf_euler.python.mp_utils.base_gnn import BaseGNNNet
from euler_estimator import SampleEstimator


# load euler graph, cora for example
euler_graph = tf_euler.dataset.get_dataset('cora')
euler_graph.load_graph()

# define model param
conv = 'sage'
dataflow = 'sage'
fanouts = [5, 5]
dims = [32] * (len(fanouts) + 1)
metapath = [['train'], ['train']]
feature_idx = ['feature']
feature_dim = [1433]

# define estimator param
params = {'batch_size': 512,
          'optimizer': 'adam',
          'learning_rate': 0.001,
          'log_steps': 20,
          'model_dir': 'ckpt',
          'sample_dir': 'sample.txt',
          'infer_dir': 'ckpt',
          'epoch': 20000}

# define encoder
class GNN(BaseGNNNet):

    def __init__(self, conv, flow,
                 dims, fanouts, metapath,
                 feature_idx, feature_dim,
                 add_self_loops=False):
        super(GNN, self).__init__(conv=conv,
                                  flow=flow,
                                  dims=dims,
                                  fanouts=fanouts,
                                  metapath=metapath,
                                  add_self_loops=add_self_loops)
        if not isinstance(feature_idx, list):
            feature_idx = [feature_idx]
        if not isinstance(feature_dim, list):
            feature_dim = [feature_dim]
        self.feature_idx = feature_idx
        self.feature_dim = feature_dim

    def to_x(self, n_id):
        x, = tf_euler.get_dense_feature(n_id,
                                        self.feature_idx,
                                        self.feature_dim)
        return x


my_encoder = GNN(conv, dataflow, dims, fanouts, metapath, feature_idx, feature_dim)

# define input parse function
def my_parse_fn(inputs):
    label = tf.string_to_number(inputs.values[0], out_type=tf.float32)
    input_node = tf.string_to_number(inputs.values[1], out_type=tf.int64)
    label = tf.reshape(label, [-1, 1])
    input_node = tf.reshape(input_node, [-1, 1])
    sample = [label, input_node]
    return sample

# define embedding parse function
def my_parse_emb_fn(embedding):
    return embedding, None, embedding

# define logit fn
my_logit_fn = tf_euler.solution.logits.DenseLogits(1)

# define model
model = SuperviseSampleSolution(my_parse_fn,
                                my_encoder,
                                my_parse_emb_fn,
                                logit_fn=my_logit_fn)

# define estimator
tf.logging.set_verbosity(tf.logging.INFO)
config = tf.estimator.RunConfig(log_step_count_steps=None)
model_estimator = SampleEstimator(model, params, config)

# run estimator
model_estimator.train()
