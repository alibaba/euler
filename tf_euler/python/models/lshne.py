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

from tf_euler.python import euler_ops
from tf_euler.python import layers
from tf_euler.python.models import base
from tf_euler.python.models.base import ModelOutput

class LsHNE(base.UnsupervisedModel):
  """
  Model LsHNE with sparse feature, MultiView Version.
  """
  def __init__(self, node_type, path_patterns, max_id, dim,
          sparse_feature_dims, feature_ids,feature_embedding_dim=16,
          walk_len=3, left_win_size=1, right_win_size=1, num_negs=5, gamma=5,
          *args, **kwargs):
    super(LsHNE, self).__init__(node_type, path_patterns, max_id,
              *args, **kwargs)
    self.node_type = node_type
    self.path_patterns = path_patterns
    self.max_id = max_id
    self.dim = dim
    self.walk_len = walk_len
    self.left_win_size = left_win_size
    self.right_win_size = right_win_size
    self.num_negs = num_negs
    self.view_num = len(path_patterns)
    if self.view_num<1:
      raise ValueError('View Number must be bigger than 1, got{}'.format(self.view_num))
    if not isinstance(sparse_feature_dims, list):
      raise TypeError('Expect list for sparse feature dimsgot {}.'.format(
          type(sparse_feature_dims).__name__))
    self.sparse_feature_dims = sparse_feature_dims
    self.feature_ids = feature_ids
    self.feature_embedding_dim = feature_embedding_dim
    self.raw_fdim = feature_embedding_dim * len(feature_ids)
    self.feature_embedding_layer = []
    for d in sparse_feature_dims:
      self.feature_embedding_layer.append(layers.SparseEmbedding(d,feature_embedding_dim,
          combiner="sum"))

    self.hidden_layer =[{}] * self.view_num
    for i in range(0, self.view_num):
        self.hidden_layer[i]['src'] = layers.Dense(256)
        self.hidden_layer[i]['tar'] = layers.Dense(256)
    self.out_layer = [{}] * self.view_num
    for i in range(0, self.view_num):
        self.out_layer[i]['src'] = layers.Dense(self.dim)
        self.out_layer[i]['tar'] = layers.Dense(self.dim)

    self.att_vec = tf.get_variable(
            'att_vec',
            shape = [self.view_num, self.dim],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    self.gamma = gamma



  def to_sample(self, inputs, view):
    path_list = [euler_ops.random_walk(inputs,pattern,
        p = 1, q = 1, default_node=-1) for pattern in self.path_patterns[view]]
    pair_list = [euler_ops.gen_pair(path,
        self.left_win_size,self.right_win_size)
        for path in path_list]
    pairs = tf.concat(pair_list,1)#[batch_size, num_pairs ,2]
    num_pairs = pairs.shape[1]
    mask = tf.fill(tf.shape(pairs),-1)
    mask = tf.cast(mask, tf.int64)
    bool_mask = tf.not_equal(pairs,mask)
    s0, s1 = tf.split(bool_mask, [1, 1], 2)
    bool_indices = tf.logical_and(s0,s1)#[bs, num_pairs, 1]
    bool_indices = tf.squeeze(bool_indices,[2])#[bs, num_pairs]
    where_true = tf.where(bool_indices)#[num_true, 2]
    res_pairs = tf.gather_nd(pairs, where_true) #[num_true,2]

    src, pos = tf.split(res_pairs,[1,1] ,axis = -1)
    src = tf.reshape(src,[-1,1])
    pos = tf.reshape(pos,[-1,1])
    num_true = tf.size(pos)
    self.real_batch_size = num_true
    negs = euler_ops.sample_node(num_true * self.num_negs, -1)
    negs = tf.reshape(negs, [num_true * self.num_negs])
    return src , pos, negs

  def source_encoder(self, inputs, view):
    raw_emb = self.feature_embedding_lookup(inputs)
    embedding = self.id_dnn_net(raw_emb, 'src', view)
    return embedding

  def context_encoder(self, inputs, view):
    raw_emb = self.feature_embedding_lookup(inputs)
    embedding = self.id_dnn_net(raw_emb, 'tar', view)
    return embedding

  def id_dnn_net(self, inputs, name, view):
    hidden = self.hidden_layer[view][name](inputs)
    out = self.out_layer[view][name](hidden)
    return out

  def decoder(self, embedding, embedding_pos, embedding_negs):
    #[batch_size * self.raw_fdim]
    pos_cos = self.cosine_fun(embedding, embedding_pos)
    pos_cos = tf.reshape(pos_cos,[self.real_batch_size,1,1])
    embedding = tf.reshape(embedding,
            [self.real_batch_size,1,self.dim])
    embedding_pos = tf.reshape(embedding_pos,
            [self.real_batch_size,1,self.dim])
    embedding_negs = tf.reshape(embedding_negs,
            [self.real_batch_size,self.num_negs,self.dim])
    embedding_tile = tf.tile(embedding,
            [1,self.num_negs,1])
    neg_cos = self.cosine_fun(embedding_tile, embedding_negs)
    neg_cos = tf.reshape(neg_cos,[self.real_batch_size,1,self.num_negs])
    mrr = self._mrr(pos_cos,neg_cos)
    true_labels = tf.ones_like(pos_cos)
    false_labels = tf.zeros_like(neg_cos)
    labels = tf.concat([true_labels,false_labels],axis=-1)
    logits = tf.concat([pos_cos,neg_cos],axis=-1)
    loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels, logits = logits)
    return loss, mrr

  def get_att_embedding(self, emb, src, view):
    all_emb = []
    for i in range(0,self.view_num):
      if i == view:
        all_emb.append(tf.expand_dims(emb,1))
      else:
        all_emb.append(tf.expand_dims(self.source_encoder(src,i),1))
    emb_vec = tf.concat(all_emb, axis = 1)
    att_logit = tf.reduce_sum(tf.multiply(emb_vec,self.att_vec),-1)
    att_weight = tf.nn.softmax(att_logit)
    att_weight = tf.expand_dims(att_weight,1)
    res = tf.squeeze(tf.matmul(att_weight,emb_vec))
    return res

  def call(self, inputs):
    single_view_loss = []
    multi_view_loss = []
    mrr_view = []
    for i in range(self.view_num):
      src, pos , negs = self.to_sample(inputs,i)
      embedding = self.source_encoder(src,i)
      embedding_pos = self.context_encoder(pos,i)
      embedding_negs = self.context_encoder(negs,i)
      loss, mrr = self.decoder(embedding, embedding_pos, embedding_negs)
      single_view_loss.append(tf.reduce_sum(loss))
      embedding_att = self.get_att_embedding(embedding,src,i)
      loss_att, mrr = self.decoder(embedding_att, embedding_pos, embedding_negs)
      mrr_view.append(tf.reduce_mean(mrr))
      multi_view_loss.append(tf.reduce_sum(loss_att))

    loss = tf.reduce_sum(single_view_loss)+tf.reduce_sum(multi_view_loss)
    mrr = tf.reduce_mean(mrr_view)
    return ModelOutput(
        embedding=embedding, loss=loss, metric_name='mrr', metric=mrr)

  def cosine_fun(self,ays_src, ays_dst):
    src_norm = tf.sqrt(tf.reduce_sum(tf.square(ays_src), -1, True))
    dst_norm = tf.sqrt(tf.reduce_sum(tf.square(ays_dst), -1, True))
    prod = tf.reduce_sum(tf.multiply(ays_src, ays_dst), -1, True)
    norm_prod = tf.multiply(src_norm, dst_norm)
    cosine = tf.truediv(prod, norm_prod)
    return cosine

  def feature_embedding_lookup(self,inputs):
    nodes = tf.reshape(inputs,[-1])
    features = euler_ops.get_sparse_feature(nodes, self.feature_ids,default_values=0)
    feature_embs = tf.concat([self.feature_embedding_layer[j](features[i])
        for i, j in enumerate(self.feature_ids)],axis=-1)
    return feature_embs
