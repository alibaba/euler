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
import tqdm
import urllib
import zipfile
import numpy as np

from tf_euler.python.dataset.base_dataset import DataSet

current_dir = os.path.dirname(os.path.realpath(__file__))
cora_dir = os.path.join(current_dir, 'MovieLens-1M')


class MovieLens_1M(DataSet):

    def __init__(self, data_dir=cora_dir, data_type='all'):
        super(MovieLens_1M, self).__init__(data_dir, data_type)
        self.source_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        self.origin_files = ['ml-1m/movies.dat',
                             'ml-1m/ratings.dat',
                             'ml-1m/users.dat']
        self.max_node_id = 9992
        self.train_node_type = ['train']
        self.train_edge_type = ['train']
        self.total_size = 9992
        self.all_node_type = -1
        self.all_edge_type = ['train', 'train_removed']
        self.id_file = os.path.join(data_dir, 'ml_1m_test.id')

        self.genre = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.genre_map = {key:i for key,i in zip(self.genre, range(len(self.genre)))}
        self.gender_map = {'M':0, 'F':1}
        self.age_map = {'1':0, '18':1, '25':2, '35':3, '45':4, '50':5, '56':6}
        self.movie_len = 3952

    def download_data(self, source_url, out_dir):
        fb_tgz_dir = os.path.join(out_dir, 'ml-1m.zip')
        urllib.urlretrieve(self.source_url, fb_tgz_dir)
        with zipfile.ZipFile(fb_tgz_dir) as fb_file:
            print('unzip data..')
            fb_file.extractall(out_dir)

    def convert2json(self, convert_dir, out_dir):
        def get_genre(genre_str):
            g = genre_str.split('|')
            result = []
            for i in g:
                result.append(self.genre_map[i])
            return result
            
        with open(out_dir, 'w') as out, open(self.id_file, 'w') as out_test:
            buf = {}
            buf["nodes"] = []
            buf["edges"] = []
            in_file = open(convert_dir + '/ml-1m/movies.dat', 'r')
            for line in in_file.readlines():
                triple = line.strip().split('::')
                node_id = int(triple[0])
                node_buf = {}
                node_buf["id"] = node_id
                node_buf["type"] = "movie" 
                node_buf["weight"] = 1.0
                node_buf["features"] = [{}]
                node_buf["features"][0]['name'] = 'genre'
                node_buf["features"][0]['type'] = 'sparse'
                node_buf["features"][0]['value'] = get_genre(triple[2])
                buf["nodes"].append(node_buf)
            in_file.close()

            in_file = open(convert_dir + '/ml-1m/users.dat', 'r')
            for line in in_file.readlines():
                triple = line.strip().split('::')
                node_id = int(triple[0]) + self.movie_len
                node_buf = {}
                node_buf["id"] = node_id
                node_buf["type"] = "user" 
                node_buf["weight"] = 1.0
                node_buf["features"] = [{},{},{},{}]
                node_buf["features"][0]['name'] = 'gender'
                node_buf["features"][0]['type'] = 'sparse'
                node_buf["features"][0]['value'] = [self.gender_map[triple[1]], ]
                node_buf["features"][1]['name'] = 'age'
                node_buf["features"][1]['type'] = 'sparse'
                node_buf["features"][1]['value'] = [self.age_map[triple[2]], ]
                node_buf["features"][2]['name'] = 'occupation'
                node_buf["features"][2]['type'] = 'sparse'
                node_buf["features"][2]['value'] = [int(triple[3]), ]
                node_buf["features"][3]['name'] = 'zip_code'
                node_buf["features"][3]['type'] = 'binary'
                node_buf["features"][3]['value'] = [str(triple[4]), ]
                buf["nodes"].append(node_buf)
            in_file.close()

            in_file = open(convert_dir + '/ml-1m/ratings.dat', 'r')
            for line in in_file.readlines():
                triple = line.strip().split('::')
                edge_buf = {}
                edge_buf['src'] = int(triple[0]) + self.movie_len
                edge_buf['dst'] = int(triple[1])
                edge_buf['type'] = "rate"
                edge_buf['weight'] = 1
                edge_buf["features"] = [{},{}]
                edge_buf["features"][0]['name'] = 'rating'
                edge_buf["features"][0]['type'] = 'sparse'
                edge_buf["features"][0]['value'] = [int(triple[2]), ]
                edge_buf["features"][1]['name'] = 'timestamp'
                edge_buf["features"][1]['type'] = 'binary'
                edge_buf["features"][1]['value'] = [str(triple[3]), ]
                buf["edges"].append(edge_buf)
            in_file.close()

            out.write(json.dumps(buf))
            for i in range(self.max_node_id):
                out_test.write(str(i+1)+ '\n')
            print("Total Node: {}, Edge: {}."
                  .format(len(buf["nodes"]), len(buf["edges"])))
