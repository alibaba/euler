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

import pickle

import faiss
import numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("embedding_file", None, "Embedings of nodes")
flags.DEFINE_string("id_file", None, "Ids of nodes")
flags.DEFINE_string("query_file", None, "Query vectors")
flags.DEFINE_string("index_type", None, "faiss index type")

flags.mark_flag_as_required("embedding_file")

def build_index(embedding):
    n = embedding.shape[0]
    d = embedding.shape[1]

    index_type = 'ivfflat'
    if FLAGS.index_type:
        index_type = FLAGS.index_type

    if index_type == 'ivfflat':
        ncent = int(4 * np.sqrt(n))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, ncent,
                                   faiss.METRIC_L2)
        index.cp.min_points_per_centroid = 4
        index.nprobe = 10
        index.train(embedding)
        index.add(embedding)

    return index


def parse_query():
    return np.loadtxt(FLAGS.query_file, dtype=np.float32, delimiter=',')

def main(argv):
    del argv

    with open(os.path.realpath(FLAGS.embedding_file), 'rb') as emb_f:
        embedding = np.load(emb_f)
    with open(os.path.realpath(FLAGS.embedding_file), 'rb') as id_f:
        idx = np.load(id_f)

    assert(len(idx) == len(embedding))

    index = build_index(embedding)

    if FLAGS.query_file:
        query = parse_query()
    else:
        query = embedding[:25]

    D, I = index.search(query, 10)
    I = np.array([[idx[col] for col in row] for row in I])

    result = {'distance':D, 'idx': I}
    with open('result.pkl', 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    app.run(main)

