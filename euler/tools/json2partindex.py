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

import json
import sys
import struct
from euler.tools import json2meta
from euler.tools.util import *
import os


def mkdirs(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        pass


class Converter(object):
    def __init__(self, meta_path, input_path, output_dir, partition_num,
                 pref='index'):
        self.meta_path = meta_path
        self.input_path = os.path.realpath(input_path)
        self.output_dir = os.path.realpath(output_dir)
        self.meta_data = json.load(open(meta_path, 'r'))

        self.partition_num = partition_num
        self.pref = pref
        self.index_data = [{} for i in range(partition_num)]

        self.meta_generator = json2meta.Generator([input_path], '',
                                                  partition_num)

        type_data = ["int8_t", "int16_t", "int32_t", "int64_t",
                     "uint8_t", "uint16_t", "uint32_t", "uint64_t",
                     "float", "double", "bool", "string"]
        self.types = {}
        for i in range(len(type_data)):
            self.types[type_data[i]] = i

    def check_dict(self, dic, key, param="list"):
        if key not in dic:
            if param == "list":
                dic[key] = []
            elif param == "dict":
                dic[key] = {}

    def do(self):
        r = open(self.input_path)
        print ("convert data...")

        data = json.loads(r.read())
        self.meta_generator.parse(data)

        self.neighbor_index = {}
        self.edge_neighbor_index = {}
        if 'node' in self.meta_data:
            for node in data['nodes']:
                pid = node['id'] % self.partition_num
                self.parse_node(node,
                                self.meta_data['node'],
                                node['id'],
                                node['weight'],
                                self.index_data[pid])

        if 'edge' in self.meta_data:
            for edge in data['edges']:
                src_id = edge['src']
                dst_id = edge['dst']
                etype = edge['type']
                pid = src_id % self.partition_num

                etype_no = self.meta_generator.gmeta.edge_type_info[str(etype)]
                self.parse_edge(edge, self.meta_data['edge'],
                                (src_id, dst_id, etype_no),
                                edge['weight'],
                                self.index_data[pid])

        if len(self.neighbor_index) > 0:
            for key in self.neighbor_index:
                for edge in data['edges']:
                    src_id = edge['src']
                    dst_id = edge['dst']
                    etype = edge['type']
                    pid = src_id % self.partition_num
                    index_data = self.index_data[pid]
                    self.check_dict(index_data, key, "dict")
                    self.check_dict(index_data[key], src_id)
                    index_data[key][src_id].append(
                        self.neighbor_index[key][dst_id])

        if len(self.edge_neighbor_index) > 0:
            for key in self.edge_neighbor_index:
                for src_id in self.edge_neighbor_index[key]:
                    pid = src_id % self.partition_num
                    index_data = self.index_data[pid]
                    self.check_dict(index_data, key, "dict")
                    t = self.edge_neighbor_index[key][src_id]
                    index_data[key][src_id] = t

        self.write_index()

    def add_index_data(self, index_data, indexkey, value, id, weight):
        if indexkey.endswith('neighbor_index'):
            self.check_dict(self.neighbor_index,
                            indexkey,
                            "dict")
            self.neighbor_index[indexkey][id] = \
                (value, id, weight)
        else:
            self.check_dict(index_data, indexkey)
            index_data[indexkey].append(
                (value, id, weight))

    def parse_node(self, data, meta, id, weight, index_data):
        for key in data:
            if key == 'features' and 'features' in meta:
                for f in data['features']:
                    fname = f['name']
                    fdict = meta['features']
                    if fname in fdict:
                        if type(fdict[fname]) == type({}):
                            for vidx in fdict[fname]:
                                indexkey = fdict[fname][vidx]
                                value = f['value'][int(vidx)]
                                self.add_index_data(index_data, indexkey, value, id, weight)
                        else:
                            indexkey = fdict[fname]
                            value = f['value']
                            self.add_index_data(index_data, indexkey, value, id, weight)
            else:
                if key in meta:
                    indexkey = meta[key]
                    if indexkey.endswith('neighbor_index'):
                        self.check_dict(self.neighbor_index, indexkey, "dict")
                        self.neighbor_index[indexkey][id] = \
                            (data[key], id, weight)
                    else:
                        self.check_dict(index_data, indexkey)
                        index_data[indexkey].append((data[key], id, weight))

    def parse_edge(self, data, meta, id, weight, index_data):
        node_id = id[0]
        dst_id = id[1]
        iid = edgeIdHash(id[0], id[1], id[2])

        for key in data:
            if key == 'features' and 'features' in meta:
                for f in data['features']:
                    fname = f['name']
                    fdict = meta['features']
                    if fname in fdict:
                        for vidx in fdict[fname]:
                            indexkey = fdict[fname][vidx]
                            value = f['value'][int(vidx)]
                            if indexkey.endswith('neighbor_index'):
                                self.check_dict(self.edge_neighbor_index,
                                                indexkey,
                                                'dict')
                                self.check_dict(
                                    self.edge_neighbor_index[indexkey],
                                    node_id)
                                self.edge_neighbor_index[indexkey][node_id].\
                                    append((value, dst_id, weight))
                            else:
                                self.check_dict(index_data, indexkey)
                                index_data[indexkey].append(
                                    (value, iid, weight))
            else:
                if key in meta:
                    indexkey = meta[key]
                    if indexkey.endswith('neighbor_index'):
                        self.check_dict(self.edge_neighbor_index,
                                        indexkey, "dict")
                        self.check_dict(self.edge_neighbor_index[indexkey],
                                        node_id)
                        self.edge_neighbor_index[indexkey][node_id].append(
                            (data[key], dst_id, weight))
                    else:
                        self.check_dict(index_data, indexkey)
                        index_data[indexkey].append((data[key], iid, weight))

    def write_meta(self):
        result = []
        data = [self.meta_data[k] for k in self.meta_data]
        new_data = []
        while len(data) > 0:
            for v in data:
                if isinstance(v, unicode):
                    result.append(v)
                else:
                    for k in v:
                        new_data.append(v[k])
            data = new_data
            new_data = []

        for key in result:
            item = key.split(':')
            if len(item) != 4:
                print('meta format error')
                exit(1)
            cur_dir = os.path.join(self.output_dir, 'Index', item[0])
            mkdirs(cur_dir)
            f = open(os.path.join(cur_dir, 'meta'), 'w')
            s = ''
            if item[3] == 'hash_index':
                s += struct.pack('i', 0)
            elif item[3] == 'range_index':
                s += struct.pack('i', 1)
            elif item[3] == 'neighbor_index':
                s += struct.pack('i', 2)
            else:
                print('index string ' + item[3] + ' error')

            s += struct.pack('i', self.types[item[2]])
            s += struct.pack('i', self.types[item[1]])
            f.write(s)
            f.close()

    def write_hash_data(self, valueType, idType, data):
        new_data = {}
        for item in data:
            if item[0] not in new_data:
                new_data[item[0]] = ([], [])
            new_data[item[0]][0].append(item[1])
            new_data[item[0]][1].append(item[2])

        s = ''
        for key in new_data:
            s += write_correct_data(valueType, key)
            # ids vector
            s += struct.pack('I', len(new_data[key][0]))
            for i in new_data[key][0]:
                s += write_correct_data(idType, i)
            # weights vector
            s += struct.pack('I', len(new_data[key][1]))
            for i in new_data[key][1]:
                s += write_correct_data("float", i)
        return s

    def write_range_data(self, valueType, idType, data):
        data.sort(key=lambda x: x[0])
        values = []
        ids = []
        weights = []
        for item in data:
            values.append(item[0])
            ids.append(item[1])
            weights.append(item[2])

        s = struct.pack('I', len(ids))
        for i in ids:
            s += write_correct_data(idType, i)

        s += struct.pack('I', len(values))
        for i in values:
            s += write_correct_data(valueType, i)

        total = 0
        s += struct.pack('I', len(weights))
        for i in weights:
            total += i
            s += write_correct_data("float", total)

        return s

    def write_neighbor_data(self, valueType, idType, data):
        s = ''
        for key in data:
            s += write_correct_data(idType, key)
            s += self.write_range_data(valueType, idType, data[key])
        return s

    def write_index_data(self, index_data, i):
        for key in index_data:
            s = ''
            item = key.split(':')
            if item[3] == 'hash_index':
                s += self.write_hash_data(item[1], item[2], index_data[key])
            elif item[3] == 'range_index':
                s += self.write_range_data(item[1], item[2], index_data[key])
            elif item[3] == 'neighbor_index':
                s += self.write_neighbor_data(item[1],
                                              item[2],
                                              index_data[key])
            name = item[0]
            path = os.path.join(self.output_dir,
                                'Index',
                                name,
                                self.pref + '_%d.dat' % (i))
            f = open(path, 'w')
            f.write(s)
            f.close()

    def write_index(self):
        self.write_meta()
        for i in range(self.partition_num):
            self.write_index_data(self.index_data[i], i)


if __name__ == '__main__':
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("python json2partindex.py meta_path input_path output_dir "
              "paritition_num [prefix]")
        exit(1)
    if len(sys.argv) == 5:
        c = Converter(sys.argv[1], os.path.realpath(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    if len(sys.argv) == 6:
        c = Converter(sys.argv[1], os.path.realpath(sys.argv[2]), sys.argv[3], int(sys.argv[4]),
                      sys.argv[5])
    c.do()
