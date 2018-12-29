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

import json
import struct


class Converter(object):
  def __init__(self, meta_path, input_path, output_path):
    self.meta_path = meta_path
    self.input_path = input_path
    self.output_path = output_path
    self.meta_data = json.load(open(meta_path, 'r'))
    self.out = open(self.output_path, 'wb')

  def do(self):
    r = open(self.input_path)
    print("convert data...")
    for line in r:
      data = json.loads(line)
      self.out.write(self.parse_block(data))
    self.close()

  def parse_block(self, data):
    # node info
    node_info_bytes = 0
    node_weight = data["node_weight"]
    node_type = data["node_type"]
    node_id = data["node_id"]
    edge_type_num = int(self.meta_data['edge_type_num'])
    edge_group_size_list = []  # int32
    edge_group_weight_list = []  # float32
    neighbor_id_list = []  # uint64
    neighbor_weight_list = []  # float32

    for i in range(0, edge_type_num):
      neighbor_dict = data['neighbor'][str(i)]
      edge_group_size_list.append(len(neighbor_dict))
      edge_group_weight_list.append(sum(i for i in neighbor_dict.values()))
      # print sum(i for i in data['neighbor'][str(i)].values())
      for key in neighbor_dict:
        # print key, neighbor_dict[key]
        neighbor_id_list.append(int(key))
        neighbor_weight_list.append(neighbor_dict[key])

    feature_slot_num = {}
    feature_size_list_dic = {}
    feature_value_list_dic = {}
    for feature_type in ['uint64', 'float', 'binary']:
      feature_slot_num[feature_type] = int(
          self.meta_data['node_' + feature_type + '_feature_num'])
      feature_size_list_dic[feature_type] = [
          len(data[feature_type + '_feature'][str(i)])
          for i in range(0, feature_slot_num[feature_type])
      ]
      feature_value_list_dic[feature_type] = []
      if feature_type != 'binary':
        for i in range(0, feature_slot_num[feature_type]):
          feature_value_list_dic[feature_type].extend(
              data[feature_type + '_feature'][str(i)])
      else:
        for i in range(0, feature_slot_num[feature_type]):
          feature_value_list_dic[feature_type].append(
              str(data[feature_type + '_feature'][str(i)]))
      # print feature_type, feature_size_list_dic[feature_type]
      # print feature_type, feature_value_list_dic[feature_type]

    # calculate node info bytes
    node_info_bytes = 32 + 4 * len(edge_group_size_list) + 4 * len(
        edge_group_weight_list) + 8 * len(neighbor_id_list) + 4 * len(
            neighbor_weight_list) + 4 * sum(
                len(i) for i in feature_size_list_dic.values()) + 8 * sum(
                    feature_size_list_dic['uint64']) + 4 * sum(
                        feature_size_list_dic['float']) + sum(
                            feature_size_list_dic['binary'])

    # pack
    fmt = '<2iQifi' + (str(len(edge_group_size_list)) + 'i') + (
        str(len(edge_group_weight_list)) + 'f'
    ) + (str(len(neighbor_id_list)) + 'Q') + (
        str(len(neighbor_weight_list)) + 'f'
    ) + (str(len(feature_size_list_dic['uint64']) + 1) + 'i') + (
        str(len(feature_value_list_dic['uint64'])) + 'Q') + (
            str(len(feature_size_list_dic['float']) + 1) + 'i') + (
                str(len(feature_value_list_dic['float'])) + 'f') + (
                    str(len(feature_size_list_dic['binary']) + 1) + 'i')

    for i in range(0, feature_slot_num['binary']):
      fmt += (str(feature_size_list_dic['binary'][i]) + 's')
    # print struct.calcsize(fmt)

    # edge info
    edge_num = len(data['edge'])
    fmt += (str(1 + edge_num) + 'i')
    edge_info_bytes = [0] * edge_num
    edge_bytes_buf = []
    for i in range(0, edge_num):
      edge_fmt, edge_bytes = self.parse_edge(data['edge'][i])
      edge_info_bytes[i] = struct.calcsize(edge_fmt)
      edge_bytes_buf.append(edge_bytes)

    block_bytes = node_info_bytes + sum(
        edge_info_bytes) + 4 + 4 + 4 * len(edge_info_bytes)
    # print 'block_bytes',block_bytes
    #pack
    # print 'node fmt', fmt
    values = [
        block_bytes, node_info_bytes, node_id, node_type, node_weight,
        edge_type_num
    ]
    values.extend(edge_group_size_list)
    values.extend(edge_group_weight_list)
    values.extend(neighbor_id_list)
    values.extend(neighbor_weight_list)
    for feature_type in ['uint64', 'float', 'binary']:
      values.append(feature_slot_num[feature_type])
      values.extend(feature_size_list_dic[feature_type])
      values.extend(feature_value_list_dic[feature_type])
    values.append(edge_num)
    values.extend(edge_info_bytes)
    result = struct.pack(fmt, *values)
    # print values,values
    for value in edge_bytes_buf:
      result += value
    return result

  def close(self):
    self.out.close()

  def parse_edge(self, data):
    src_id = data['src_id']
    dst_id = data['dst_id']
    edge_type = data['edge_type']
    weight = data['weight']
    feature_slot_num = {}
    feature_size_list_dic = {}
    feature_value_list_dic = {}
    for feature_type in ['uint64', 'float', 'binary']:
      feature_slot_num[feature_type] = int(
          self.meta_data['edge_' + feature_type + '_feature_num'])
      feature_size_list_dic[feature_type] = [
          len(data[feature_type + '_feature'][str(i)])
          for i in range(0, feature_slot_num[feature_type])
      ]
      feature_value_list_dic[feature_type] = []
      if feature_type != 'binary':
        for i in range(0, feature_slot_num[feature_type]):
          feature_value_list_dic[feature_type].extend(
              data[feature_type + '_feature'][str(i)])
      else:
        for i in range(0, feature_slot_num[feature_type]):
          feature_value_list_dic[feature_type].append(
              str(data[feature_type + '_feature'][str(i)]))
      # print feature_type, feature_size_list_dic[feature_type]
      # print feature_type, feature_value_list_dic[feature_type]
    # pack
    fmt = '<2Qif' + (str(len(feature_size_list_dic['uint64']) + 1) + 'i') + (
        str(len(feature_value_list_dic['uint64'])) + 'Q') + (
            str(len(feature_size_list_dic['float']) + 1) + 'i') + (
                str(len(feature_value_list_dic['float'])) + 'f') + (
                    str(len(feature_size_list_dic['binary']) + 1) + 'i')
    for i in range(0, feature_slot_num['binary']):
      fmt += (str(feature_size_list_dic['binary'][i]) + 's')

    values = [src_id, dst_id, edge_type, weight]
    for feature_type in ['uint64', 'float', 'binary']:
      values.append(feature_slot_num[feature_type])
      values.extend(feature_size_list_dic[feature_type])
      values.extend(feature_value_list_dic[feature_type])
    # print values
    # print len(values),fmt
    bytes = struct.pack(fmt, *values)
    return fmt, bytes
