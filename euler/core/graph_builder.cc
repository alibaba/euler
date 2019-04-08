/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "euler/core/graph_builder.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <vector>
#include <thread>

#include "glog/logging.h"

#include "euler/common/bytes_reader.h"

namespace euler {
namespace core {

bool GraphBuilder::LoadData(LoaderType loader_type,
                            const std::vector<std::string>& file_list,
                            Graph* graph, std::string addr, int32_t port,
                            NODEVEC &np, EDGEVEC &ep,
                            int32_t& n_type_num, int32_t& e_type_num) {
  for (size_t i = 0; i < file_list.size(); ++i) {
    euler::common::FileIO* reader = nullptr;
    euler::common::FileIO::ConfigMap config;
    if (loader_type == local) {
      config.insert({"filename", file_list[i]});
      config.insert({"read", "true"});
    } else {
      config.insert({"addr", addr});
      config.insert({"port", std::to_string(port)});
      config.insert({"path", file_list[i]});
      config.insert({"read", "true"});
    }

    euler::common::FileIOFactory* file_io_factory = nullptr;
    if (loader_type == local) {
      if (euler::common::factory_map.find("local") ==
          euler::common::factory_map.end()) {
        LOG(ERROR) << "no local file io factory register";
        return false;
      }
      file_io_factory = euler::common::factory_map["local"];
    } else {
      if (euler::common::factory_map.find("hdfs") ==
          euler::common::factory_map.end()) {
        LOG(ERROR) << "no hdfs file io factory register";
        return false;
      }
      file_io_factory = euler::common::factory_map["hdfs"];
    }
    reader = file_io_factory->GetFileIO(config);
    if (reader == nullptr) {
      LOG(ERROR) << file_list[i] << " reader error!";
      return false;
    }
    int32_t block_size = 0;
    while (reader->Read(&block_size)) {
      if (!ParseBlock(reader, graph, block_size, np, ep, n_type_num, e_type_num)) {
        LOG(ERROR) << file_list[i] << " data error!";
        return false;
      }
    }

    delete reader;
    LOG(INFO) << "Load Done: " << file_list[i];
  }

  return true;
}

Graph* GraphBuilder::BuildGraph(const std::vector<std::string>& file_names,
                                LoaderType loader_type, std::string addr,
                                int32_t port,
                                GlobalSamplerType global_sampler_type) {
  int THREAD_NUM = std::thread::hardware_concurrency();
  Graph* graph = factory_->CreateGraph();
  bool load_success = true;
  std::vector<std::thread> thread_list;
  int p_num = file_names.size() / THREAD_NUM + 1;
  NODEVEC tmp_node_vec[THREAD_NUM];
  EDGEVEC tmp_edge_vec[THREAD_NUM];
  int32_t n_type_num[THREAD_NUM];
  int32_t e_type_num[THREAD_NUM];
  memset(n_type_num, 0, sizeof(n_type_num));
  memset(e_type_num, 0, sizeof(e_type_num));
  for (int i = 0; i < THREAD_NUM; ++i) {
    std::vector<std::string> file_list;
    int j = i * p_num;
    int j_end = std::min(j + p_num, static_cast<int>(file_names.size()));
    for (; j < j_end; ++j) {
      file_list.push_back(file_names[j]);
    }

    //LOG(INFO) << "Thread " << i <<  ", job size: " << file_list.size();
    thread_list.push_back(std::thread(
        [this, loader_type, file_list, graph, addr, port,i, &tmp_node_vec, &tmp_edge_vec,
        &n_type_num, &e_type_num] (bool* success) {
          *success = *success && LoadData(loader_type, file_list, graph, addr, port, tmp_node_vec[i],
                                          tmp_edge_vec[i], n_type_num[i], e_type_num[i]);
        }, &load_success));
  }
  for (int i = 0; i < THREAD_NUM; ++i) {
    thread_list[i].join();
  }

  int64_t node_size = 0, edge_size = 0;
  for(int i = 0 ; i < THREAD_NUM ; i++) {
    node_size += tmp_node_vec[i].size();
    edge_size += tmp_edge_vec[i].size();
  }
  LOG(INFO) << "Each Thread Load Finish! Node Count:" << node_size<< " Edge Count:"<< edge_size;


  if (!load_success) {
    LOG(ERROR) << "Graph build failed!";
    return nullptr;
  }
  else {
    LOG(INFO) << "Graph Loading Finish!";
  }

  for(int i = 0 ; i < THREAD_NUM ; i++) {
    graph->AddNodeFrom(tmp_node_vec[i]);
    tmp_node_vec[i].clear();
    graph->AddEdgeFrom(tmp_edge_vec[i]);
    tmp_edge_vec[i].clear();
    graph->SetNodeTypeNum(n_type_num[i]);
    graph->SetEdgeTypeNum(e_type_num[i]);
  }

    LOG(INFO) << "Graph Load Finish! Node Count:" << graph->getNodeSize()<< " Edge Count:"
        << graph->getEdgeSize();

  if (global_sampler_type == node) {
    graph->BuildGlobalSampler();
    LOG(INFO) << "Done: build node sampler";
  } else if (global_sampler_type == edge) {
    graph->BuildGlobalEdgeSampler();
    LOG(INFO) << "Done: build edge sampler";
  } else if (global_sampler_type == all) {
    graph->BuildGlobalSampler();
    graph->BuildGlobalEdgeSampler();
    LOG(INFO) << "Done: build all sampler";
  }

  LOG(INFO) << "Graph build finish";
  return graph;
}

bool GraphBuilder::ParseBlock(euler::common::FileIO* file_io, Graph* graph,
                              int32_t checksum, NODEVEC &np, EDGEVEC &ep,
                              int32_t& n_type_num, int32_t& e_type_num) {
  int32_t node_info_bytes = 0;
  std::string node_info;
  if (!file_io->Read(&node_info_bytes)) {
    return false;
  }

  if (!file_io->Read(static_cast<size_t>(node_info_bytes), &node_info)) {
    return false;
  }

  Node* node = factory_->CreateNode();
  if (!node->DeSerialize(node_info)) {
    return false;
  }

  np.push_back(node);
  int tmp = node->GetType() + 1;
  n_type_num = tmp > n_type_num ? tmp: n_type_num;

  int32_t edges_num = 0;
  if (!file_io->Read(&edges_num)) {
    return false;
  }

  std::vector<int32_t> edges_info_bytes;
  if (!file_io->Read(edges_num, &edges_info_bytes)) {
    return false;
  }

  for (auto& byte_num : edges_info_bytes) {
    std::string edge_info;
    if (!file_io->Read(static_cast<size_t>(byte_num), &edge_info)) {
      return false;
    }
    Edge* edge = factory_->CreateEdge();
    if (!edge->DeSerialize(edge_info)) {
      return false;
    }
    ep.push_back(edge);
    int tmp_e = edge->GetType() + 1;
    e_type_num = tmp_e > e_type_num ? tmp_e: e_type_num;
  }
  int32_t total_edges_info_bytes = 0;
  for (size_t i = 0; i < edges_info_bytes.size(); ++i) {
    total_edges_info_bytes += edges_info_bytes[i];
  }

  // checksum
  if (4 + 4 + 4 * edges_info_bytes.size() + node_info_bytes +
      total_edges_info_bytes == static_cast<size_t>(checksum)) {
    return true;
  } else {
    LOG(ERROR) << "checksum fail";
    return false;
  }
}

}  // namespace core
}  // namespace euler
