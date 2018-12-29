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

#include <algorithm>
#include <vector>
#include <thread>

#include "glog/logging.h"

#include "euler/common/bytes_reader.h"

#define THREAD_NUM 100

namespace euler {
namespace core {

bool GraphBuilder::LoadData(const std::vector<std::string>& file_list,
                            Graph* graph, std::string addr, int32_t port) {
  for (size_t i = 0; i < file_list.size(); ++i) {
    euler::common::FileIO* reader = nullptr;
    euler::common::FileIO::ConfigMap config;
    if (addr.empty()) {
      reader = new euler::common::LocalFileIO();
      config.insert({"filename", file_list[i]});
      config.insert({"read", "true"});
    } else {
      reader = new euler::common::HdfsFileIO();
      config.insert({"addr", addr});
      config.insert({"port", std::to_string(port)});
      config.insert({"path", file_list[i]});
      config.insert({"read", "true"});
    }

    if (!reader->Initialize(config)) {
      return false;
    }

    int32_t block_size = 0;
    while (reader->Read(&block_size) && ParseBlock(reader, graph)) { }

    delete reader;
    LOG(INFO) << "Load Done: " << file_list[i];
  }

  return true;
}

Graph* GraphBuilder::BuildGraph(const std::vector<std::string>& file_names,
                                LoaderType loader_type, std::string addr,
                                int32_t port,
                                GlobalSamplerType global_sampler_type) {
  Graph* graph = factory_->CreateGraph();
  bool load_success = true;
  if (loader_type == local) {
    load_success = LoadData(file_names, graph, "", 0);
  } else {
    std::vector<std::thread> thread_list;
    int p_num = file_names.size() / THREAD_NUM + 1;
    for (int i = 0; i < THREAD_NUM; ++i) {
      std::vector<std::string> file_list;
      int j = i * p_num;
      int j_end = std::min(j + p_num, static_cast<int>(file_names.size()));
      for (; j < j_end; ++j) {
        file_list.push_back(file_names[j]);
      }

      LOG(INFO) << "Thread " << i <<  ", job size: " << file_list.size();
      thread_list.push_back(std::thread(
          [this, file_list, graph, addr, port] (bool* success) {
            *success = *success && LoadData(file_list, graph, addr, port);
          }, &load_success));
    }
    for (size_t i = 0; i < THREAD_NUM; ++i) {
      thread_list[i].join();
    }
  }

  if (!load_success) {
    LOG(ERROR) << "Graph build failed!";
    return nullptr;
  }

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

bool GraphBuilder::ParseBlock(euler::common::FileIO* file_io, Graph* graph) {
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

  graph->AddNode(node);
  graph->SetNodeTypeNum(node->GetType() + 1);

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
    graph->AddEdge(edge);
    graph->SetEdgeTypeNum(edge->GetType() + 1);
  }

  return true;
}

}  // namespace core
}  // namespace euler
