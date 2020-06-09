/* Copyright 2020 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "euler/core/graph/graph_builder.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <memory>

#include "euler/common/logging.h"
#include "euler/common/env.h"
#include "euler/core/graph/graph_meta.h"
#include "euler/core/graph/graph.h"
#include "euler/common/signal.h"

namespace euler {

enum TaskType { kLoadNodeTask, kLoadEdgeTask };

std::string TaskTypeString(TaskType type) {
  switch (type) {
    case kLoadNodeTask:
      return "Node";
    case kLoadEdgeTask:
      return "Edge";
    default:
      assert(false);
  }
}

struct TaskInfo {
  FileIO *dir;
  std::vector<std::string>* files;
  int start;
  int end;
  TaskType type;
  NodeVec node_vec;
  EdgeVec edge_vec;
};

Status GraphBuilder::Build(Graph* graph, Slice path,
                           GlobalSamplerType sampler_type,
                           FileIO::FilterFunc filter,
                           GraphDataType data_type) {
  int thread_num = 8;
  int job_num = thread_num * 8;

  std::vector<TaskInfo> task_infos;
  std::unique_ptr<FileIO> node_dir, edge_dir;
  std::vector<std::string> node_files, edge_files;

  if (data_type & kLoadNode) {
    auto node_dir_path = JoinPath(path, "Node");
    RETURN_IF_ERROR(Env::Default()->NewFileIO(node_dir_path, true, &node_dir));

    if (!node_dir->initialized() || !node_dir->IsDirectory()) {
      EULER_LOG(ERROR) << "No such directory found, path: " << node_dir_path;
      return Status::NotFound("Directory ", node_dir_path, " not found!");
    }

    node_files = node_dir->ListDirectory(filter);
    int job_size = node_files.size() / job_num;
    if (node_files.size() % job_num != 0) {
      ++job_size;
    }

    for (int i = 0; i < job_num; ++i) {
      task_infos.emplace_back();
      auto& task_info = task_infos.back();
      task_info.dir = node_dir.get();
      task_info.files = &node_files;
      task_info.start = i * job_size;
      task_info.end = task_info.start + job_size;
      task_info.type = kLoadNodeTask;
    }
  }

  if (data_type & kLoadEdge) {
    auto edge_dir_path = JoinPath(path, "Edge");
    RETURN_IF_ERROR(Env::Default()->NewFileIO(edge_dir_path, true, &edge_dir));

    if (!edge_dir->initialized() || !edge_dir->IsDirectory()) {
      EULER_LOG(ERROR) << "No such directory found, path: " << edge_dir_path;
      return Status::NotFound("Directory ", edge_dir_path, " not found!");
    }

    edge_files = edge_dir->ListDirectory(filter);
    int job_size = edge_files.size() / job_num;
    if (edge_files.size() % job_num != 0) {
      ++job_size;
    }

    for (int i = 0; i < job_num; ++i) {
      task_infos.emplace_back();
      auto& task_info = task_infos.back();
      task_info.dir = edge_dir.get();
      task_info.files = &edge_files;
      task_info.start = i * job_size;
      task_info.end = task_info.start + job_size;
      task_info.type = kLoadEdgeTask;
    }
  }

  std::unique_ptr<ThreadPool> thread_pool(
      Env::Default()->StartThreadPool("GraphBuilder", thread_num));
  std::atomic<bool> success(true);
  std::atomic<int> counter(task_infos.size());
  Signal signal;
  for (auto &task_info : task_infos) {
    thread_pool->Schedule(
        [this, &counter, &signal, &task_info, &success] () {
          if (success && !LoadData(&task_info)) {
            success = false;
          }
          if (--counter == 0) {
            signal.Notify();
          }
        });
  }

  if (data_type) {
    signal.Wait();
  }
  thread_pool->Shutdown();

  GraphMeta meta;
  auto meta_path = JoinPath(path, "euler.meta");
  RETURN_IF_ERROR(LoadMeta(meta_path, &meta));
  graph->set_meta(meta);

  if (!success) {
    EULER_LOG(ERROR) << "Load data error!";
    return Status::Internal("Load data failed!");
  }

  RETURN_IF_ERROR(
      AddToGraph(graph, task_infos.data(),
                  task_infos.data() + task_infos.size()));
  RETURN_IF_ERROR(BuildSampler(graph, sampler_type));
  EULER_LOG(INFO) << "Graph build completely!";
  return Status::OK();
}

Status GraphBuilder::AddToGraph(
    Graph* graph, const TaskInfo* start, const TaskInfo* end) {
  for (; start < end; ++start) {
    const auto& task_info = *start;
    graph->AddNodeFrom(task_info.node_vec);
    graph->AddEdgeFrom(task_info.edge_vec);
  }

  EULER_LOG(INFO) << "Graph Node Count:" << graph->getNodeSize();
  EULER_LOG(INFO) << "Graph Edge Count:" << graph->getEdgeSize();

  return Status::OK();
}

bool GraphBuilder::LoadData(TaskInfo* task_info) {
  auto& files = *task_info->files;
  int start = task_info->start;
  int end = std::min(task_info->end, static_cast<int>(files.size()));
  for (; start < end; ++start) {
    auto reader = task_info->dir->Open(files[start], true);
    std::string file_name =
        JoinPath(TaskTypeString(task_info->type), files[start]);
    if (reader == nullptr) {
      EULER_LOG(ERROR) << "Open file " << file_name << " failed!!";
      return false;
    }
    bool success;
    switch (task_info->type) {
      case kLoadNodeTask:
        success = ParseNodes(reader.get(), &task_info->node_vec);
        break;
      case kLoadEdgeTask:
        success = ParseEdges(reader.get(), &task_info->edge_vec);
        break;
      default:
        assert(false);
    }
    if (!success) {
      EULER_LOG(ERROR) << file_name << " data error!";
      return false;
    }
    EULER_LOG(INFO) << "load data file ok: " << file_name;
  }

  return true;
}

Status GraphBuilder::BuildSampler(Graph* graph,
                                  GlobalSamplerType sampler_type) {
  switch (sampler_type) {
    case kNone:
      break;
    case kNode:
      graph->BuildGlobalSampler();
      break;
    case kEdge:
      graph->BuildGlobalEdgeSampler();
      break;
    case kAll:
      graph->BuildGlobalSampler();
      graph->BuildGlobalEdgeSampler();
      break;
    default:
      assert(false);
  }

  return Status::OK();
}


Status GraphBuilder::LoadMeta(Slice meta_path, GraphMeta* meta) {
  std::unique_ptr<FileIO> reader;
  RETURN_IF_ERROR(Env::Default()->NewFileIO(meta_path, true, &reader));

  std::vector<std::string> meta_infos;
  std::string name, version;
  uint64_t node_count, edge_count;
  int partitions_num = 0;

  reader->Read(&name);
  reader->Read(&version);
  reader->Read(&node_count);
  reader->Read(&edge_count);
  reader->Read(&partitions_num);

  uint32_t node_meta_count;
  reader->Read(&node_meta_count);

  FeatureInfoMap nfm, efm;
  for (size_t i = 0; i < node_meta_count; i++) {
    std::string fname;
    FeatureType type;
    int32_t idx;
    int64_t dim;
    reader->Read(&fname);
    reader->Read(&type);
    reader->Read(&idx);
    reader->Read(&dim);
    nfm.insert(
        std::make_pair(
            fname, std::make_tuple(type, idx, dim)));
  }

  uint32_t edge_meta_count;
  reader->Read(&edge_meta_count);
  for (size_t i = 0; i < edge_meta_count; i++) {
    std::string fname;
    FeatureType type;
    int32_t idx;
    int64_t dim;
    reader->Read(&fname);
    reader->Read(&type);
    reader->Read(&idx);
    reader->Read(&dim);
    efm.insert(
        std::make_pair(
            fname, std::make_tuple(type, idx, dim)));
  }

  std::unordered_map<std::string, uint32_t> node_type_map;
  std::unordered_map<std::string, uint32_t> edge_type_map;

  uint32_t node_type_num = 0;
  reader->Read(&node_type_num);
  for (uint32_t i = 0; i < node_type_num; ++i) {
    std::string node_type;
    uint32_t index = 0;
    reader->Read(&node_type);
    reader->Read(&index);
    node_type_map.insert({node_type, index});
  }

  uint32_t edge_type_num = 0;
  reader->Read(&edge_type_num);
  for (uint32_t i = 0; i < edge_type_num; ++i) {
    std::string edge_type;
    uint32_t index = 0;
    reader->Read(&edge_type);
    reader->Read(&index);
    edge_type_map.insert({edge_type, index});
  }

  meta->Init(name, version, node_count, edge_count,
             partitions_num, nfm, efm, node_type_map, edge_type_map);
  EULER_LOG(INFO) << "Meta File Load Done: " << meta_path
                  << ", Meta: " << meta->ToString();

  return Status::OK();
}

bool GraphBuilder::ParseNodes(FileIO* reader, NodeVec* node_vec) {
  std::string node_info;
  while (reader->Read(&node_info)) {
    Node* node = new Node;
    if (!node->DeSerialize(node_info)) {
      return false;
    }
    node_vec->emplace_back(node);
  }
  return true;
}

bool GraphBuilder::ParseEdges(FileIO* reader, EdgeVec* edge_vec) {
  std::string edge_info;
  while (reader->Read(&edge_info)) {
    Edge* edge = new Edge;
    if (!edge->DeSerialize(edge_info)) {
      return false;
    }
    edge_vec->emplace_back(edge);
  }
  return true;
}

}  // namespace euler
