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

#include "euler/client/client_manager.h"

namespace euler {

bool ClientManager::RetrieveShardMeta(
    int32_t shard_index, const std::string& key,
    std::vector<std::vector<float>>* weights) {
  std::string weight;
  if (!server_monitor_->GetShardMeta(shard_index, key, &weight)) {
    EULER_LOG(ERROR) << "Retrieve shard meta failed, key: " << key
                     << " shard index: " << shard_index;
    return false;
  }

  std::vector<std::string> weight_vec;
  weight_vec = Split(weight, ',');
  if (weight_vec.empty()) {
    EULER_LOG(ERROR) << "Invalid weight meta failed, shard: "
                     << shard_index << " weight meta: " << weight;
    return false;
  }

  // Resize weights vector to fit type if necessary
  if (weights->size() < weight_vec.size()) {
    weights->resize(weight_vec.size());
  }

  for (auto& wv : *weights) {
    wv.resize(shard_index + 1, 0.0);
  }

  for (size_t j = 0; j < weight_vec.size(); ++j) {
    auto& wv = weights->at(j);
    wv[shard_index] = atof(weight_vec[j].c_str());
  }

  EULER_LOG(INFO) << "Retrieve Shard Meta Info successfully, shard: "
                  << shard_index << ", Key: " << key
                  << ", Meta Info: " << weight;
  return true;
}

bool ClientManager::RetrieveShardMeta(
    int32_t shard_index, const std::string& key,
    std::unordered_set<std::string>* graph_label) {
  std::string shard_label;
  if (!server_monitor_->GetShardMeta(shard_index, key, &shard_label)) {
    EULER_LOG(ERROR) << "Retrieve shard meta failed, key: " << key
                     << " shard index: " << shard_index;
    return false;
  }

  std::vector<std::string> label_vec;
  label_vec = Split(shard_label, ',');
  for (std::string label : label_vec) {
    graph_label->insert(label);
  }
  return true;
}

bool ClientManager::RetrieveMeta(
    const std::string& key, std::string* value) {
  if (!server_monitor_->GetMeta(key, value)) {
    EULER_LOG(ERROR) << "Retrieve partition number from server failed!";
    return false;
  }
  return true;
}

bool ClientManager::InitGraphMeta() {
  std::string meta;
  if (!server_monitor_->GetShardMeta(0, "graph_meta", &meta)) {
    EULER_LOG(ERROR) << "get shard meta fail";
    return false;
  }

  if (!meta_.Deserialize(meta)) {
    EULER_LOG(ERROR) << "Inavlid meta got, meta size: " << meta.size();
    return false;
  }

  return true;
}

bool ClientManager::init_succ_ = false;
ClientManager* ClientManager::instance_ = nullptr;

}  // namespace euler
