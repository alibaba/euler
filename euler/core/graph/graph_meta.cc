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


#include "euler/core/graph/graph_meta.h"

#include "euler/common/logging.h"

#include "euler/common/bytes_io.h"

namespace euler {

#define FIND_INFO(MAP, RES, DEF)                              \
  auto it = MAP.find(feature_name);                           \
  if (it == MAP.end()) {                                      \
    EULER_LOG(ERROR)                                          \
        << "Find feature error, Unknown feature name: "       \
        << feature_name;                                      \
    return DEF;                                               \
  }                                                           \
  return RES;


FeatureType GraphMeta::GetFeatureType(const std::string &feature_name) const {
  FIND_INFO(node_feature_info_, std::get<0>(it->second), UNK);
}

int32_t GraphMeta::GetFeatureId(const std::string &feature_name) const {
  FIND_INFO(node_feature_info_, std::get<1>(it->second), -1);
}

int64_t GraphMeta::GetFeatureDim(const std::string &feature_name) const {
  FIND_INFO(node_feature_info_, std::get<2>(it->second), -1);
}

FeatureInfo GraphMeta::GetFeatureInfo(const std::string &feature_name) const {
  FIND_INFO(node_feature_info_, it->second, std::make_tuple(UNK, 0, 0));
}

FeatureType GraphMeta::GetEdgeFeatureType(
    const std::string& feature_name) const {
  FIND_INFO(edge_feature_info_, std::get<0>(it->second), UNK);
}

int32_t GraphMeta::GetEdgeFeatureId(const std::string &feature_name) const {
  FIND_INFO(edge_feature_info_, std::get<1>(it->second), -1);
}

int64_t GraphMeta::GetEdgeFeatureDim(const std::string &feature_name) const {
  FIND_INFO(edge_feature_info_, std::get<2>(it->second), -1);
}

FeatureInfo GraphMeta::GetEdgeFeatureInfo(
    const std::string& feature_name) const {
  FIND_INFO(edge_feature_info_, it->second, std::make_tuple(UNK, 0, 0));
}

#undef FIND_INFO


uint64_t GraphMeta::GetNodeCount() const {
  return node_count_;
}

uint64_t GraphMeta::GetEdgeCount() const {
  return edge_count_;
}


std::string GraphMeta::ToString() const {
  std::stringstream ss;
  ss << "Name: " << name_ << ";\n";
  ss << "Version: " << version_ << ";\n";
  ss << "Node count:" << node_count_ << ";\n";
  ss << "Edge count: " << edge_count_ <<";\n";
  ss << "\n";

  ss << "Node feature info:\n";
  for (auto& it : node_feature_info_) {
    ss << "Name: " << it.first << ", " << "Type: ";
    switch (std::get<0>(it.second)) {
      case kSparse: ss << "Sparse"; break;
      case kBinary: ss << "Binary"; break;
      case kDense: ss << "Dense"; break;
      case UNK: ss << "UNK"; break;
    }
    ss << ", Dim: " << std::get<2>(it.second) << ";\n";
  }
  ss << "\n";

  ss << "Edge feature info:\n";
  for (auto& it : edge_feature_info_) {
    ss << "Name: " << it.first << ", " << "Type: ";
    switch (std::get<0>(it.second)) {
      case kSparse: ss << "Sparse"; break;
      case kBinary: ss << "Binary"; break;
      case kDense: ss << "Dense"; break;
      case UNK: ss << "UNK"; break;
    }
    ss << ", Dim: " << std::get<2>(it.second) << ";\n";
  }
  ss << "\n";

  ss << "Node type info: \n";
  for (auto& it : node_type_map_) {
    ss << "Name: " << it.first << ", Index: " << it.second << ";\n";
  }
  ss << "\n";

  ss << "Edge type info: \n";
  for (auto& it : edge_type_map_) {
    ss << "Name: " << it.first << ", Index: " << it.second << ";\n";
  }

  return ss.str();
}

bool GraphMeta::Serialize(std::string* s) {
  BytesWriter writer;
  writer.Write(name_);
  writer.Write(version_);
  writer.Write(node_count_);
  writer.Write(edge_count_);
  writer.Write(partitions_num_);

  uint32_t count = node_feature_info_.size();
  writer.Write(count);
  for (auto& item : node_feature_info_) {
    writer.Write(item.first);
    writer.Write(std::get<0>(item.second));
    writer.Write(std::get<1>(item.second));
    writer.Write(std::get<2>(item.second));
  }

  count = edge_feature_info_.size();
  writer.Write(count);
  for (auto& item : edge_feature_info_) {
    writer.Write(item.first);
    writer.Write(std::get<0>(item.second));
    writer.Write(std::get<1>(item.second));
    writer.Write(std::get<2>(item.second));
  }

  count = node_type_map_.size();
  writer.Write(count);
  for (auto& item : node_type_map_) {
    writer.Write(item.first);
    writer.Write(static_cast<uint32_t>(item.second));
  }

  count = edge_type_map_.size();
  writer.Write(count);
  for (auto& item : edge_type_map_) {
    writer.Write(item.first);
    writer.Write(static_cast<uint32_t>(item.second));
  }

  *s = writer.data();
  return true;
}

bool GraphMeta::Deserialize(const std::string& data) {
  BytesReader reader(data.c_str(), data.size());

#define READ_CHECK(cond)                                                \
  if (!(cond)) {                                                        \
    EULER_LOG(ERROR) << "Deserialize GraphMeta failed, line: "          \
                     << __LINE__;                                       \
    return false;                                                       \
}

  READ_CHECK(reader.Read(&name_));
  READ_CHECK(reader.Read(&version_));
  READ_CHECK(reader.Read(&node_count_));
  READ_CHECK(reader.Read(&edge_count_));
  READ_CHECK(reader.Read(&partitions_num_));

  uint32_t count;
  READ_CHECK(reader.Read(&count));
  for (uint32_t i = 0; i < count; i++) {
    std::string fname;
    FeatureType type;
    int32_t idx;
    int64_t dim;
    READ_CHECK(reader.Read(&fname));
    READ_CHECK(reader.Read(&type));
    READ_CHECK(reader.Read(&idx));
    READ_CHECK(reader.Read(&dim));
    node_feature_info_.insert(
        std::make_pair(
            fname, std::make_tuple(type, idx, dim)));
  }

  READ_CHECK(reader.Read(&count));
  for (uint32_t i = 0; i < count; i++) {
    std::string fname;
    FeatureType type;
    int32_t idx;
    int64_t dim;
    READ_CHECK(reader.Read(&fname));
    READ_CHECK(reader.Read(&type));
    READ_CHECK(reader.Read(&idx));
    READ_CHECK(reader.Read(&dim));
    edge_feature_info_.insert(
        std::make_pair(
            fname, std::make_tuple(type, idx, dim)));
  }

  READ_CHECK(reader.Read(&count));
  for (uint32_t i = 0; i < count; ++i) {
    std::string node_type;
    uint32_t index = 0;
    READ_CHECK(reader.Read(&node_type));
    READ_CHECK(reader.Read(&index));
    node_type_map_.insert({node_type, index});
  }

  READ_CHECK(reader.Read(&count));
  for (uint32_t i = 0; i < count; ++i) {
    std::string edge_type;
    uint32_t index = 0;
    READ_CHECK(reader.Read(&edge_type));
    READ_CHECK(reader.Read(&index));
    edge_type_map_.insert({edge_type, index});
  }

#undef READ_CHECK  // READ_CHECK

  return true;
}

}  // namespace euler
