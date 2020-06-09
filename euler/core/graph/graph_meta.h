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

#ifndef EULER_CORE_GRAPH_GRAPH_META_H_
#define EULER_CORE_GRAPH_GRAPH_META_H_

#include <stdint.h>

#include <vector>
#include <unordered_map>
#include <string>
#include <tuple>

namespace euler {

enum FeatureType {
  kSparse = 0,
  kDense = 1,
  kBinary = 2,
  UNK = 3
};

// type, id, dim
typedef std::tuple<FeatureType, int32_t, int64_t> FeatureInfo;

typedef std::unordered_map<
  std::string, std::tuple<FeatureType, int32_t, int64_t>> FeatureInfoMap;

class GraphMeta {
 public:
  GraphMeta()
      : name_("euler_default"),
        version_("0"),
        node_count_(0),
        edge_count_(0),
        partitions_num_(0) {
  }

  GraphMeta(std::string name,
            std::string version,
            uint64_t node_count,
            uint64_t edge_count,
            int partitions_num,
            const FeatureInfoMap& node_feature_info,
            const FeatureInfoMap& edge_feature_info,
            const std::unordered_map<std::string, uint32_t>& node_type_map,
            const std::unordered_map<std::string, uint32_t>& edge_type_map)
      : name_(name),
        version_(version),
        node_count_(node_count),
        edge_count_(edge_count),
        partitions_num_(partitions_num),
        node_feature_info_(node_feature_info),
        edge_feature_info_(edge_feature_info),
        node_type_map_(node_type_map),
        edge_type_map_(edge_type_map) {
  }

  void Init(const std::string& name,
            std::string version,
            uint64_t node_count,
            uint64_t edge_count,
            int partitions_num,
            const FeatureInfoMap& node_feature_info,
            const FeatureInfoMap& edge_feature_info,
            const std::unordered_map<std::string, uint32_t>& node_type_map,
            const std::unordered_map<std::string, uint32_t>& edge_type_map) {
    name_ = name;
    version_ = version;
    node_count_ = node_count;
    edge_count_ = edge_count;
    partitions_num_ = partitions_num;
    node_feature_info_ = node_feature_info;
    edge_feature_info_ = edge_feature_info;
    node_type_map_ = node_type_map;
    edge_type_map_ = edge_type_map;
  }

  FeatureType GetFeatureType(const std::string &feature_name) const;

  int32_t GetFeatureId(const std::string &feature_name) const;

  int64_t GetFeatureDim(const std::string &feature_name) const;

  FeatureInfo GetFeatureInfo(const std::string &feature_name) const;

  FeatureType GetEdgeFeatureType(const std::string &feature_name) const;

  int32_t GetEdgeFeatureId(const std::string &feature_name) const;

  int64_t GetEdgeFeatureDim(const std::string &feature_name) const;

  FeatureInfo GetEdgeFeatureInfo(const std::string &feature_name) const;

  const std::unordered_map<std::string, uint32_t>& node_type_map() const {
    return node_type_map_;
  }

  const std::unordered_map<std::string, uint32_t>& edge_type_map() const {
    return edge_type_map_;
  }

  uint64_t GetNodeCount() const;

  uint64_t GetEdgeCount() const;

  std::string ToString() const;

  bool Serialize(std::string* s);

  bool Deserialize(const std::string& s);

 private:
  friend class Graph;

  std::string name_;
  std::string version_;
  uint64_t node_count_;
  uint64_t edge_count_;
  int partitions_num_;
  FeatureInfoMap node_feature_info_;
  FeatureInfoMap edge_feature_info_;
  std::unordered_map<std::string, uint32_t> node_type_map_;
  std::unordered_map<std::string, uint32_t> edge_type_map_;
};

}  // namespace euler

#endif  // EULER_CORE_GRAPH_GRAPH_META_H_
