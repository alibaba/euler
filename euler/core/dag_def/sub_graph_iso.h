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

#ifndef EULER_CORE_DAG_DEF_SUB_GRAPH_ISO_H_
#define EULER_CORE_DAG_DEF_SUB_GRAPH_ISO_H_

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

#include "euler/core/dag_def/dag_def.h"
namespace euler {
class MatchMap {
 public:
  explicit MatchMap(const DAGDef& dag_p) {
    for (auto& e : dag_p.GetNodeMap()) {
      un_match_.insert(e.first);
    }
  }

  void AddPair(int32_t node_p, int32_t node_m) {
    p2m_[node_p] = node_m;
    m2p_[node_m] = node_p;
    un_match_.erase(node_p);
  }

  void DeletePair(int32_t node_p, int32_t node_m) {
    p2m_.erase(node_p);
    m2p_.erase(node_m);
    un_match_.insert(node_p);
  }

  int32_t P2M(int32_t node_p) {
    if (p2m_.find(node_p) == p2m_.end()) {
      return -1;
    } else {
      return p2m_[node_p];
    }
  }

  int32_t M2P(int32_t node_m) {
    if (m2p_.find(node_m) == m2p_.end()) {
      return -1;
    } else {
      return m2p_[node_m];
    }
  }

  int32_t OfferUnMatch() {
    if (un_match_.empty()) {
      return -1;
    } else {
      return *(un_match_.begin());
    }
  }

  std::unordered_map<int32_t, int32_t> GetP2M() {
    return p2m_;
  }

 private:
  std::unordered_map<int32_t, int32_t> p2m_;

  std::unordered_map<int32_t, int32_t> m2p_;

  std::unordered_set<int32_t> un_match_;
};

bool NodeMatch(const NodeDef& node_m,
               const NodeDef& node_p,
               const std::unordered_map<std::string,
                     bool(*)(const NodeDef& node_m)>& extra_cond,
               MatchMap* match);

bool DAGMatch(const DAGDef& gm, const DAGDef& gp,
              const NodeDef& node_m, const NodeDef& node_p,
              const std::unordered_map<std::string,
                    bool(*)(const NodeDef& node_m)>& extra_cond,
              MatchMap* match);

std::vector<std::unordered_map<int32_t, int32_t>> SubGraphMatch(
    const DAGDef& gm, const DAGDef& gp,
    const std::unordered_map<std::string,
          bool(*)(const NodeDef& node_m)>& extra_cond);

}  // namespace euler
#endif  // EULER_CORE_DAG_DEF_SUB_GRAPH_ISO_H_
