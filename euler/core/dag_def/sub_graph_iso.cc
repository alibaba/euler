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

#include <string>
#include <unordered_set>
#include <vector>
#include <memory>
#include <utility>

#include "euler/core/dag_def/sub_graph_iso.h"
#include "euler/common/str_util.h"

namespace euler {

bool NodeMatch(const NodeDef& node_m,
               const NodeDef& node_p,
               const std::unordered_map<std::string,
                     bool(*)(const NodeDef& node_m)>& extra_cond,
               MatchMap* match) {
  // name unmatched
  if (node_m.name_ != node_p.name_) {
    return false;
  }
  std::string extra_cond_key = ToString(node_p.name_, ",", node_p.id_);
  if (extra_cond.find(extra_cond_key) != extra_cond.end() &&
      !extra_cond.at(extra_cond_key)(node_m)) {
    return false;
  }
  // pre struct unmatched
  if (node_p.pre_.size() > node_m.pre_.size()) {
    return false;
  }
  int32_t p_un_match_pre = 0;
  for (int32_t p_pre : node_p.pre_) {
    int32_t match_m_pre_id = match->P2M(p_pre);
    if (match_m_pre_id == -1) {
      ++p_un_match_pre;
    } else if (node_m.pre_.find(match_m_pre_id) ==
               node_m.pre_.end()) {
      return false;
    }
  }
  int32_t m_un_match_pre = 0;
  for (int32_t m_pre : node_m.pre_) {
    if (match->M2P(m_pre) == -1) {
      ++m_un_match_pre;
    }
  }
  if (p_un_match_pre > m_un_match_pre) {
    return false;
  }
  // succ struct unmatched
  if (node_p.succ_.size() > node_m.succ_.size()) {
    return false;
  }
  int32_t p_un_match_succ = 0;
  for (int32_t p_succ : node_p.succ_) {
    int32_t match_m_succ_id = match->P2M(p_succ);
    if (match_m_succ_id == -1) {
      ++p_un_match_succ;
    } else if (node_m.succ_.find(match_m_succ_id) ==
               node_m.succ_.end()) {
      return false;
    }
  }
  int32_t m_un_match_succ = 0;
  for (int32_t m_succ : node_m.succ_) {
    if (match->M2P(m_succ) == -1) {
      ++m_un_match_succ;
    }
  }
  if (p_un_match_succ > m_un_match_succ) {
    return false;
  }
  return true;
}

bool Match(const DAGDef& gm, const DAGDef& gp,
           const NodeDef& node_m, const NodeDef& node_p,
           const std::unordered_map<std::string,
                 bool(*)(const NodeDef& node_m)>& extra_cond,
           MatchMap* match) {
  if (match->P2M(node_p.id_) == node_m.id_) {  // already match
    int32_t un_match = match->OfferUnMatch();
    if (un_match == -1) {
      return true;
    }
    bool result = false;
    std::shared_ptr<NodeDef> un_match_node_p = gp.GetNodeById(un_match);
    for (auto& e : gm.GetNodeMap()) {
      result = result ||
               Match(gm, gp, *e.second, *un_match_node_p, extra_cond, match);
    }
    return result;
  } else if (match->P2M(node_p.id_) == -1 &&
             match->M2P(node_m.id_) == -1) {  // two candidates are both free
    if (NodeMatch(node_m, node_p, extra_cond, match)) {
      match->AddPair(node_p.id_, node_m.id_);
      int32_t node_p_succ_id = node_p.succ_.empty() ? -1 :
          (*node_p.succ_.begin());
      bool result = false;
      if (node_p_succ_id == -1) {
        result = Match(gm, gp, node_m, node_p, extra_cond, match);
      } else {
        for (int32_t node_m_succ_id : node_m.succ_) {
          result = result ||
              Match(gm, gp, *gm.GetNodeById(node_m_succ_id),
                    *gp.GetNodeById(node_p_succ_id), extra_cond, match);
        }
      }
      if (!result) {
        match->DeletePair(node_p.id_, node_m.id_);
      }
      return result;
    } else {
      return false;
    }
  } else {  // wrong candidates
    return false;
  }
}

std::vector<std::unordered_map<int32_t, int32_t>> SubGraphMatch(
    const DAGDef& gm, const DAGDef& gp,
    const std::unordered_map<std::string,
          bool(*)(const NodeDef& node_m)>& extra_cond) {
  std::unordered_set<int32_t> gm_matched_set;
  std::vector<std::unordered_map<int32_t, int32_t>> results;
  for (auto& gp_pair : gp.GetNodeMap()) {
    for (auto& gm_pair : gm.GetNodeMap()) {
      // didn't match before
      if (gm_matched_set.find(gm_pair.first) == gm_matched_set.end()) {
        MatchMap match(gp);
        Match(gm, gp, *gm_pair.second, *gp_pair.second,
              extra_cond, &match);
        if (!match.GetP2M().empty()) {
          results.push_back(match.GetP2M());
          for (const std::pair<int32_t, int32_t>& pair : match.GetP2M()) {
            gm_matched_set.insert(pair.second);
          }
        }
      }
    }
  }
  return results;
}

}  // namespace euler
