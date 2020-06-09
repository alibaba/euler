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

#include <vector>
#include <string>
#include <map>
#include <iostream>

#include "gtest/gtest.h"

#include "euler/common/logging.h"
#include "euler/core/graph/graph.h"
#include "euler/core/graph/graph_meta.h"
#include "euler/core/graph/graph_builder.h"
#include "euler/common/data_types.h"
#include "euler/common/env.h"

namespace euler {

#define CHECK_PAIR_VEC(V1, V2, V3) {         \
  ASSERT_EQ(V1.size(), V2.size());           \
  for (size_t i = 0; i < V1.size(); ++i) {   \
    ASSERT_EQ(std::get<0>(V1[i]), V2[i]);    \
    ASSERT_EQ(std::get<1>(V1[i]), V3[i]);    \
  }                                          \
}

#define CHECK_TUPLE_VEC(V1, V2) {                         \
  ASSERT_EQ(V1.size(), V2.size());                        \
  for (size_t i = 0; i < V1.size(); ++i) {                \
    ASSERT_EQ(std::get<0>(V1[i]), std::get<0>(V2[i]));    \
    ASSERT_EQ(std::get<1>(V1[i]), std::get<1>(V2[i]));    \
    ASSERT_EQ(std::get<2>(V1[i]), std::get<2>(V2[i]));    \
  }                                                       \
}

#define CHECK_VEC_VEC(V1, V2) {                 \
  ASSERT_EQ(V1.size(), V2.size());              \
  for (size_t i = 0; i < V1.size(); ++i) {      \
    ASSERT_EQ(V1[i].size(), V2[i].size());      \
    for (size_t j = 0; j < V1[i].size(); ++j) { \
    ASSERT_EQ(V1[i][j], V2[i][j]);              \
    }                                           \
  }                                             \
}

#define CHECK_VEC(V1, V2) {                     \
  ASSERT_EQ(V1.size(), V2.size());              \
  for (size_t i = 0; i < V1.size(); ++i) {      \
    ASSERT_EQ(V1[i], V2[i]);                    \
  }                                             \
}



/* Test Graph Adj:
 * 1: 2,3,4
 * 2: 3,5
 * 3: 4
 * 4: 5
 * 5: 2,6
 * 6: 1,3,5
 *
*/
Graph* GetLocalGraph() {
  Graph* g = &(Graph::Instance());
  static bool builded = false;
  if (!builded) {
    auto s = g->Init(0, 1, "all", "/tmp/euler", "all");

    if (s.ok()) {
      EULER_LOG(INFO) << "Build successfully!";
      builded = true;
    }
  }
  return g;
}


TEST(GraphTest, Node) {
  Graph* g = GetLocalGraph();
  ASSERT_TRUE(g != nullptr);
  {
    Node* node = g->GetNodeByID(1);
    std::vector<int32_t> edge_types{0};
    auto r = node->GetFullNeighbor(edge_types);

    std::vector<uint64_t> neighbor{2, 4};
    std::vector<float> weight{2.0, 4.0};
    CHECK_PAIR_VEC(r, neighbor, weight);
  }
  {
    std::vector<int32_t> fids{0, 1};
    std::vector<std::vector<uint64_t>> r;
    Node* node = g->GetNodeByID(1);
    node->GetUint64Feature(fids, &r);
    std::vector<std::vector<uint64_t>> expect;
    expect.push_back({11, 12});
    expect.push_back({12, 11});
    CHECK_VEC_VEC(r, expect);
  }
  {
    std::vector<int32_t> fids{0, 1};
    std::vector<std::vector<float>> r;
    Node* node = g->GetNodeByID(1);
    node->GetFloat32Feature(fids, &r);
    std::vector<std::vector<float>> expect;
    expect.push_back({1.1, 1.2});
    expect.push_back({1.3, 1.4, 1.5});
    CHECK_VEC_VEC(r, expect);
  }
  {
    std::vector<int32_t> fids{0};
    std::vector<std::string> r;
    Node* node = g->GetNodeByID(4);
    node->GetBinaryFeature(fids, &r);
    ASSERT_EQ(r.size(), 1);
    ASSERT_EQ(r[0], "4a");
  }
  {
    std::vector<int32_t> edge_types{0};
    Node* node = g->GetNodeByID(5);
    auto r = node->SampleNeighbor(edge_types, 100000);
    ASSERT_EQ(r.size(), 100000);
    std::vector<int32_t> cnts(10);
    for (auto i : r) {
      cnts[std::get<0>(i)] += 1;
    }
    ASSERT_TRUE(cnts[6] * 1.0 / cnts[2] > 2.8 && cnts[6]*1.0/cnts[2] < 3.2);
  }

  {
    std::vector<int32_t> edge_types{0};
    Node* node = g->GetNodeByID(3);
    auto r = node->SampleNeighbor(edge_types, 100000);
    ASSERT_EQ(r.size(), 100000);
    std::vector<int32_t> cnts(10);
    for (auto i : r) {
      cnts[std::get<0>(i)] += 1;
    }
    ASSERT_EQ(cnts[4], 100000);
  }

  {
    euler::common::EdgeID eid(1, 2, 0);
    Graph::UID uid =  g->EdgeIdToUID(eid);
    EULER_LOG(INFO) << "eid to uid:" << uid;
    eid = g->UIDToEdgeId(uid);
    EULER_LOG(INFO) << "uid to eid:(" << std::get<0>(eid) << ","
                    << std::get<1>(eid) << "," << std::get<2>(eid) << ")";
    ASSERT_EQ(std::get<0>(eid), 1);
    ASSERT_EQ(std::get<1>(eid), 2);
    ASSERT_EQ(std::get<2>(eid), 0);
  }
}

TEST(GraphTest, SampleNode) {
  auto g = GetLocalGraph();
  EULER_LOG(INFO)<< "node_sample_distribution";
  {
    std::vector<int> node_cnt;
    node_cnt.resize(7);
    for (size_t i = 0; i < 420000 ; i++) {
      node_cnt[g->SampleNode(-1, 1)[0]]++;
    }
    for (size_t i = 0; i < node_cnt.size(); i++) {
      EULER_LOG(INFO) << " node" << i << " cnt:" << node_cnt[i];
    }

    for (size_t i = 2; i < node_cnt.size(); i++) {
      float ratio = node_cnt[i]*1.0/node_cnt[i-1];
      float truth = 1.0 * i / (i - 1);
      EULER_LOG(INFO) << "i:" << i << " ratio"
                      << node_cnt[i] * 1.0 / node_cnt[i-1]
                      << " answer:" << truth;
      ASSERT_TRUE(ratio > truth * 0.98 && ratio < truth * 1.02);
    }
  }
  {
    std::vector<int> node_cnt;
    node_cnt.resize(7);
    for (size_t i = 0; i < 420000 ; i++) {
      node_cnt[g->SampleNode({0, 1}, 1)[0]]++;
    }
    for (size_t i = 0; i < node_cnt.size(); i++) {
      EULER_LOG(INFO) << " node" << i << " cnt:" << node_cnt[i];
    }

    for (size_t i = 2; i < node_cnt.size(); i++) {
      float ratio = node_cnt[i]*1.0/node_cnt[i-1];
      float truth = 1.0 * i / (i - 1);
      EULER_LOG(INFO) << "i:" << i << " ratio"
                      << node_cnt[i] * 1.0 / node_cnt[i-1]
                      << " answer:" << truth;
      ASSERT_TRUE(ratio > truth * 0.98 && ratio < truth * 1.02);
    }
  }
  std::vector<euler::common::EdgeID> edge_sample_results =
      g->SampleEdge(1, 10);
  for (size_t i = 0; i < edge_sample_results.size(); i++) {
    EULER_LOG(INFO) << std::get<0>(edge_sample_results[i]) << " "
                    << std::get<1>(edge_sample_results[i]);
  }

  ASSERT_EQ(g->GetNodeFeatureId("sparse_f1"), 0);
  ASSERT_EQ(g->GetNodeFeatureId("dense_f3"), 0);
  ASSERT_EQ(g->GetNodeFeatureType("sparse_f2"), kSparse);
}


TEST(GraphMetaTest, GraphMetaDeserialize) {
  std::unique_ptr<FileIO> reader;
  ASSERT_TRUE(
    Env::Default()->NewFileIO("/tmp/euler/euler.meta", true, &reader).ok());

  std::vector<std::string> meta_infos;
  std::string name, version;
  uint64_t node_count, edge_count;
  int partitions_num = 0;

  ASSERT_TRUE(reader->Read(&name));
  ASSERT_TRUE(reader->Read(&version));

  ASSERT_TRUE(reader->Read(&node_count));
  ASSERT_TRUE(reader->Read(&edge_count));

  ASSERT_TRUE(reader->Read(&partitions_num));
  EULER_LOG(INFO) << partitions_num;

  uint32_t node_meta_count;
  ASSERT_TRUE(reader->Read(&node_meta_count));

  FeatureInfoMap nfm, efm;
  for (size_t i = 0; i < node_meta_count; i++) {
    std::string fname;
    FeatureType type;
    int32_t idx;
    int64_t dim;
    ASSERT_TRUE(reader->Read(&fname));
    ASSERT_TRUE(reader->Read(&type));
    ASSERT_TRUE(reader->Read(&idx));
    ASSERT_TRUE(reader->Read(&dim));

    nfm.insert(
        std::make_pair(
            fname, std::make_tuple(type, idx, dim)));
  }
  uint32_t edge_meta_count;
  ASSERT_TRUE(reader->Read(&edge_meta_count));
  EULER_LOG(INFO) << edge_meta_count;
  for (size_t i = 0; i < edge_meta_count; i++) {
    std::string fname;
    FeatureType type;
    int32_t idx;
    int64_t dim;
    ASSERT_TRUE(reader->Read(&fname));
    ASSERT_TRUE(reader->Read(&type));
    ASSERT_TRUE(reader->Read(&idx));
    ASSERT_TRUE(reader->Read(&dim));
    efm.insert(
        std::make_pair(fname, std::make_tuple(type, idx, dim)));
  }

  std::unordered_map<std::string, uint32_t> type_map;
  GraphMeta meta(name, version, node_count, edge_count,
                 partitions_num, nfm, efm, type_map, type_map);

  ASSERT_EQ(meta.GetFeatureId("sparse_f1"), 0);
  ASSERT_EQ(meta.GetFeatureId("dense_f3"), 0);
  ASSERT_EQ(meta.GetFeatureType("sparse_f2"), kSparse);
  ASSERT_EQ(meta.GetFeatureType("dense_f3"), kDense);
  ASSERT_EQ(meta.GetFeatureDim("dense_f4"), 3);
}

TEST(GraphTest, GraphSerialize) {
  Graph* g = GetLocalGraph();
  std::unique_ptr<FileIO> writer;
  ASSERT_TRUE(Env::Default()->NewFileIO(
      "graph_test_com.dat", false, &writer).ok());
  ASSERT_TRUE(g->Dump(writer.get()));
}

#define CHECK_FLAT_FEATURE(FV, EFV) {          \
  for (size_t i = 0; i < EFV.size(); ++i) {    \
    ASSERT_EQ(FV[i], EFV[i]);                  \
  }                                            \
}

TEST(GraphTest, CheckNodeFeature) {
  auto g = GetLocalGraph();
  ASSERT_EQ(g->GetNodeFeatureId("sparse_f1"), 0);
  ASSERT_EQ(g->GetNodeFeatureId("dense_f3"), 0);
  ASSERT_EQ(g->GetNodeFeatureId("binary_f6"), 1);
  ASSERT_EQ(g->GetNodeFeatureType("sparse_f1"), kSparse);
  ASSERT_EQ(g->GetNodeFeatureType("dense_f3"), kDense);
  ASSERT_EQ(g->GetNodeFeatureType("binary_f5"), kBinary);
  uint64_t node_id = 3;
  auto node = g->GetNodeByID(node_id);
  std::vector<uint32_t> float_feature_nums;
  std::vector<float> float_feature_values;
  std::vector<int32_t> float_feature_ids = {0, 1};
  std::vector<uint32_t> expect_float_feature_nums = {2, 3};
  node->GetFloat32Feature(float_feature_ids,
                          &float_feature_nums,
                          &float_feature_values);
  CHECK_FLAT_FEATURE(float_feature_nums,
                 expect_float_feature_nums);

  std::vector<uint32_t> float_feature_nums_err;
  std::vector<float> float_feature_values_err;
  std::vector<int32_t> float_feature_ids_err = {0, 99};
  std::vector<uint32_t> expect_float_feature_nums_err = {2, 0};
  node->GetFloat32Feature(float_feature_ids_err,
                          &float_feature_nums_err,
                          &float_feature_values_err);
  CHECK_FLAT_FEATURE(float_feature_nums_err,
                 expect_float_feature_nums_err);
}

TEST(GraphTest, CheckEdgeInfo) {
  auto g = GetLocalGraph();
  euler::common::EdgeID edge_id(1, 2, 0);
  Edge* edge = g->GetEdgeByID(edge_id);
  ASSERT_EQ(std::get<0>(edge->GetID()), 1);
  ASSERT_EQ(std::get<1>(edge->GetID()), 2);
  ASSERT_EQ(std::get<2>(edge->GetID()), 0);
  ASSERT_EQ(edge->GetWeight(), (float)2.0);

  euler::common::EdgeID edge_id1(6, 3, 1);
  Edge* edge1 = g->GetEdgeByID(edge_id1);
  ASSERT_EQ(std::get<0>(edge1->GetID()), 6);
  ASSERT_EQ(std::get<1>(edge1->GetID()), 3);
  ASSERT_EQ(std::get<2>(edge1->GetID()), 1);
  ASSERT_EQ(edge1->GetWeight(), (float)3.0);
}

#define CHECK_NEIGHBOR(NODE, EDGE_TYPE, METHOD, ...) {            \
    std::vector<euler::common::IDWeightPair> id_weight_pair =     \
        NODE->METHOD(EDGE_TYPE, ##__VA_ARGS__);                   \
    ASSERT_EQ(id_weight_pair.size(), 3);                          \
    for (size_t i = 0; i < id_weight_pair.size(); ++i) {          \
      std::cout << std::get<0>(id_weight_pair[i]) <<              \
          " : " << std::get<1>(id_weight_pair[i]) <<              \
          " : " << std::get<2>(id_weight_pair[i]) << std::endl;   \
    }                                                             \
}                                                                 \

TEST(GraphTest, CheckNeighbor) {
  auto g = GetLocalGraph();
  std::vector<int> edge_types = {0, 1};
  uint64_t node_id1 = 1;
  std::cout << "get neighbor node id: " << node_id1 << std::endl;
  Node* node1 = g->GetNodeByID(node_id1);
  CHECK_NEIGHBOR(node1, edge_types, GetFullNeighbor);
  CHECK_NEIGHBOR(node1, edge_types, GetSortedFullNeighbor);
  CHECK_NEIGHBOR(node1, edge_types, GetTopKNeighbor, 3);
}

#define CHECK_NEIGHBOR_COUNT(NODE, EDGE_TYPE, METHOD) {         \
    std::vector<euler::common::IDWeightPair> id_weight_pair =   \
        NODE->METHOD(EDGE_TYPE, 1);                             \
    ASSERT_EQ(id_weight_pair.size(), 1);                        \
  }

#define CHECK_NEIGHBOR_TYPE(NODE, EDGE_TYPE, METHOD, ...) {     \
    std::vector<euler::common::IDWeightPair> id_weight_pair =   \
        NODE->METHOD(EDGE_TYPE, ##__VA_ARGS__);                 \
    for (size_t i = 0; i < id_weight_pair.size(); ++i) {        \
      ASSERT_EQ(std::get<2>(id_weight_pair[i]), 0);             \
    }                                                           \
}

TEST(GraphTest, CheckGraph) {
  auto g = GetLocalGraph();
  int32_t node_type = 0;
  for (size_t i = 0; i < 100; i++) {
    auto sample_result = g->SampleNode(node_type, 1);
    auto one_sample_id = sample_result[0];
    auto one_sample = g->GetNodeByID(one_sample_id);
    // check node info
    ASSERT_EQ((float)one_sample_id, one_sample->GetWeight());
    ASSERT_EQ(node_type, one_sample->GetType());
    // std::cout << "node info done" << std::endl;
    // check node feature
    std::vector<uint32_t> float_feature_nums;
    std::vector<float> float_feature_values;
    std::vector<int32_t> float_feature_ids = {0, 1};
    std::vector<uint32_t> expect_float_feature_nums = {2, 3};
    one_sample->GetFloat32Feature(float_feature_ids,
                                  &float_feature_nums,
                                  &float_feature_values);
    CHECK_FLAT_FEATURE(float_feature_nums,
                       expect_float_feature_nums);
    // std::cout << "node fea done" << std::endl;
    // check neighbor
    std::vector<int> edge_types = {0};
    CHECK_NEIGHBOR_TYPE(one_sample, edge_types, GetFullNeighbor);
    std::vector<int> edge_types1 = {0, 1};
    CHECK_NEIGHBOR_COUNT(one_sample, edge_types1, GetTopKNeighbor);
    // check neighbor sample and edge info
    // std::cout << "--sample neighbor test--" << endl;
    std::vector<int> edge_types0 = {0};
    std::vector<euler::common::IDWeightPair> sample_id_weight_pair =
        one_sample->SampleNeighbor(edge_types0, 1);
    if (sample_id_weight_pair.size() > 0) {
      auto one_nei = std::get<0>(sample_id_weight_pair[0]);
      euler::common::EdgeID sample_edge_id(one_sample_id, one_nei, 0);
      Edge* sample_edge = g->GetEdgeByID(sample_edge_id);
      ASSERT_EQ(std::get<0>(sample_edge->GetID()), one_sample_id);
      ASSERT_EQ(std::get<1>(sample_edge->GetID()), one_nei);
      ASSERT_EQ(std::get<2>(sample_edge->GetID()), one_nei%2);
      ASSERT_EQ(sample_edge->GetWeight(), (float)one_nei);
    }
  }
}

#define CHECK_NEI_SAMPLE(NODE, EDGE_TYPE,                               \
                         EXPCT_CNT, SAMPLE_NUM, METHOD, ...) {          \
  std::vector<euler::common::IDWeightPair>                              \
      sample_id_weight_pair = NODE->METHOD(EDGE_TYPE, SAMPLE_NUM);      \
  std::vector<int32_t> cnt(EXPCT_CNT.size());                           \
  for (size_t i = 0; i < sample_id_weight_pair.size(); ++i) {           \
    auto one_nei = std::get<0>(sample_id_weight_pair[i]);               \
    ++cnt[one_nei];                                                     \
  }                                                                     \
  for (size_t i = 0; i < expct_cnt.size(); ++i) {                       \
    ASSERT_NEAR(EXPCT_CNT[i], cnt[i], 200);                             \
  }                                                                     \
}


TEST(GraphTest, CheckSampleNeighbor) {
  auto g = GetLocalGraph();
  std::vector<int> edge_types = {0, 1};
  std::vector<int> edge_types0 = {0};
  std::vector<int> edge_types1 = {1};
  uint64_t node_id = 1;
  Node* node = g->GetNodeByID(node_id);
  std::vector<int32_t> expct_cnt = {0, 0, 2000, 3000, 4000, 0, 0};
  std::vector<int32_t> expct_cnt0 = {0, 0, 2000, 0, 4000, 0, 0};
  std::vector<int32_t> expct_cnt1 = {0, 0, 0, 9000, 0, 0, 0};
  CHECK_NEI_SAMPLE(node, edge_types, expct_cnt, 9000, SampleNeighbor);
  CHECK_NEI_SAMPLE(node, edge_types0, expct_cnt0, 6000, SampleNeighbor);
  CHECK_NEI_SAMPLE(node, edge_types1, expct_cnt1, 9000, SampleNeighbor);
}

TEST(GraphTest, CheckSampleInNeighbor) {
  {
    auto g = GetLocalGraph();
    std::vector<int> edge_types = {0, 1};
    std::vector<int> edge_types0 = {0};
    std::vector<int> edge_types1 = {1};
    uint64_t node_id = 3;
    Node* node = g->GetNodeByID(node_id);
    std::vector<int32_t> expct_cnt = {0, 3000, 3000, 0, 0, 0, 3000};
    std::vector<int32_t> expct_cnt0 = {0, 0, 0, 0, 0, 0, 0};
    std::vector<int32_t> expct_cnt1 = {0, 3000, 3000, 0, 0, 0, 3000};
    CHECK_NEI_SAMPLE(node, edge_types, expct_cnt, 9000, SampleInNeighbor);
    CHECK_NEI_SAMPLE(node, edge_types0, expct_cnt0, 9000, SampleInNeighbor);
    CHECK_NEI_SAMPLE(node, edge_types1, expct_cnt1, 9000, SampleInNeighbor);
  }
  {
    auto g = GetLocalGraph();
    std::vector<int> edge_types = {0, 1};
    std::vector<int> edge_types0 = {0};
    std::vector<int> edge_types1 = {1};
    uint64_t node_id = 5;
    Node* node = g->GetNodeByID(node_id);
    std::vector<int32_t> expct_cnt = {0, 0, 3000, 0, 3000, 0, 3000};
    std::vector<int32_t> expct_cnt0 = {0, 0, 0, 0, 0, 0, 0};
    std::vector<int32_t> expct_cnt1 = {0, 0, 3000, 0, 3000, 0, 3000};
    CHECK_NEI_SAMPLE(node, edge_types, expct_cnt, 9000, SampleInNeighbor);
    CHECK_NEI_SAMPLE(node, edge_types0, expct_cnt0, 9000, SampleInNeighbor);
    CHECK_NEI_SAMPLE(node, edge_types1, expct_cnt1, 9000, SampleInNeighbor);
  }
}

#undef CHECK_NEI_SAMPLE
}  // namespace euler
