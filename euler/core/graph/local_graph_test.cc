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
#include <iostream>

#include "gtest/gtest.h"

#include "euler/core/graph/graph_builder.h"
#include "euler/core/graph/compact_graph_factory.h"
#include "euler/core/graph/fast_graph_factory.h"
#include "euler/core/graph/graph.h"
#include "euler/core/graph/graph_engine.h"
#include "euler/common/data_types.h"
#include "euler/common/local_file_io.h"
#include "euler/common/timmer.h"

namespace euler {

Graph* GetCompactGraph() {
  CompactGraphFactory* factory = new CompactGraphFactory();
  GraphBuilder graph_builder(factory);
  std::vector<std::string> files = {"0.dat", "1.dat"};
  Graph* graph = graph_builder.BuildGraph(files, local, "", 0, all);
  return graph;
}

Graph* GetFastGraph() {
  FastGraphFactory* factory = new FastGraphFactory();
  GraphBuilder graph_builder(factory);
  std::vector<std::string> files = {"0.dat", "1.dat"};
  Graph* graph = graph_builder.BuildGraph(files, local, "", 0, all);
  return graph;
}

void CheckNodeInfo(Node* node) {
  ASSERT_EQ(3, node->GetID());
  ASSERT_EQ((float)3.0, node->GetWeight());
}


void CheckSampler(Node* node,
                  std::vector<int32_t> edge_types,
                  std::vector<int32_t> expct_cnt,
                  int32_t sample_num) {
  std::vector<euler::common::IDWeightPair> sample_id_weight_pair =
      node->SampleNeighbor(edge_types, sample_num);
  std::vector<int32_t> cnt(expct_cnt.size());
  for (size_t i = 0; i < sample_id_weight_pair.size(); ++i) {
    ++cnt[std::get<0>(sample_id_weight_pair[i])];
  }
  for (size_t i = 0; i < 5; ++i) {
    ASSERT_NEAR(expct_cnt[i], cnt[i], 200);
  }
}

#define CHECK_NEIGHBOR(NODE, EDGE_TYPE, EXPCT, METHOD, ...) { \
  std::vector<euler::common::IDWeightPair> id_weight_pair =   \
      NODE->METHOD(EDGE_TYPE, ##__VA_ARGS__);                 \
  for (size_t i = 0; i < id_weight_pair.size(); ++i) {        \
    ASSERT_EQ(std::get<0>(id_weight_pair[i]),                 \
              std::get<0>(EXPCT[i]));                         \
    ASSERT_EQ(std::get<1>(id_weight_pair[i]),                 \
              std::get<1>(EXPCT[i]));                         \
    ASSERT_EQ(std::get<2>(id_weight_pair[i]),                 \
              std::get<2>(EXPCT[i]));                         \
  }                                                           \
}                                                             \

TEST(LocalGraphTest, CheckNeighbor) {
  uint64_t node_id = 1;
  std::vector<int> edge_types = {0, 1};
  std::vector<std::tuple<int64_t, float, int32_t>> expect;
  expect.push_back(std::tuple<int64_t, float, int32_t>(2, 2, 0));
  expect.push_back(std::tuple<int64_t, float, int32_t>(4, 4, 0));
  expect.push_back(std::tuple<int64_t, float, int32_t>(3, 3, 1));
  std::vector<std::tuple<int64_t, float, int32_t>> expect2;
  expect2.push_back(std::tuple<int64_t, float, int32_t>(2, 2, 0));
  expect2.push_back(std::tuple<int64_t, float, int32_t>(3, 3, 1));
  expect2.push_back(std::tuple<int64_t, float, int32_t>(4, 4, 0));
  std::vector<std::tuple<int64_t, float, int32_t>> expect3;
  expect3.push_back(std::tuple<int64_t, float, int32_t>(4, 4, 0));
  expect3.push_back(std::tuple<int64_t, float, int32_t>(3, 3, 1));
  std::vector<std::tuple<int64_t, float, int32_t>> expect4;
  expect4.push_back(std::tuple<int64_t, float, int32_t>(4, 4, 0));
  expect4.push_back(std::tuple<int64_t, float, int32_t>(3, 3, 1));
  expect4.push_back(std::tuple<int64_t, float, int32_t>(2, 2, 0));
  std::vector<std::tuple<int64_t, float, int32_t>> expect1;
  expect1.push_back(std::tuple<int64_t, float, int32_t>(3, 3, 1));
  expect1.push_back(std::tuple<int64_t, float, int32_t>(5, 5, 1));
  std::vector<std::tuple<int64_t, float, int32_t>> expect12;
  expect12.push_back(std::tuple<int64_t, float, int32_t>(5, 5, 1));
  expect12.push_back(std::tuple<int64_t, float, int32_t>(3, 3, 1));

  std::vector<int> edge_types2 = {0};
  std::vector<std::tuple<int64_t, float, int32_t>> expect5;
  expect5.push_back(std::tuple<int64_t, float, int32_t>(2, 2, 0));
  expect5.push_back(std::tuple<int64_t, float, int32_t>(4, 4, 0));
  std::vector<std::tuple<int64_t, float, int32_t>> expect6;
  expect6.push_back(std::tuple<int64_t, float, int32_t>(2, 2, 0));
  expect6.push_back(std::tuple<int64_t, float, int32_t>(4, 4, 0));
  std::vector<std::tuple<int64_t, float, int32_t>> expect7;
  expect7.push_back(std::tuple<int64_t, float, int32_t>(4, 4, 0));
  expect7.push_back(std::tuple<int64_t, float, int32_t>(2, 2, 0));

  uint64_t node_id2 = 2;
  Graph* c_graph = GetCompactGraph();
  Graph* f_graph = GetFastGraph();
  Node* c_node = c_graph->GetNodeByID(node_id);
  Node* c_node2 = c_graph->GetNodeByID(node_id2);
  Node* f_node = f_graph->GetNodeByID(node_id);
  Node* f_node2 = f_graph->GetNodeByID(node_id2);

  CHECK_NEIGHBOR(c_node, edge_types, expect, GetFullNeighbor);
  CHECK_NEIGHBOR(f_node, edge_types, expect, GetFullNeighbor);
  CHECK_NEIGHBOR(c_node2, edge_types, expect1, GetFullNeighbor);
  CHECK_NEIGHBOR(f_node2, edge_types, expect1, GetFullNeighbor);

  CHECK_NEIGHBOR(c_node, edge_types, expect2, GetSortedFullNeighbor);
  CHECK_NEIGHBOR(f_node, edge_types, expect2, GetSortedFullNeighbor);
  CHECK_NEIGHBOR(c_node2, edge_types, expect1, GetSortedFullNeighbor);
  CHECK_NEIGHBOR(f_node2, edge_types, expect1, GetSortedFullNeighbor);

  CHECK_NEIGHBOR(c_node, edge_types, expect3, GetTopKNeighbor, 2);
  CHECK_NEIGHBOR(f_node, edge_types, expect3, GetTopKNeighbor, 2);
  CHECK_NEIGHBOR(c_node2, edge_types, expect12, GetTopKNeighbor, 3);
  CHECK_NEIGHBOR(f_node2, edge_types, expect12, GetTopKNeighbor, 3);

  CHECK_NEIGHBOR(f_node, edge_types, expect4, GetTopKNeighbor, 5);

  CHECK_NEIGHBOR(f_node, edge_types2, expect5, GetFullNeighbor);
  CHECK_NEIGHBOR(f_node, edge_types2, expect6, GetSortedFullNeighbor);
  CHECK_NEIGHBOR(f_node, edge_types2, expect7, GetTopKNeighbor, 3);
}

TEST(LocalGraphTest, CheckNodeInfo) {
  uint64_t node_id = 3;
  Graph* c_graph = GetCompactGraph();
  Node* c_node = c_graph->GetNodeByID(node_id);
  CheckNodeInfo(c_node);

  Graph* f_graph = GetFastGraph();
  Node* f_node = f_graph->GetNodeByID(node_id);
  CheckNodeInfo(f_node);
}

TEST(LocalGraphTest, CheckSampler) {
  uint64_t node_id = 3;
  std::vector<int32_t> edge_types = {0, 1};
  std::vector<int32_t> expct_cnt = {0, 0, 0, 0, 9000};
  uint64_t node_id2 = 3;
  std::vector<int32_t> edge_types2 = {1};
  std::vector<int32_t> expct_cnt2 = {0, 0, 0, 0, 0};
  uint64_t node_id3 = 1;
  std::vector<int32_t> edge_types3 = {0, 1};
  std::vector<int32_t> expct_cnt3 = {0, 0, 2000, 3000, 4000};
  uint64_t node_id4 = 1;
  std::vector<int32_t> edge_types4 = {0};
  std::vector<int32_t> expct_cnt4 = {0, 0, 2000, 0, 4000};

  Graph* c_graph = GetCompactGraph();
  Node* c_node = c_graph->GetNodeByID(node_id);
  CheckSampler(c_node, edge_types, expct_cnt, 9000);

  Graph* f_graph = GetFastGraph();
  Node* f_node = f_graph->GetNodeByID(node_id);
  CheckSampler(f_node, edge_types, expct_cnt, 9000);

  Node* c_node2 = c_graph->GetNodeByID(node_id2);
  CheckSampler(c_node2, edge_types2, expct_cnt2, 9000);

  Node* f_node2 = f_graph->GetNodeByID(node_id2);
  CheckSampler(f_node2, edge_types2, expct_cnt2, 9000);

  Node* c_node3 = c_graph->GetNodeByID(node_id3);
  CheckSampler(c_node3, edge_types3, expct_cnt3, 9000);

  Node* f_node3 = f_graph->GetNodeByID(node_id3);
  CheckSampler(f_node3, edge_types3, expct_cnt3, 9000);

  Node* c_node4 = c_graph->GetNodeByID(node_id4);
  CheckSampler(c_node4, edge_types4, expct_cnt4, 6000);

  Node* f_node4 = f_graph->GetNodeByID(node_id4);
  CheckSampler(f_node4, edge_types4, expct_cnt4, 6000);
}

#define CHECK_FLAT_FEATURE(FV, EFV, FN, EFN) { \
  for (size_t i = 0; i < EFV.size(); ++i) {    \
    ASSERT_EQ(FV[i], EFV[i]);                  \
  }                                            \
  for (size_t i = 0; i < EFN.size(); ++i) {    \
    ASSERT_EQ(FN[i], EFN[i]);                  \
  }                                            \
}

TEST(LocalGraphTest, CheckNodeFeatures) {
  uint64_t node_id = 3;
  Graph* c_graph = GetCompactGraph();
  Node* c_node = c_graph->GetNodeByID(node_id);

  std::vector<int32_t> float_feature_ids = {0, 1};
  std::vector<uint32_t> expect_float_feature_nums = {2, 3};
  std::vector<float> expect_float_feature_values =
      {2.4, 3.6, 4.5, 6.7, 8.9};

  std::vector<int32_t> uint64_feature_ids = {0, 1};
  std::vector<uint32_t> expect_uint64_feature_nums = {2, 0};
  std::vector<uint64_t> expect_uint64_feature_values =
      {1234, 5678};

  // error feature id
  std::vector<int32_t> err_uint64_feature_ids = {0, 100};
  std::vector<uint32_t> err_expect_uint64_feature_nums = {2, 0};
  std::vector<uint64_t> err_expect_uint64_feature_values =
      {1234, 5678};

  std::vector<int32_t> binary_feature_ids = {0, 1};
  std::vector<uint32_t> expect_binary_feature_nums = {3, 3};
  std::vector<char> expect_binary_feature_values =
      {'e', 'a', 'a', 'e', 'b', 'b'};

  std::vector<uint32_t> uint64_feature_nums;
  std::vector<uint64_t> uint64_feature_values;
  c_node->GetUint64Feature(uint64_feature_ids,
                           &uint64_feature_nums,
                           &uint64_feature_values);
  CHECK_FLAT_FEATURE(uint64_feature_values,
                     expect_uint64_feature_values,
                     uint64_feature_nums,
                     expect_uint64_feature_nums);

  std::vector<uint32_t> float_feature_nums;
  std::vector<float> float_feature_values;
  c_node->GetFloat32Feature(float_feature_ids,
                           &float_feature_nums,
                           &float_feature_values);
  CHECK_FLAT_FEATURE(float_feature_values,
                     expect_float_feature_values,
                     float_feature_nums,
                     expect_float_feature_nums);

  std::vector<uint32_t> err_uint64_feature_nums;
  std::vector<uint64_t> err_uint64_feature_values;
  c_node->GetUint64Feature(err_uint64_feature_ids,
                           &err_uint64_feature_nums,
                           &err_uint64_feature_values);
  CHECK_FLAT_FEATURE(err_uint64_feature_values,
                     err_expect_uint64_feature_values,
                     err_uint64_feature_nums,
                     err_expect_uint64_feature_nums);

  std::vector<uint32_t> binary_feature_nums;
  std::vector<char> binary_feature_values;
  c_node->GetBinaryFeature(binary_feature_ids,
                           &binary_feature_nums,
                           &binary_feature_values);
  CHECK_FLAT_FEATURE(binary_feature_values,
                     expect_binary_feature_values,
                     binary_feature_nums,
                     expect_binary_feature_nums);
}

void CheckEdgeInfo(Edge* edge) {
  ASSERT_EQ(std::get<0>(edge->GetID()), 1);
  ASSERT_EQ(std::get<1>(edge->GetID()), 2);
  ASSERT_EQ(std::get<2>(edge->GetID()), 0);
  ASSERT_EQ(edge->GetWeight(), (float)2.0);
}

TEST(LocalGraphTest, CheckEdgeIDFunc) {
  euler::common::EdgeID edge_id(1, 2, 0);
  euler::common::EdgeID edge_id_2(1, 4, 0);
  Graph* graph = GetCompactGraph();
  euler::core::UID uid = graph->EdgeIdToUID(edge_id);
  euler::core::UID uid2 = graph->EdgeIdToUID(edge_id_2);
  std::cout << "uid:" << uid << " uid2:" << uid2 << "\n";
}

TEST(LocalGraphTest, CheckEdgeInfo) {
  euler::common::EdgeID edge_id(1, 2, 0);
  Graph* c_graph = GetCompactGraph();
  Edge* c_edge = c_graph->GetEdgeByID(edge_id);
  CheckEdgeInfo(c_edge);

  Graph* f_graph = GetFastGraph();
  Edge* f_edge = f_graph->GetEdgeByID(edge_id);
  CheckEdgeInfo(f_edge);
}

TEST(LocalGraphTest, CheckEdgeFeatures) {
  euler::common::EdgeID edge_id(1, 2, 0);
  Graph* c_graph = GetCompactGraph();
  Edge* c_edge = c_graph->GetEdgeByID(edge_id);

  std::vector<int32_t> float_feature_ids = {0, 1};
  std::vector<uint32_t> expect_float_feature_nums = {2, 3};
  std::vector<float> expect_float_feature_values =
      {2.4, 3.6, 4.5, 6.7, 8.9};

  std::vector<int32_t> uint64_feature_ids = {0, 1};
  std::vector<uint32_t> expect_uint64_feature_nums = {2, 2};
  std::vector<uint64_t> expect_uint64_feature_values =
      {1234, 5678, 8888, 9999};

  // error feature id
  std::vector<int32_t> err_uint64_feature_ids = {0, 100};
  std::vector<uint32_t> err_expect_uint64_feature_nums = {2, 0};
  std::vector<uint64_t> err_expect_uint64_feature_values =
      {1234, 5678};

  std::vector<int32_t> binary_feature_ids = {0, 1};
  std::vector<uint32_t> expect_binary_feature_nums = {3, 3};
  std::vector<char> expect_binary_feature_values =
      {'e', 'a', 'a', 'e', 'b', 'b'};

  std::vector<uint32_t> uint64_feature_nums;
  std::vector<uint64_t> uint64_feature_values;
  c_edge->GetUint64Feature(uint64_feature_ids,
                           &uint64_feature_nums,
                           &uint64_feature_values);
  CHECK_FLAT_FEATURE(uint64_feature_values,
                     expect_uint64_feature_values,
                     uint64_feature_nums,
                     expect_uint64_feature_nums);

  std::vector<uint32_t> float_feature_nums;
  std::vector<float> float_feature_values;
  c_edge->GetFloat32Feature(float_feature_ids,
                           &float_feature_nums,
                           &float_feature_values);
  CHECK_FLAT_FEATURE(float_feature_values,
                     expect_float_feature_values,
                     float_feature_nums,
                     expect_float_feature_nums);

  std::vector<uint32_t> err_uint64_feature_nums;
  std::vector<uint64_t> err_uint64_feature_values;
  c_edge->GetUint64Feature(err_uint64_feature_ids,
                           &err_uint64_feature_nums,
                           &err_uint64_feature_values);
  CHECK_FLAT_FEATURE(err_uint64_feature_values,
                     err_expect_uint64_feature_values,
                     err_uint64_feature_nums,
                     err_expect_uint64_feature_nums);

  std::vector<uint32_t> binary_feature_nums;
  std::vector<char> binary_feature_values;
  c_edge->GetBinaryFeature(binary_feature_ids,
                           &binary_feature_nums,
                           &binary_feature_values);
  CHECK_FLAT_FEATURE(binary_feature_values,
                     expect_binary_feature_values,
                     binary_feature_nums,
                     expect_binary_feature_nums);
}


TEST(LocalGraphTest, GraphEngine) {
  // 测试graph engine
  GraphEngine graph_engine(compact);
  graph_engine.Initialize("./");

  std::cout<< "node_sample_distribution \n";
  std::vector<int> node_cnt;
  node_cnt.resize(7);
  for (size_t i = 0; i < 100000 ; i++) {
    node_cnt[graph_engine.SampleNode(0, 1)[0]]++;
  }
  for (size_t i = 0; i < node_cnt.size(); i++) {
    std::cout << " node" << i << " cnt:" << node_cnt[i];
  }
  std::cout << "\n";
  std::cout << "edge_sample_results: should be 123 456\n";
  std::vector<euler::common::EdgeID> edge_sample_results =
      graph_engine.SampleEdge(1, 10);
  for (size_t i = 0; i < edge_sample_results.size(); i++) {
    std::cout << std::get<0>(edge_sample_results[i]) << " "
        << std::get<1>(edge_sample_results[i]) << "\n";
  }
  std::cout << "\n";
}

/*
TEST(LocalGraphTest, GraphEngine) {
  GraphEngine graph_engine(compact);
  graph_engine.Initialize("./");

  std::vector<euler::common::NodeID> node_ids(50000);
  for (int i = 0; i < 50000; ++i) {
    node_ids[i] = i + 10000;
  }
  std::vector<int32_t> fids = {1};
  std::vector<uint32_t> feature_nums;
  std::vector<float> feature_values;
  euler::common::TimmerBegin();
  graph_engine.GetNodeFloat32Feature(node_ids, fids, &feature_nums, &feature_values);
  std::cout << euler::common::GetTimmerInterval() << std::endl;
}*/

}  // namespace euler
