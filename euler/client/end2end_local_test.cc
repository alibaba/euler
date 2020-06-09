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
#include <vector>

#include "gtest/gtest.h"

#include "euler/common/logging.h"
#include "euler/common/signal.h"
#include "euler/common/data_types.h"
#include "euler/client/query_proxy.h"
#include "euler/client/query.h"
#include "euler/parser/optimizer.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/types.pb.h"
#include "euler/core/api/api.h"

namespace euler {

TEST(End2EndLocalTest, Execute) {
  GraphConfig graph_config;
  graph_config.Add("data_path", "/tmp/euler");
  graph_config.Add("sampler_type", "all");
  graph_config.Add("data_type", "all");
  graph_config.Add("mode", "local");

  QueryProxy::Init(graph_config);

  QueryProxy* proxy = QueryProxy::GetInstance();

  {
    Query* query = new Query(
      "sampleN(node_type, count).as(node_id).values(fid).as(p3)");
    std::string fid = "sparse_f1";
    TensorShape shape({1});
    TensorShape fid_shape({1});
    Tensor* node_type_t = query->AllocInput("node_type", shape, kInt32);
    Tensor* count_t = query->AllocInput("count", shape, kInt32);
    Tensor* fid_t = query->AllocInput("fid", fid_shape, kString);
    int32_t node_type = 0;
    GetNodeType("0", &node_type);
    *(node_type_t->Raw<int32_t>()) = node_type;
    *(count_t->Raw<int32_t>()) = 100000;
    *(fid_t->Raw<std::string*>()[0]) = fid;

    Signal s;
    std::vector<std::string> result_names = {"node_id:0", "p3:0", "p3:1"};
    std::function<void()> callback = [query, &s, result_names](){
      std::unordered_map<std::string, Tensor*> results_map = query->GetResult(
          result_names);

      std::vector<uint64_t> ids(7);
      ASSERT_EQ(results_map["node_id:0"]->NumElements(), 100000);
      for (int32_t i = 0; i < results_map["node_id:0"]->NumElements(); ++i) {
        auto id = results_map["node_id:0"]->Raw<uint64_t>()[i];
        ids[id] += 1;
      }
      ASSERT_TRUE(ids[4] * 1.0 / ids[2] > 1.9 && ids[4] * 1.0 / ids[2] < 2.1);
      ASSERT_TRUE(ids[6] * 1.0 / ids[2] > 2.9 && ids[6] * 1.0 / ids[2] < 3.1);

      for (int32_t i = 0; i < 100; ++i) {
        auto start = results_map["p3:0"]->Raw<int32_t>()[i * 2];
        auto end = results_map["p3:0"]->Raw<int32_t>()[i * 2 + 1];
        ASSERT_EQ(start + 2, end);
        auto nodeid = results_map["node_id:0"]->Raw<uint64_t>()[i];
        if (nodeid == 2) {
          ASSERT_EQ(results_map["p3:1"]->Raw<uint64_t>()[start], 21);
          ASSERT_EQ(results_map["p3:1"]->Raw<uint64_t>()[start + 1], 22);
        } else if (nodeid == 4) {
          ASSERT_EQ(results_map["p3:1"]->Raw<uint64_t>()[start], 41);
          ASSERT_EQ(results_map["p3:1"]->Raw<uint64_t>()[start + 1], 42);
        } else {
          ASSERT_EQ(results_map["p3:1"]->Raw<uint64_t>()[start], 61);
          ASSERT_EQ(results_map["p3:1"]->Raw<uint64_t>()[start + 1], 62);
        }
      }
      delete query;
      s.Notify();
    };
    proxy->RunAsyncGremlin(query, callback);
    s.Wait();
  }

  {
    std::string gremlin = "e(edges).values(fid).as(e_feature)";
    Query query(gremlin);
    TensorShape shape({3, 3});
    TensorShape fid_shape({1});
    Tensor* edges_t = query.AllocInput("edges", shape, kUInt64);
    Tensor* fid_t = query.AllocInput("fid", fid_shape, kString);
    std::vector<uint64_t> edges = {6, 1, 1, 5, 6, 0, 4, 5, 1};
    std::copy(edges.begin(), edges.end(), edges_t->Raw<uint64_t>());
    std::string fid = "sparse_f1";
    *(fid_t->Raw<std::string*>()[0]) = fid;
    std::vector<std::string> result_names = {"e_feature:0", "e_feature:1"};
    std::unordered_map<std::string, Tensor*> results_map =
        proxy->RunGremlin(&query, result_names);
    std::vector<uint64_t> e_f_values = {611, 612, 561, 562, 451, 452};
    for (int32_t i = 0; i < results_map["e_feature:1"]->NumElements(); ++i) {
      ASSERT_EQ(e_f_values[i], results_map["e_feature:1"]->Raw<uint64_t>()[i]);
    }
  }

  {
    std::string gremlin = "e(edges).values(fid1, fid2).min(fid2).as(e_feature)";
    Query query(gremlin);
    TensorShape shape({3, 3});
    TensorShape fid_shape({1});
    Tensor* edges_t = query.AllocInput("edges", shape, kUInt64);
    Tensor* fid1_t = query.AllocInput("fid1", fid_shape, kString);
    Tensor* fid2_t = query.AllocInput("fid2", fid_shape, kString);
    std::vector<uint64_t> edges = {6, 1, 1, 5, 6, 0, 4, 5, 1};
    std::copy(edges.begin(), edges.end(), edges_t->Raw<uint64_t>());
    std::string fid1 = "sparse_f1";
    std::string fid2 = "sparse_f2";
    *(fid1_t->Raw<std::string*>()[0]) = fid1;
    *(fid2_t->Raw<std::string*>()[0]) = fid2;
    std::vector<std::string> result_names = {"e_feature:0", "e_feature:1",
                                             "e_feature:2", "e_feature:3"};
    std::unordered_map<std::string, Tensor*> results_map =
        proxy->RunGremlin(&query, result_names);
    std::vector<uint64_t> e_f_values1 = {611, 612, 561, 562, 451, 452};
    for (int32_t i = 0; i < results_map["e_feature:1"]->NumElements(); ++i) {
      ASSERT_EQ(e_f_values1[i], results_map["e_feature:1"]->Raw<uint64_t>()[i]);
    }
    std::vector<uint64_t> e_f_values2 = {613, 563, 453};
    for (int32_t i = 0; i < results_map["e_feature:3"]->NumElements(); ++i) {
      ASSERT_EQ(e_f_values2[i], results_map["e_feature:3"]->Raw<uint64_t>()[i]);
    }
  }

  {
    std::string gremlin = "e(edges).values(fid1, fid2).max(fid2).as(e_feature)";
    Query query(gremlin);
    TensorShape shape({3, 3});
    TensorShape fid_shape({1});
    Tensor* edges_t = query.AllocInput("edges", shape, kUInt64);
    Tensor* fid1_t = query.AllocInput("fid1", fid_shape, kString);
    Tensor* fid2_t = query.AllocInput("fid2", fid_shape, kString);
    std::vector<uint64_t> edges = {6, 1, 1, 5, 6, 0, 4, 5, 1};
    std::copy(edges.begin(), edges.end(), edges_t->Raw<uint64_t>());
    std::string fid1 = "sparse_f1";
    std::string fid2 = "sparse_f2";
    *(fid1_t->Raw<std::string*>()[0]) = fid1;
    *(fid2_t->Raw<std::string*>()[0]) = fid2;
    std::vector<std::string> result_names = {"e_feature:0", "e_feature:1",
                                             "e_feature:2", "e_feature:3"};
    std::unordered_map<std::string, Tensor*> results_map =
        proxy->RunGremlin(&query, result_names);
    std::vector<uint64_t> e_f_values1 = {611, 612, 561, 562, 451, 452};
    for (int32_t i = 0; i < results_map["e_feature:1"]->NumElements(); ++i) {
      ASSERT_EQ(e_f_values1[i], results_map["e_feature:1"]->Raw<uint64_t>()[i]);
    }
    std::vector<uint64_t> e_f_values2 = {614, 564, 454};
    for (int32_t i = 0; i < results_map["e_feature:3"]->NumElements(); ++i) {
      ASSERT_EQ(e_f_values2[i], results_map["e_feature:3"]->Raw<uint64_t>()[i]);
    }
  }

  {
    std::string gremlin = "e(edges).values(fid3).mean(fid3).as(e_feature)";
    Query query(gremlin);
    TensorShape shape({3, 3});
    TensorShape fid_shape({1});
    Tensor* edges_t = query.AllocInput("edges", shape, kUInt64);
    Tensor* fid1_t = query.AllocInput("fid3", fid_shape, kString);
    std::vector<uint64_t> edges = {6, 1, 1, 5, 6, 0, 4, 5, 1};
    std::copy(edges.begin(), edges.end(), edges_t->Raw<uint64_t>());
    std::string fid3 = "dense_f3";
    *(fid1_t->Raw<std::string*>()[0]) = fid3;
    std::vector<std::string> result_names = {"e_feature:0", "e_feature:1"};
    std::unordered_map<std::string, Tensor*> results_map =
        proxy->RunGremlin(&query, result_names);
    std::vector<float> e_f_values = {61.15, 56.15, 45.15};
    for (int32_t i = 0; i < results_map["e_feature:1"]->NumElements(); ++i) {
      ASSERT_EQ(e_f_values[i], results_map["e_feature:1"]->Raw<float>()[i]);
    }
  }

  {
    std::string gremlin = R"(sampleN(node_type, n_count).as(node_id).select(node_id).outV(edge_types).order_by(id, asc).limit(3).as(nb).values(fid).as(nb_feature).v_select(node_id).values(fid).as(n_feature))";
    Query query(gremlin);

    TensorShape shape({1});
    TensorShape shape1({2});
    TensorShape fid_shape({1});
    Tensor* node_type_t = query.AllocInput("node_type", shape, kInt32);
    Tensor* n_count_t = query.AllocInput("n_count", shape, kInt32);
    Tensor* edge_types_t = query.AllocInput("edge_types", shape1, kInt32);
    Tensor* nb_count_t = query.AllocInput("nb_count", shape, kInt32);
    Tensor* fid_t = query.AllocInput("fid", fid_shape, kString);
    int32_t node_type = 0;
    GetNodeType("0", &node_type);
    *(node_type_t->Raw<int32_t>()) = node_type;
    *(n_count_t->Raw<int32_t>()) = 10;
    std::vector<int32_t> edge_types = {0, 0};
    GetEdgeType("0", &edge_types[0]);
    GetEdgeType("1", &edge_types[1]);
    edge_types_t->Raw<int32_t>()[0] = edge_types[0];
    edge_types_t->Raw<int32_t>()[1] = edge_types[1];
    *(nb_count_t->Raw<int32_t>()) = 5;
    std::string fid = "sparse_f1";
    *(fid_t->Raw<std::string*>()[0]) = fid;

    std::vector<std::string> result_names = {"node_id:0", "nb:0", "nb:1",
                                             "nb_feature:0", "nb_feature:1",
                                              "n_feature:0", "n_feature:1"};
    std::unordered_map<std::string, Tensor*> results_map =
        proxy->RunGremlin(&query, result_names);

    std::unordered_map<uint64_t, std::vector<uint64_t>> node_nb_set;
    std::unordered_map<uint64_t, std::vector<uint64_t>> node_f_set;
    node_nb_set[2] = {3, 5}; node_nb_set[4] = {5}; node_nb_set[6] = {1, 3, 5};
    node_f_set[1] = {11, 12}; node_f_set[2] = {21, 22};
    node_f_set[3] = {31, 32}; node_f_set[4] = {41, 42};
    node_f_set[5] = {51, 52}; node_f_set[6] = {61, 62};

    std::vector<int64_t> node_ids;
    ASSERT_EQ(10, results_map["node_id:0"]->NumElements());
    for (int32_t i = 0; i < results_map["node_id:0"]->NumElements(); ++i) {
      auto tmp = node_nb_set.find(results_map["node_id:0"]->Raw<int64_t>()[i]);
      ASSERT_TRUE(tmp != node_nb_set.end());
      node_ids.push_back(results_map["node_id:0"]->Raw<int64_t>()[i]);
    }

    int32_t total_nb_num = 0;
    ASSERT_EQ(node_ids.size() * 2, results_map["nb:0"]->NumElements());
    for (int32_t i = 0; i < results_map["nb:0"]->NumElements(); i += 2) {
      auto tmp = results_map["nb:0"]->Raw<int32_t>()[i + 1] -
                 results_map["nb:0"]->Raw<int32_t>()[i];
      ASSERT_EQ(tmp, node_nb_set[node_ids[i / 2]].size());
      total_nb_num += node_nb_set[node_ids[i / 2]].size();
    }

    std::vector<int64_t> nb_ids;
    ASSERT_EQ(total_nb_num, results_map["nb:1"]->NumElements());
    for (int32_t i = 0; i < results_map["nb:0"]->NumElements(); i += 2) {
      int32_t begin = results_map["nb:0"]->Raw<int32_t>()[i];
      int32_t end = results_map["nb:0"]->Raw<int32_t>()[i + 1];
      int64_t root = node_ids[i / 2];
      for (int32_t j = begin; j < end; ++j) {
        ASSERT_EQ(node_nb_set[root][j - begin],
                  results_map["nb:1"]->Raw<int64_t>()[j]);
        nb_ids.push_back(results_map["nb:1"]->Raw<int64_t>()[j]);
      }
    }

    int32_t total_nf_num = 0;
    ASSERT_EQ(nb_ids.size() * 2, results_map["nb_feature:0"]->NumElements());
    auto size = results_map["nb_feature:0"]->NumElements();
    for (int32_t i = 0; i < size; i += 2) {
      auto tmp = results_map["nb_feature:0"]->Raw<int32_t>()[i + 1] -
                 results_map["nb_feature:0"]->Raw<int32_t>()[i];
      ASSERT_EQ(tmp, node_f_set[nb_ids[i / 2]].size());
      total_nf_num += node_f_set[nb_ids[i / 2]].size();
    }

    ASSERT_EQ(total_nf_num, results_map["nb_feature:1"]->NumElements());
    size = results_map["nb_feature:0"]->NumElements();
    for (int32_t i = 0; i < size; i += 2) {
      int32_t begin = results_map["nb_feature:0"]->Raw<int32_t>()[i];
      int32_t end = results_map["nb_feature:0"]->Raw<int32_t>()[i + 1];
      int64_t root = nb_ids[i / 2];
      for (int32_t j = begin; j < end; ++j) {
        ASSERT_EQ(node_f_set[root][j - begin],
                  results_map["nb_feature:1"]->Raw<int64_t>()[j]);
      }
    }

    int32_t total_f_num = 0;
    ASSERT_EQ(node_ids.size() * 2, results_map["n_feature:0"]->NumElements());
    for (int32_t i = 0; i < results_map["n_feature:0"]->NumElements(); i += 2) {
      auto tmp = results_map["n_feature:0"]->Raw<int32_t>()[i + 1] -
                 results_map["n_feature:0"]->Raw<int32_t>()[i];
      ASSERT_EQ(tmp, node_f_set[node_ids[i / 2]].size());
      total_f_num += node_f_set[node_ids[i / 2]].size();
    }

    ASSERT_EQ(total_f_num, results_map["n_feature:1"]->NumElements());
    for (int32_t i = 0; i < results_map["n_feature:0"]->NumElements(); i += 2) {
      int32_t begin = results_map["n_feature:0"]->Raw<int32_t>()[i];
      int32_t end = results_map["n_feature:0"]->Raw<int32_t>()[i + 1];
      int64_t root = node_ids[i / 2];
      for (int32_t j = begin; j < end; ++j) {
        ASSERT_EQ(node_f_set[root][j - begin],
                  results_map["n_feature:1"]->Raw<int64_t>()[j]);
      }
    }
  }

  {
    std::string gremlin = R"(v(nodes).sampleNB(edge_types, n, 0).as(n1).sampleNB(edge_types, n, 0).as(n2))";
    Query query(gremlin);

    TensorShape nodes_shape({4});
    TensorShape edge_types_shape({2});
    TensorShape scalar_shape({1});
    Tensor* nodes_t = query.AllocInput("nodes", nodes_shape, kUInt64);
    Tensor* edge_types_t = query.AllocInput("edge_types",
                                            edge_types_shape, kInt32);
    Tensor* n_t = query.AllocInput("n", scalar_shape, kInt32);
    std::vector<uint64_t> nodes = {1, 2, 0, 3};
    std::vector<int32_t> edge_types = {0, 1};
    GetEdgeType("0", &edge_types[0]);
    GetEdgeType("1", &edge_types[1]);
    int32_t n = 3;
    std::copy(nodes.begin(), nodes.end(), nodes_t->Raw<uint64_t>());
    std::copy(edge_types.begin(), edge_types.end(),
              edge_types_t->Raw<int32_t>());
    n_t->Raw<int32_t>()[0] = n;
    std::vector<std::string> result_names = {"n1:0", "n1:1", "n2:0", "n2:1"};
    std::unordered_map<std::string, Tensor*> results_map =
        proxy->RunGremlin(&query, result_names);
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> nb_sets(6);
    nb_sets[1] = {2, 3, 4}; nb_sets[2] = {3, 5}; nb_sets[3] = {4};
    nb_sets[4] = {5}; nb_sets[5] = {2, 6};
    nb_sets[6] = {1, 3, 5}; nb_sets[0] = {euler::common::DEFAULT_UINT64};
    for (size_t i = 0; i < nodes.size(); ++i) {
      int32_t begin = results_map["n1:0"]->Raw<int32_t>()[i * 2];
      int32_t end = results_map["n1:0"]->Raw<int32_t>()[i * 2 + 1];
      uint64_t root_id = nodes[i];
      for (int32_t j = begin; j < end; ++j) {
        auto t = nb_sets[root_id].find(results_map["n1:1"]->Raw<uint64_t>()[j]);
        ASSERT_TRUE(t != nb_sets[root_id].end());
      }
    }
    for (int32_t i = 0; i < results_map["n1:1"]->NumElements(); ++i) {
      int32_t begin = results_map["n2:0"]->Raw<int32_t>()[i * 2];
      int32_t end = results_map["n2:0"]->Raw<int32_t>()[i * 2 + 1];
      uint64_t root_id = results_map["n1:1"]->Raw<uint64_t>()[i];
      for (int32_t j = begin; j < end; ++j) {
        auto t = nb_sets[root_id].find(results_map["n2:1"]->Raw<uint64_t>()[j]);
        ASSERT_TRUE(t != nb_sets[root_id].end());
      }
    }
  }

  {
    std::string gremlin = "v(nodes).sampleLNB(edge_types, n, m, 0).as(layer)";
    Query query(gremlin);

    TensorShape nodes_shape({3});
    TensorShape edge_types_shape({2});
    TensorShape scalar_shape({1});
    Tensor* nodes_t = query.AllocInput("nodes", nodes_shape, kUInt64);
    Tensor* edge_types_t = query.AllocInput("edge_types",
                                            edge_types_shape, kInt32);
    Tensor* n_t = query.AllocInput("n", scalar_shape, kInt32);
    Tensor* m_t = query.AllocInput("m", scalar_shape, kInt32);
    std::vector<uint64_t> nodes = {1, 2, 3};
    std::vector<int32_t> edge_types = {0, 1};
    GetEdgeType("0", &edge_types[0]);
    GetEdgeType("1", &edge_types[1]);
    int32_t n = 3, m = 10;
    std::copy(nodes.begin(), nodes.end(), nodes_t->Raw<uint64_t>());
    std::copy(edge_types.begin(), edge_types.end(),
              edge_types_t->Raw<int32_t>());
    n_t->Raw<int32_t>()[0] = n;
    m_t->Raw<int32_t>()[0] = m;

    std::vector<std::string> result_names = {"layer:0", "layer:1", "layer:2"};
    std::unordered_map<std::string, Tensor*> results_map =
        proxy->RunGremlin(&query, result_names);

    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> node_nb_set;
    std::unordered_map<uint64_t, float> node_w_set;
    std::unordered_map<uint64_t, int32_t> node_t_set;
    node_nb_set[1].insert(2); node_nb_set[1].insert(3);
    node_nb_set[1].insert(4);
    node_nb_set[2].insert(3); node_nb_set[2].insert(5);
    node_nb_set[3].insert(4);
    node_nb_set[4].insert(5);
    node_nb_set[5].insert(2); node_nb_set[5].insert(6);
    node_nb_set[6].insert(1); node_nb_set[6].insert(3);
    node_nb_set[6].insert(5);
    ASSERT_EQ(nodes.size() * 2, results_map["layer:0"]->NumElements());
    for (size_t i = 0; i < nodes.size(); ++i) {
      uint64_t root = nodes[i];
      int32_t begin = results_map["layer:0"]->Raw<int32_t>()[i * 2];
      int32_t end = results_map["layer:0"]->Raw<int32_t>()[i * 2 + 1];
      for (int32_t j = begin; j < end; ++j) {
        uint64_t nb = results_map["layer:1"]->Raw<uint64_t>()[j];
        ASSERT_TRUE(node_nb_set[root].find(nb) != node_nb_set[root].end());
      }
    }
    ASSERT_EQ(m, results_map["layer:2"]->NumElements());
  }

  {
    Query query("API_SPARSE_GET_ADJ", "get_adj", 2,
                {"root_batch", "l_nb"}, {"edge_types", "m"});
    std::vector<uint64_t> root_batch = {1, 0, 3, 0, 5, 0,
                                        2, 1, 4, 1, 6, 1};
    std::vector<uint64_t> l_nb = {1, 2, 3, 4, 5, 6,
                                  1, 2, 3, 4, 5, 6};
    std::vector<int32_t> edges = {0, 1};
    GetEdgeType("0", &edges[0]);
    GetEdgeType("1", &edges[1]);
    TensorShape root_batch_shape({6, 2});
    TensorShape l_nb_shape({12, 1});
    TensorShape edge_types_shape({edges.size()});
    TensorShape scalar_shape({1});
    Tensor* root_batch_t = query.AllocInput("root_batch", root_batch_shape,
                                            DataType::kUInt64);
    Tensor* l_nb_t = query.AllocInput("l_nb", l_nb_shape, DataType::kUInt64);
    Tensor* edge_types_t = query.AllocInput("edge_types", edge_types_shape,
                                            DataType::kInt32);
    Tensor* m_t = query.AllocInput("m", scalar_shape, DataType::kInt32);
    std::copy(root_batch.begin(), root_batch.end(),
              root_batch_t->Raw<uint64_t>());
    std::copy(l_nb.begin(), l_nb.end(), l_nb_t->Raw<uint64_t>());
    std::copy(edges.begin(), edges.end(), edge_types_t->Raw<int32_t>());
    m_t->Raw<int32_t>()[0] = 6;

    std::vector<std::string> result_names = {"get_adj:0", "get_adj:1"};
    std::unordered_map<std::string, Tensor*> results_map =
        proxy->RunGremlin(&query, result_names);
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> nb_sets(6);
    nb_sets[1] = {2, 3, 4}; nb_sets[2] = {3, 5}; nb_sets[3] = {4};
    nb_sets[4] = {5}; nb_sets[5] = {2, 6};
    nb_sets[6] = {1, 3, 5}; nb_sets[0] = {0};

    for (int32_t i = 0; i < 6; ++i) {
      uint64_t root_id = root_batch[i * 2];
      int32_t begin = results_map["get_adj:0"]->Raw<int32_t>()[i * 2];
      int32_t end = results_map["get_adj:0"]->Raw<int32_t>()[i * 2 + 1];
      for (int32_t j = begin; j < end; ++j) {
        auto t = results_map["get_adj:1"]->Raw<uint64_t>()[j];
        ASSERT_TRUE(nb_sets[root_id].find(t) !=
                    nb_sets[root_id].end());
      }
    }
  }

  {
    Query query("API_SAMPLE_GRAPH_LABEL", "sample_graph", 1,
                {"count"}, {});
    Tensor* count_t = query.AllocInput("count", {1}, DataType::kInt32);
    count_t->Raw<int32_t>()[0] = 20;
    std::vector<std::string> result_names = {"sample_graph:0"};
    std::unordered_map<std::string, Tensor*> results_map =
        proxy->RunGremlin(&query, result_names);

    std::string result_s(results_map["sample_graph:0"]->Raw<char>(),
                         results_map["sample_graph:0"]->NumElements());
    EULER_LOG(INFO) << result_s;
  }

  {
    Query query("API_GET_GRAPH_BY_LABEL", "graphs", 2, {"labels"}, {});
    Tensor* labels = query.AllocInput("labels", {3}, DataType::kString);
    *(labels->Raw<std::string*>()[0]) = "1";
    *(labels->Raw<std::string*>()[1]) = "2";
    *(labels->Raw<std::string*>()[2]) = "3";

    std::vector<std::string> result_names = {"graphs:0", "graphs:1"};
    std::unordered_map<std::string, Tensor*> results_map =
        proxy->RunGremlin(&query, result_names);
    std::vector<int32_t> expect_idx = {0, 1, 1, 2, 2, 3};
    std::vector<uint64_t> expect_data = {1, 2, 3};
    for (int32_t i = 0; i < results_map["graphs:0"]->NumElements(); ++i) {
      ASSERT_EQ(expect_idx[i], results_map["graphs:0"]->Raw<int32_t>()[i]);
    }
    for (int32_t i = 0; i < results_map["graphs:1"]->NumElements(); ++i) {
      ASSERT_EQ(expect_data[i], results_map["graphs:1"]->Raw<uint64_t>()[i]);
    }
  }
}

}  // namespace euler
