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

#include <sys/prctl.h>
#include <signal.h>
#include <unistd.h>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/generic/generic_stub.h"

#include "euler/service/grpc_server.h"
#include "euler/common/logging.h"
#include "euler/common/server_register.h"
#include "euler/common/data_types.h"
#include "euler/client/query_proxy.h"
#include "euler/client/query.h"
#include "euler/parser/optimizer.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/types.pb.h"
#include "euler/core/api/api.h"
#include "euler/core/kernels/common.h"

namespace euler {
static int32_t do_abort = 0;
void handle_signal(int32_t signo) {
  if (signo == SIGHUP) {
    do_abort = 1;
  }
}


class End2EndTest: public ::testing::Test {
 protected:
  static const char zk_path_[];
  std::shared_ptr<ServerRegister> register_;
  pid_t pid_;

  void SetUp() override {
    EulerGraph();
    pid_ = fork();
    if (pid_ > 0) {
      // Create a grpc server and start it
      ServerDef server_def = {"grpc", 0, 2, {}};
      server_def.options.insert({"port", "9190"});
      server_def.options.insert({"data_path", "/tmp/euler"});
      server_def.options.insert({"load_data_type", "all"});
      server_def.options.insert({"global_sampler_type", "all"});
      server_def.options.insert({"zk_server", "127.0.0.1:2181"});
      server_def.options.insert({"zk_path", zk_path_});
      auto s = NewServer(server_def, &server_);
      ASSERT_TRUE(s.ok()) << s.DebugString();
      s = server_->Start();
      ASSERT_TRUE(s.ok()) << s;
    } else if (pid_ == 0) {
      ServerDef server_def = {"grpc", 1, 2, {}};
      server_def.options.insert({"port", "9191"});
      server_def.options.insert({"data_path", "/tmp/euler"});
      server_def.options.insert({"load_data_type", "all"});
      server_def.options.insert({"global_sampler_type", "all"});
      server_def.options.insert({"zk_server", "127.0.0.1:2181"});
      server_def.options.insert({"zk_path", zk_path_});
      auto s = NewServer(server_def, &server_);
      ASSERT_TRUE(s.ok()) << s.DebugString();
      s = server_->Start();
      ASSERT_TRUE(s.ok()) << s;
    }
  }

  void TearDown() override {
    auto s = server_->Stop();
    ASSERT_TRUE(s.ok()) << s;
  }

  std::unique_ptr<ServerInterface> server_;
};

const char End2EndTest::zk_path_[] = "/euler-2.0-test1";

TEST_F(End2EndTest, Execute) {
  if (pid_ > 0) {
    GraphConfig graph_config;
    graph_config.Add("zk_server", "127.0.0.1:2181");
    graph_config.Add("zk_path", End2EndTest::zk_path_);
    graph_config.Add("num_retries", 1);
    graph_config.Add("shard_num", 2);
    graph_config.Add("mode", "remote");

    QueryProxy::Init(graph_config);

    QueryProxy* proxy = QueryProxy::GetInstance();

    {
      std::string gremlin = "sampleE(edge_type, count).as(eid)";
      Query* query = new Query(gremlin);

      auto t_edge_type = query->AllocInput("edge_type", {1}, euler::kInt32);
      auto t_count = query->AllocInput("count", {1}, euler::kInt32);
      *(t_edge_type->Raw<int32_t>()) = 1;
      *(t_count->Raw<int32_t>()) = 10;

      Signal s;
      auto callback = [&s, query] () {
        auto res = query->GetResult("eid:0");
        ASSERT_EQ(res->NumElements(), 10 * 3);
        for (int32_t i = 0; i < res->NumElements(); ++i) {
          if (i % 3 == 2) {
            ASSERT_EQ(res->Raw<uint64_t>()[i], 1);
          }
        }
        delete query;
        s.Notify();
      };
      proxy->RunAsyncGremlin(query, callback);
      s.Wait();
    }

    {
      std::string gremlin =R"(sampleN(node_type, count).as(node).sampleNB(edge_types, count, 0).as(nb))";
      Query query(gremlin);

      TensorShape shape({1});
      TensorShape edge_types_shape({2});
      Tensor* node_type_t = query.AllocInput("node_type", shape, kInt32);
      Tensor* count_t = query.AllocInput("count", shape, kInt32);
      Tensor* edge_types_t = query.AllocInput("edge_types",
                                              edge_types_shape, kInt32);
      int32_t node_type = 0;
      GetNodeType("0", &node_type);
      *(node_type_t->Raw<int32_t>()) = node_type;
      *(count_t->Raw<int32_t>()) = 3;
      std::vector<int32_t> edge_types = {0, 0};
      GetEdgeType("0", &edge_types[0]);
      GetEdgeType("1", &edge_types[1]);
      edge_types_t->Raw<int32_t>()[0] = edge_types[0];
      edge_types_t->Raw<int32_t>()[1] = edge_types[1];

      std::vector<std::string> result_names = {"node:0", "nb:0",
                                               "nb:1", "nb:2"};
      std::unordered_map<std::string, Tensor*> results_map =
          proxy->RunGremlin(&query, result_names);
      std::unordered_set<uint64_t> nodes = {2, 4, 6};
      std::unordered_map<uint64_t, std::unordered_set<uint64_t>> node_nb_set;
      node_nb_set[2] = {3, 5}; node_nb_set[4] = {5}; node_nb_set[6] = {1, 3, 5};

      ASSERT_EQ(3, results_map["node:0"]->NumElements());
      for (int32_t i = 0; i < results_map["node:0"]->NumElements(); ++i) {
        auto tmp = nodes.find(results_map["node:0"]->Raw<uint64_t>()[i]);
        ASSERT_TRUE(tmp != nodes.end());
      }
      for (int32_t i = 0; i < 3; ++i) {
        uint64_t root_id = results_map["node:0"]->Raw<uint64_t>()[i];
        int32_t begin = results_map["nb:0"]->Raw<int32_t>()[i * 2];
        int32_t end = results_map["nb:0"]->Raw<int32_t>()[i * 2 + 1];
        for (int32_t j = begin; j < end; ++j) {
          auto t = results_map["nb:1"]->Raw<uint64_t>()[j];
          ASSERT_TRUE(node_nb_set[root_id].find(t) !=
                      node_nb_set[root_id].end());
        }
      }
    }

    {
      Query* query =
        new Query("sampleN(node_type, count).as(node_id).values(fid).as(p3)");

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
      std::string gremlin = R"(v(nodes).outV(edge_types).has(price gt 2).order_by(id, asc).limit(2).as(nb))";
      Query query(gremlin);

      TensorShape shape({3});
      TensorShape shape1({2});
      Tensor* nodes_t = query.AllocInput("nodes", shape, kUInt64);
      Tensor* edge_types_t = query.AllocInput("edge_types", shape1, kInt32);
      std::vector<uint64_t> nodes = {2, 5, 6};
      std::copy(nodes.begin(), nodes.end(), nodes_t->Raw<uint64_t>());
      std::vector<int32_t> edge_types = {0, 0};
      GetEdgeType("0", &edge_types[0]);
      GetEdgeType("1", &edge_types[1]);
      edge_types_t->Raw<int32_t>()[0] = edge_types[0];
      edge_types_t->Raw<int32_t>()[1] = edge_types[1];

      std::vector<std::string> result_names = {"nb:0", "nb:1", "nb:3"};
      std::unordered_map<std::string, Tensor*> results_map =
          proxy->RunGremlin(&query, result_names);

      std::unordered_map<uint64_t, std::vector<uint64_t>> node_nb_set;
      std::unordered_map<uint64_t, std::vector<int32_t>> node_nb_t_set;
      std::unordered_map<uint64_t, std::vector<uint64_t>> node_f_set;
      node_nb_set[2] = {3, 5}; node_nb_set[5] = {2, 6}; node_nb_set[6] = {3, 5};
      node_nb_t_set[2] = {1, 1}; node_nb_t_set[5] = {0, 0};
      node_nb_t_set[6] = {1, 1};

      for (int32_t i = 0; i < results_map["nb:0"]->NumElements(); i += 2) {
        int32_t begin = results_map["nb:0"]->Raw<int32_t>()[i];
        int32_t end = results_map["nb:0"]->Raw<int32_t>()[i + 1];
        uint64_t root = nodes[i / 2];
        for (int32_t j = begin; j < end; ++j) {
          ASSERT_EQ(node_nb_set[root][j - begin],
                    results_map["nb:1"]->Raw<uint64_t>()[j]);
          ASSERT_EQ(node_nb_t_set[root][j - begin],
                    results_map["nb:3"]->Raw<int32_t>()[j]);
        }
      }
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
        ASSERT_EQ(e_f_values[i],
                  results_map["e_feature:1"]->Raw<uint64_t>()[i]);
      }
    }

    {
      std::string gremlin = "sampleNWithTypes(types, counts).as(n)";
      Query query(gremlin);
      TensorShape shape({2});
      Tensor* types_t = query.AllocInput("types", shape, kInt32);
      Tensor* counts_t = query.AllocInput("counts", shape, kInt32);
      std::vector<int32_t> types = {0, 0};
      GetNodeType("0", &types[0]);
      GetNodeType("1", &types[1]);
      std::vector<int32_t> counts = {4, 8};
      std::copy(types.begin(), types.end(), types_t->Raw<int32_t>());
      std::copy(counts.begin(), counts.end(), counts_t->Raw<int32_t>());
      std::vector<std::string> result_names = {"n:0", "n:1"};

      std::unordered_map<std::string, Tensor*> results_map =
          proxy->RunGremlin(&query, result_names);
      std::unordered_set<uint64_t> node0 = {2, 4, 6};
      std::unordered_set<uint64_t> node1 = {1, 3, 5};
      for (int32_t i = 0; i < 2; ++i) {
        int32_t begin = results_map["n:0"]->Raw<int32_t>()[i * 2];
        int32_t end = results_map["n:0"]->Raw<int32_t>()[i * 2 + 1];
        for (int32_t j = begin; j < end; ++j) {
          if (i == 0) {
            ASSERT_TRUE(node0.find(results_map["n:1"]->Raw<uint64_t>()[j]) !=
                        node0.end());
          } else {
            ASSERT_TRUE(node1.find(results_map["n:1"]->Raw<uint64_t>()[j]) !=
                        node1.end());
          }
        }
      }
    }

    {
      std::string gremlin = R"(sampleN(node_type, n_count).as(node_id).outV(edge_types).order_by(id, asc).limit(2).as(nb).values(fid).as(nb_feature).v_select(node_id).values(fid).as(n_feature))";
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
                                               "nb:3", "nb_feature:0",
                                               "nb_feature:1", "n_feature:0",
                                               "n_feature:1"};
      std::unordered_map<std::string, Tensor*> results_map =
          proxy->RunGremlin(&query, result_names);

      std::unordered_map<uint64_t, std::vector<uint64_t>> node_nb_set;
      std::unordered_map<uint64_t, std::vector<int32_t>> node_nb_t_set;
      std::unordered_map<uint64_t, std::vector<uint64_t>> node_f_set;
      node_nb_set[2] = {3, 5}; node_nb_set[4] = {5}; node_nb_set[6] = {1, 3};
      node_nb_t_set[2] = {1, 1}; node_nb_t_set[4] = {1};
      node_nb_t_set[6] = {1, 1};
      node_f_set[1] = {11, 12}; node_f_set[2] = {21, 22};
      node_f_set[3] = {31, 32};
      node_f_set[4] = {41, 42}; node_f_set[5] = {51, 52};
      node_f_set[6] = {61, 62};

      std::vector<uint64_t> node_ids;
      ASSERT_EQ(10, results_map["node_id:0"]->NumElements());
      for (int32_t i = 0; i < results_map["node_id:0"]->NumElements(); ++i) {
        auto t = node_nb_set.find(results_map["node_id:0"]->Raw<uint64_t>()[i]);
        ASSERT_TRUE(t != node_nb_set.end());
        node_ids.push_back(results_map["node_id:0"]->Raw<uint64_t>()[i]);
      }

      int32_t total_nb_num = 0;
      ASSERT_EQ(node_ids.size() * 2, results_map["nb:0"]->NumElements());
      for (int32_t i = 0; i < results_map["nb:0"]->NumElements(); i += 2) {
        auto tmp = results_map["nb:0"]->Raw<int32_t>()[i + 1] -
                   results_map["nb:0"]->Raw<int32_t>()[i];
        ASSERT_EQ(tmp, node_nb_set[node_ids[i / 2]].size());
        total_nb_num += node_nb_set[node_ids[i / 2]].size();
      }

      std::vector<uint64_t> nb_ids;
      ASSERT_EQ(total_nb_num, results_map["nb:1"]->NumElements());
      for (int32_t i = 0; i < results_map["nb:0"]->NumElements(); i += 2) {
        int32_t begin = results_map["nb:0"]->Raw<int32_t>()[i];
        int32_t end = results_map["nb:0"]->Raw<int32_t>()[i + 1];
        uint64_t root = node_ids[i / 2];
        for (int32_t j = begin; j < end; ++j) {
          ASSERT_EQ(node_nb_set[root][j - begin],
                    results_map["nb:1"]->Raw<uint64_t>()[j]);
          ASSERT_EQ(node_nb_t_set[root][j - begin],
                    results_map["nb:3"]->Raw<int32_t>()[j]);
          nb_ids.push_back(results_map["nb:1"]->Raw<uint64_t>()[j]);
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
        uint64_t root = nb_ids[i / 2];
        for (int32_t j = begin; j < end; ++j) {
          ASSERT_EQ(node_f_set[root][j - begin],
                    results_map["nb_feature:1"]->Raw<uint64_t>()[j]);
        }
      }

      int32_t total_f_num = 0;
      ASSERT_EQ(node_ids.size() * 2, results_map["n_feature:0"]->NumElements());
      size = results_map["n_feature:0"]->NumElements();
      for (int32_t i = 0; i < size; i += 2) {
        auto tmp = results_map["n_feature:0"]->Raw<int32_t>()[i + 1] -
                   results_map["n_feature:0"]->Raw<int32_t>()[i];
        ASSERT_EQ(tmp, node_f_set[node_ids[i / 2]].size());
        total_f_num += node_f_set[node_ids[i / 2]].size();
      }

      ASSERT_EQ(total_f_num, results_map["n_feature:1"]->NumElements());
      size = results_map["n_feature:0"]->NumElements();
      for (int32_t i = 0; i < size; i += 2) {
        int32_t begin = results_map["n_feature:0"]->Raw<int32_t>()[i];
        int32_t end = results_map["n_feature:0"]->Raw<int32_t>()[i + 1];
        uint64_t root = node_ids[i / 2];
        for (int32_t j = begin; j < end; ++j) {
          ASSERT_EQ(node_f_set[root][j - begin],
                    results_map["n_feature:1"]->Raw<uint64_t>()[j]);
        }
      }
    }

    {
      std::string gremlin = R"(v(nodes).sampleNB(edge_types, n, 0).as(n1).sampleNB(edge_types, n, 0).as(n2).v_select(n1).values(fid).as(n1_f))";
      Query query(gremlin);

      TensorShape nodes_shape({4});
      TensorShape edge_types_shape({2});
      TensorShape scalar_shape({1});
      TensorShape fid_shape({1});
      Tensor* nodes_t = query.AllocInput("nodes", nodes_shape, kUInt64);
      Tensor* edge_types_t = query.AllocInput("edge_types",
                                              edge_types_shape, kInt32);
      Tensor* n_t = query.AllocInput("n", scalar_shape, kInt32);
      Tensor* fid_t = query.AllocInput("fid", fid_shape, kString);
      std::vector<uint64_t> nodes = {1, 0, 2, 3};
      std::vector<int32_t> edge_types = {0, 1};
      GetEdgeType("0", &edge_types[0]);
      GetEdgeType("1", &edge_types[1]);
      int32_t n = 3;
      std::copy(nodes.begin(), nodes.end(), nodes_t->Raw<uint64_t>());
      std::copy(edge_types.begin(), edge_types.end(),
                edge_types_t->Raw<int32_t>());
      n_t->Raw<int32_t>()[0] = n;
      std::string fid = "sparse_f1";
      *(fid_t->Raw<std::string*>()[0]) = fid;

      std::vector<std::string> result_names = {"n1:0", "n1:1", "n2:0",
                                               "n2:1", "n1_f:1"};
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
          auto t = results_map["n1:1"]->Raw<uint64_t>()[j];
          ASSERT_TRUE(nb_sets[root_id].find(t) !=
                      nb_sets[root_id].end());
        }
      }
      for (int32_t i = 0; i < results_map["n1:1"]->NumElements(); ++i) {
        int32_t begin = results_map["n2:0"]->Raw<int32_t>()[i * 2];
        int32_t end = results_map["n2:0"]->Raw<int32_t>()[i * 2 + 1];
        uint64_t root_id = results_map["n1:1"]->Raw<uint64_t>()[i];
        for (int32_t j = begin; j < end; ++j) {
          auto t = results_map["n2:1"]->Raw<uint64_t>()[j];
          ASSERT_TRUE(nb_sets[root_id].find(t) !=
                      nb_sets[root_id].end());
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
      std::string gremlin =
        "v(nodes).sampleLNB(edge_types, n, m, sqrt, 0).as(layer)";
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
      std::string gremlin =
        "v(nodes).sampleNB(edge_types, n, 0).limit(5).as(nb)";
      Query query(gremlin);

      TensorShape nodes_shape({3});
      TensorShape edge_types_shape({2});
      TensorShape scalar_shape({1});
      Tensor* nodes_t = query.AllocInput("nodes", nodes_shape, kUInt64);
      Tensor* edge_types_t = query.AllocInput("edge_types",
                                              edge_types_shape, kInt32);
      Tensor* n_t = query.AllocInput("n", scalar_shape, kInt32);

      std::vector<uint64_t> nodes = {1, 2, 3};
      std::vector<int32_t> edge_types = {0, 1};
      GetEdgeType("0", &edge_types[0]);
      GetEdgeType("1", &edge_types[1]);
      int32_t n = 10;
      std::copy(nodes.begin(), nodes.end(), nodes_t->Raw<uint64_t>());
      std::copy(edge_types.begin(), edge_types.end(),
                edge_types_t->Raw<int32_t>());
      n_t->Raw<int32_t>()[0] = n;

      std::vector<std::string> result_names = {"nb:0", "nb:1"};
      std::unordered_map<std::string, Tensor*> results_map =
          proxy->RunGremlin(&query, result_names);

      std::unordered_map<uint64_t, std::unordered_set<uint64_t>> nb_set;
      nb_set[1] = {2, 3, 4}; nb_set[2] = {3, 5}, nb_set[3] = {4};

      for (size_t i = 0; i < 3; ++i) {
        int32_t begin = results_map["nb:0"]->Raw<int32_t>()[i * 2];
        int32_t end = results_map["nb:0"]->Raw<int32_t>()[i * 2 + 1];
        uint64_t root_id = nodes[i];
        ASSERT_EQ(5, end - begin);
        for (int32_t j = begin; j < end; ++j) {
          auto t = results_map["nb:1"]->Raw<uint64_t>()[j];
          ASSERT_TRUE(nb_set[root_id].find(t) !=
                      nb_set[root_id].end());
        }
      }
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
      Query query("API_GET_GRAPH_BY_LABEL", "graphs", 2, {"labels"}, {});
      Tensor* labels = query.AllocInput("labels", {6}, DataType::kString);
      *(labels->Raw<std::string*>()[0]) = "1";
      *(labels->Raw<std::string*>()[1]) = "2";
      *(labels->Raw<std::string*>()[2]) = "3";
      *(labels->Raw<std::string*>()[3]) = "4";
      *(labels->Raw<std::string*>()[4]) = "5";
      *(labels->Raw<std::string*>()[5]) = "6";

      std::vector<std::string> result_names = {"graphs:0", "graphs:1"};
      std::unordered_map<std::string, Tensor*> results_map =
          proxy->RunGremlin(&query, result_names);
      std::vector<int32_t> expect_idx = {0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6};
      std::vector<uint64_t> expect_data = {1, 2, 3, 4, 5, 6};
      for (int32_t i = 0; i < results_map["graphs:0"]->NumElements(); ++i) {
        ASSERT_EQ(expect_idx[i], results_map["graphs:0"]->Raw<int32_t>()[i]);
      }
      for (int32_t i = 0; i < results_map["graphs:1"]->NumElements(); ++i) {
        ASSERT_EQ(expect_data[i], results_map["graphs:1"]->Raw<uint64_t>()[i]);
      }
    }
  } else {
    signal(SIGHUP, handle_signal);
    prctl(PR_SET_PDEATHSIG, SIGHUP);
    while (!do_abort) {
      sleep(10);
    }
  }
}

}  // namespace euler
