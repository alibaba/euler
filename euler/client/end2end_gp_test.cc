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


class End2EndGPTest: public ::testing::Test {
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
      server_def.options.insert({"data_path", "/tmp/gp_euler"});
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
      server_def.options.insert({"data_path", "/tmp/gp_euler"});
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

const char End2EndGPTest::zk_path_[] = "/euler-2.0-test1";

TEST_F(End2EndGPTest, Execute) {
  if (pid_ > 0) {
    GraphConfig graph_config;
    graph_config.Add("zk_server", "127.0.0.1:2181");
    graph_config.Add("zk_path", End2EndGPTest::zk_path_);
    graph_config.Add("num_retries", 1);
    graph_config.Add("shard_num", 2);
    graph_config.Add("mode", "remote");

    QueryProxy::Init(graph_config);

    QueryProxy* proxy = QueryProxy::GetInstance();
    /*
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
    */
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

  } else {
    signal(SIGHUP, handle_signal);
    prctl(PR_SET_PDEATHSIG, SIGHUP);
    while (!do_abort) {
      sleep(10);
    }
  }
}

}  // namespace euler
