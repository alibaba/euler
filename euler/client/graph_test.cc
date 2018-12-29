/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

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

#include "euler/client/graph.h"

#include <algorithm>
#include <iostream>

#include "gtest/gtest.h"

#include "euler/common/server_monitor.h"
#include "euler/client/rpc_client.h"
#include "euler/client/impl_register.h"
#include "euler/client/remote_graph.h"
#include "euler/proto/graph_service.pb.h"

namespace euler {
namespace client {


///////////////////////////// Local Graph Test /////////////////////////////

GraphConfig LocalGraphConfig() {
  GraphConfig config;
  config.Add("mode", "Local");
  config.Add("directory", "testdata/localgraph");
  config.Add("load_type", "compact");
  return config;
}

std::unique_ptr<Graph> NewLocalGraph() {
  auto config = LocalGraphConfig();
  auto graph = Graph::NewGraph(config);
  return graph;
}

TEST(LocalGraphTest, TestNewGraph) {
  GraphConfig config;
  auto graph = Graph::NewGraph(config);
  ASSERT_EQ(nullptr, graph);

  config.Add("mode", "Local");
  graph = Graph::NewGraph(config);
  ASSERT_EQ(nullptr, graph);

  config.Add("directory", "testdata/localgraph");
  config.Add("load_type", "compact");
  graph = Graph::NewGraph(config);
  ASSERT_NE(nullptr, graph);
}

TEST(LocalGraphTest, TestSampleNode) {
  auto graph = NewLocalGraph();
  NodeIDVec node_id_vec;
  auto callback = [&node_id_vec] (const NodeIDVec& result) {
    std::swap(node_id_vec, const_cast<NodeIDVec&>(result));
    for (auto it = node_id_vec.begin(); it != node_id_vec.end(); ++it) {
      std::cout << *it << std::endl;
    }
    ASSERT_EQ(10u, node_id_vec.size());
  };
  graph->SampleNode(1, 10, callback);
}

TEST(LocalGraphTest, TestSampleEdge) {
  auto graph = NewLocalGraph();
  EdgeIDVec edge_id_vec;
  auto callback = [&edge_id_vec] (const EdgeIDVec& result) {
    std::swap(edge_id_vec, const_cast<EdgeIDVec&>(result));
    for (auto it = edge_id_vec.begin(); it != edge_id_vec.end(); ++it) {
      std::cout << "(" <<
          std::get<0>(*it) << ", " <<
          std::get<1>(*it) << ", " <<
          std::get<2>(*it) <<
          ")" << std::endl;
    }
    ASSERT_EQ(10u, edge_id_vec.size());
  };
  graph->SampleEdge(1, 10, callback);
}

#define GET_FEATURE(FEATURE_TYPE, IDS, FIDS, METHOD)                  \
  auto features = new FEATURE_TYPE();                                 \
  auto cb = [features, IDS, FIDS] (const FEATURE_TYPE& result) {      \
    std::swap(*features, const_cast<FEATURE_TYPE&>(result));          \
    ASSERT_EQ(IDS.size(), features->size());                          \
    ASSERT_EQ(FIDS.size(), features->at(0).size());                   \
    for (auto it = features->begin(); it != features->end(); ++it) {  \
      for (auto ij = it->begin(); ij != it->end(); ++ij) {            \
        for (auto ik = ij->begin(); ik != ij->end(); ++ik) {          \
          std::cout << *ik << " ";                                    \
        }                                                             \
        std::cout << std::endl;                                       \
      }                                                               \
      std::cout << std::endl;                                         \
    }                                                                 \
  };                                                                  \
  graph->METHOD(IDS, FIDS, cb);

TEST(LocalGraphTest, TestGetNodeFloat32Feature) {
  std::vector<NodeID> node_ids = {123};
  std::vector<int> fids = {0, 1};
  auto graph = NewLocalGraph();
  GET_FEATURE(FloatFeatureVec, node_ids, fids, GetNodeFloat32Feature);
}

TEST(LocalGraphTest, TestGetNodeUint64Feature) {
  std::vector<NodeID> node_ids = {123};
  std::vector<int> fids = {0, 1};
  auto graph = NewLocalGraph();
  GET_FEATURE(UInt64FeatureVec, node_ids, fids, GetNodeUint64Feature);
}

TEST(LocalGraphTest, TestGetNodeBinaryFeature) {
  std::vector<NodeID> node_ids = {123};
  std::vector<int> fids = {0, 1};
  auto graph = NewLocalGraph();
  GET_FEATURE(BinaryFatureVec, node_ids, fids, GetNodeBinaryFeature);
}

TEST(LocalGraphTest, TestGetEdgeFloat32Feature) {
  EdgeIDVec edge_ids;
  edge_ids.push_back(EdgeID(123, 456, 0));
  std::vector<int> fids = {0, 1};
  auto graph = NewLocalGraph();
  GET_FEATURE(FloatFeatureVec, edge_ids, fids, GetEdgeFloat32Feature);
}

TEST(LocalGraphTest, TestGetEdgeUint64Feature) {
  EdgeIDVec edge_ids;
  edge_ids.push_back(EdgeID(123, 456, 0));
  std::vector<int> fids = {0, 1};
  auto graph = NewLocalGraph();
  GET_FEATURE(UInt64FeatureVec, edge_ids, fids, GetEdgeUint64Feature);
}

TEST(LocalGraphTest, TestGetEdgeBinaryFeature) {
  EdgeIDVec edge_ids;
  edge_ids.push_back(EdgeID(123, 456, 0));
  std::vector<int> fids = {0, 1};
  auto graph = NewLocalGraph();
  GET_FEATURE(BinaryFatureVec, edge_ids, fids, GetEdgeBinaryFeature);
}

#define GET_NEIGHBOR(METHOD, ...)                                       \
  auto neighbors = new IDWeightPairVec();                               \
  auto callback = [node_ids, edge_types, neighbors] (                   \
      const IDWeightPairVec& result) {                                  \
    std::swap(*neighbors, const_cast<IDWeightPairVec&>(result));        \
    ASSERT_EQ(node_ids.size(), neighbors->size());                      \
    for (auto it = neighbors->begin(); it != neighbors->end(); ++it) {  \
      for (auto ij = it->begin(); ij != it->end(); ++ij) {              \
        std::cout << "("                                                \
                  << std::get<0>(*ij) << ", "                           \
                  << std::get<1>(*ij) << ", "                           \
                  << std::get<2>(*ij) <<                                \
                  ")" << std::endl;                                     \
      }                                                                 \
      std::cout << std::endl;                                           \
    }                                                                   \
  };                                                                    \
  graph->METHOD(node_ids, edge_types, ##__VA_ARGS__, callback);         \

TEST(LocalGraphTest, TestGetFullNeighbor) {
  std::vector<NodeID> node_ids = {123};
  std::vector<int> edge_types = {1};
  auto graph = NewLocalGraph();
  GET_NEIGHBOR(GetFullNeighbor);
}

TEST(LocalGraphTest, TestGetSortedFullNeighbor) {
  std::vector<NodeID> node_ids = {123};
  std::vector<int> edge_types = {1};
  auto graph = NewLocalGraph();
  GET_NEIGHBOR(GetSortedFullNeighbor);
}

TEST(LocalGraphTest, TestGetTopKNeighbor) {
  std::vector<NodeID> node_ids = {123};
  std::vector<int> edge_types = {1};
  auto graph = NewLocalGraph();
  GET_NEIGHBOR(GetTopKNeighbor, 2);
}

TEST(LocalGraphTest, TestSampleNeighbor) {
  std::vector<NodeID> node_ids = {123};
  std::vector<int> edge_types = {1};
  auto graph = NewLocalGraph();
  GET_NEIGHBOR(SampleNeighbor, 2);
}

///////////////////////////// Remote Graph Test /////////////////////////////

class MockServerMonitor: public euler::common::ServerMonitor {
 public:
  explicit MockServerMonitor(int shard): shard_(shard) { }

  bool Initialize() override {
    return true;
  }

  bool SetShardCallback(
      size_t shard_index, const common::ShardCallback *callback) override {
    (void) shard_index;
    (void) callback;
    return false;
  }

  bool UnsetShardCallback(
      size_t shard_index, const common::ShardCallback *callback) override {
    (void) shard_index;
    (void) callback;
    return false;
  }

  bool GetNumShards(int* value) override {
    *value = shard_;
    return true;
  }

  bool  GetMeta(const std::string& key, std::string* value) override {
    if (key == "num_partitions") {
      *value = std::to_string(shard_);
      return true;
    }
    return false;
  }

  bool GetShardMeta(size_t shard_index,
                    const std::string &key, std::string* value) override {
    if (key == "node_sum_weight" || key == "edge_sum_weight") {
      if (shard_index == 0) {
        value->assign("0.2, 0.2, 0.2");
      } else if (shard_index == 1) {
        value->assign("0.3, 0.3, 0.3");
      } else if (shard_index == 2) {
        value->assign("0.5, 0.5, 0.5");
      } else {
        value->assign("0.0, 0.0, 0.0");
      }
      return true;
    }
    return false;
  }

 private:
  int shard_;
};


static const int kNodeCount = 60;

class MockRpcClient: public RpcClient {
 public:
  bool Initialize(std::shared_ptr<common::ServerMonitor> monitor,
                  size_t index, const GraphConfig &/*config*/) override {
    int shard_number = 0;
    monitor->GetNumShards(&shard_number);
    shard_number_ = shard_number;
    index_ = index;
    std::cout << "Initialize Rpc client for shard " << index_ << std::endl;
    return true;
  }

  void IssueRpcCall(const std::string& method,
                    const google::protobuf::Message& request,
                    google::protobuf::Message* response,
                    std::function<void(const Status&)> done) override {
    std::string rpc_prefix = "/euler.proto.GraphService/";
    if (method == rpc_prefix + "SampleNode") {
      SampleNode(request, response, done);
    } else if (method == rpc_prefix + "SampleEdge") {
      SampleEdge(request, response, done);
    } else if (method == rpc_prefix + "GetNodeFloat32Feature") {
      GetNodeFloat32Feature(request, response, done);
    } else if (method == rpc_prefix + "GetNodeUInt64Feature") {
      GetNodeUInt64Feature(request, response, done);
    } else if (method == rpc_prefix + "GetNodeBinaryFeature") {
      GetNodeBinaryFeature(request, response, done);
    } else if (method == rpc_prefix + "GetEdgeFloat32Feature") {
      GetEdgeFloat32Feature(request, response, done);
    } else if (method == rpc_prefix + "GetEdgeUInt64Feature") {
      GetEdgeUInt64Feature(request, response, done);
    } else if (method == rpc_prefix + "GetEdgeBinaryFeature") {
      GetEdgeBinaryFeature(request, response, done);
    } else if (method == rpc_prefix + "GetFullNeighbor") {
      GetFullNeighbor(request, response, done);
    } else if (method == rpc_prefix + "GetSortedNeighbor") {
      GetSortedNeighbor(request, response, done);
    } else if (method == rpc_prefix + "GetTopKNeighbor") {
      GetTopKNeighbor(request, response, done);
    } else if (method == rpc_prefix + "SampleNeighbor") {
      SampleNeighbor(request, response, done);
    } else {
      std::cerr << "No rpc method for " << method << std::endl;
    }
  }

 private:
  void SampleNode(const google::protobuf::Message& request,
                  google::protobuf::Message* response,
                  std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const euler::proto::SampleNodeRequest&>(
        request);
    auto res = reinterpret_cast<euler::proto::SampleNodeReply*>(response);
    int i = 0;
    int count = req.count();
    while (res->node_ids_size() < count) {
      int node_id = i % kNodeCount + 1;
      if (node_id % shard_number_ == index_) {
        res->mutable_node_ids()->Add(node_id);
      }
      ++i;
    }
    done(Status());
  }

  void SampleEdge(const google::protobuf::Message& request,
                  google::protobuf::Message* response,
                  std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const euler::proto::SampleEdgeRequest&>(
        request);
    auto res = reinterpret_cast<euler::proto::SampleEdgeReply*>(response);
    int i = 0;
    int count = req.count();
    while (res->edge_ids_size() < count) {
      int src_id = i % kNodeCount + 1;
      if (src_id % shard_number_ == index_) {
        auto eid = res->mutable_edge_ids()->Add();
        eid->set_src_node(src_id);
        eid->set_dst_node(src_id + 1);
      }
      ++i;
    }
    done(Status());
  }

  void GetNodeFloat32Feature(const google::protobuf::Message& request,
                             google::protobuf::Message* response,
                             std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const proto::GetNodeFloat32FeatureRequest&>(
        request);
    auto res = reinterpret_cast<proto::GetFloat32FeatureReply*>(response);
    for (auto it = req.node_ids().begin(); it != req.node_ids().end(); ++it) {
      for (auto ij = req.feature_ids().begin();
           ij != req.feature_ids().end(); ++ij) {
        res->mutable_value_nums()->Add(16);
        for (auto ik = 0; ik < 16; ++ik) {
          res->mutable_values()->Add(*it * 1000 + *ij + 0.5);
        }
      }
    }

    done(Status());
  }

  void GetNodeUInt64Feature(const google::protobuf::Message& request,
                            google::protobuf::Message* response,
                            std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const proto::GetNodeUInt64FeatureRequest&>(
        request);
    auto res = reinterpret_cast<proto::GetUInt64FeatureReply*>(response);
    for (auto it = req.node_ids().begin(); it != req.node_ids().end(); ++it) {
      for (auto ij = req.feature_ids().begin();
           ij != req.feature_ids().end(); ++ij) {
        res->mutable_value_nums()->Add(16);
        for (auto ik = 0; ik < 16; ++ik) {
          res->mutable_values()->Add(*it * 1000 + *ij);
        }
      }
    }

    done(Status());
  }

  void GetNodeBinaryFeature(const google::protobuf::Message& request,
                            google::protobuf::Message* response,
                            std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const proto::GetNodeBinaryFeatureRequest&>(
        request);
    auto res = reinterpret_cast<proto::GetBinaryFeatureReply*>(response);
    for (auto it = req.node_ids().begin(); it != req.node_ids().end(); ++it) {
      for (auto ij = req.feature_ids().begin();
           ij != req.feature_ids().end(); ++ij) {
        res->mutable_value_nums()->Add(16);
        for (auto ik = 0; ik < 16; ++ik) {
          res->mutable_values()->push_back(*it + 'A');
        }
      }
    }

    done(Status());
  }

  void GetEdgeFloat32Feature(const google::protobuf::Message& request,
                             google::protobuf::Message* response,
                             std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const proto::GetEdgeFloat32FeatureRequest&>(
        request);
    auto res = reinterpret_cast<proto::GetFloat32FeatureReply*>(response);
    for (auto it = req.edge_ids().begin(); it != req.edge_ids().end(); ++it) {
      for (auto ij = req.feature_ids().begin();
           ij != req.feature_ids().end(); ++ij) {
        res->mutable_value_nums()->Add(16);
        for (auto ik = 0; ik < 16; ++ik) {
          res->mutable_values()->Add(it->src_node() * 1000 + *ij + 0.75);
        }
      }
    }

    done(Status());
  }

  void GetEdgeUInt64Feature(const google::protobuf::Message& request,
                            google::protobuf::Message* response,
                            std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const proto::GetEdgeUInt64FeatureRequest&>(
        request);
    auto res = reinterpret_cast<proto::GetUInt64FeatureReply*>(response);
    for (auto it = req.edge_ids().begin(); it != req.edge_ids().end(); ++it) {
      for (auto ij = req.feature_ids().begin();
           ij != req.feature_ids().end(); ++ij) {
        res->mutable_value_nums()->Add(16);
        for (auto ik = 0; ik < 16; ++ik) {
          res->mutable_values()->Add(it->src_node() * 1000 + *ij);
        }
      }
    }

    done(Status());
  }

  void GetEdgeBinaryFeature(const google::protobuf::Message& request,
                            google::protobuf::Message* response,
                            std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const proto::GetEdgeBinaryFeatureRequest&>(
        request);
    auto res = reinterpret_cast<proto::GetBinaryFeatureReply*>(response);
    for (auto it = req.edge_ids().begin(); it != req.edge_ids().end(); ++it) {
      for (auto ij = req.feature_ids().begin();
           ij != req.feature_ids().end(); ++ij) {
        res->mutable_value_nums()->Add(16);
        for (auto ik = 0; ik < 16; ++ik) {
          res->mutable_values()->push_back(it->src_node() + 'A');
        }
      }
    }

    done(Status());
  }

  void GetFullNeighbor(const google::protobuf::Message& request,
                       google::protobuf::Message* response,
                       std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const proto::GetFullNeighborRequest&>(request);
    auto res = reinterpret_cast<proto::GetNeighborReply*>(response);
    for (auto it = req.node_ids().begin(); it != req.node_ids().end(); ++it) {
      res->mutable_neighbor_nums()->Add(6);
      for (int i = 0; i < 6; ++i) {
        res->mutable_node_ids()->Add((*it + i + 1) % kNodeCount);
        res->mutable_weights()->Add(i + 1.0);
        res->mutable_types()->Add(1);
      }
    }

    done(Status());
  }

  void GetSortedNeighbor(const google::protobuf::Message& request,
                         google::protobuf::Message* response,
                         std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const proto::GetSortedNeighborRequest&>(
        request);
    auto res = reinterpret_cast<proto::GetNeighborReply*>(response);
    for (auto it = req.node_ids().begin(); it != req.node_ids().end(); ++it) {
      res->mutable_neighbor_nums()->Add(6);
      std::vector<IDWeightPair> id_weight_vec;
      for (int i = 0; i < 6; ++i) {
        id_weight_vec.push_back(
            IDWeightPair((*it + i + 1) % kNodeCount, i + 1.0, 0));
      }
      std::sort(id_weight_vec.begin(), id_weight_vec.end(),
                [] (const IDWeightPair& a, const IDWeightPair& b) {
                  return std::get<0>(a) < std::get<1>(b);
                });
      for (auto& idw : id_weight_vec) {
        res->mutable_node_ids()->Add(std::get<0>(idw));
        res->mutable_weights()->Add(std::get<1>(idw));
        res->mutable_types()->Add(std::get<2>(idw));
      }
    }

    done(Status());
  }

  void GetTopKNeighbor(const google::protobuf::Message& request,
                       google::protobuf::Message* response,
                       std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const proto::GetTopKNeighborRequest&>(request);
    auto res = reinterpret_cast<proto::GetNeighborReply*>(response);
    for (auto it = req.node_ids().begin(); it != req.node_ids().end(); ++it) {
      res->mutable_neighbor_nums()->Add(req.k());
      for (int i = 0; i < static_cast<int>(req.k()); ++i) {
        res->mutable_node_ids()->Add((*it + i + 1) % kNodeCount);
        res->mutable_weights()->Add(i + 1.0);
        res->mutable_types()->Add(1);
      }
    }

    done(Status());
  }

  void SampleNeighbor(const google::protobuf::Message& request,
                      google::protobuf::Message* response,
                      std::function<void(const Status&)> done) {
    auto& req = reinterpret_cast<const proto::SampleNeighborRequest&>(request);
    auto res = reinterpret_cast<proto::GetNeighborReply*>(response);
    for (auto it = req.node_ids().begin(); it != req.node_ids().end(); ++it) {
      res->mutable_neighbor_nums()->Add(req.count());
      for (int i = 0; i < static_cast<int>(req.count()); ++i) {
        res->mutable_node_ids()->Add((*it + i + 1) % kNodeCount);
        res->mutable_weights()->Add(i + 1.5);
        res->mutable_types()->Add(1);
      }
    }

    done(Status());
  }

 private:
  size_t index_;
  size_t shard_number_;
};

GraphConfig RemoteGraphConfig() {
  REGISTER_IMPL(RpcClient, MockRpcClient);
  GraphConfig config;
  config.Add("mode", "Remote");
  config.Add("init", "lazy");
  return config;
}

std::unique_ptr<Graph> NewRemoteGraph() {
  auto config = RemoteGraphConfig();
  auto graph = Graph::NewGraph(config);
  std::shared_ptr<MockServerMonitor> monitor(new MockServerMonitor(3));
  reinterpret_cast<RemoteGraph*>(graph.get())->set_server_monitor(monitor);
  if (!graph->Initialize(config)) {
    return nullptr;
  }
  return graph;
}

TEST(RemoteGraphTest, TestNewGraph) {
  auto config = RemoteGraphConfig();
  auto graph = Graph::NewGraph(config);
  ASSERT_NE(nullptr, graph);
  std::shared_ptr<MockServerMonitor> monitor(new MockServerMonitor(3));
  reinterpret_cast<RemoteGraph*>(graph.get())->set_server_monitor(monitor);
  ASSERT_TRUE(graph->Initialize(config));
}

TEST(RemoteGraphTest, TestBiasedSampleNeighbor) {
  auto graph = NewRemoteGraph();
  std::vector<NodeID> node_ids = {1, 2, 3};
  std::vector<NodeID> parent_ids = {3, 2, 1};
  std::vector<int> edge_types = {1, 2};
  std::vector<int> parent_edge_types = {1, 2};
  auto callback = [] (const IDWeightPairVec& result) {
    ASSERT_EQ(3u, result.size());
    for (auto& neighbors : result) {
      ASSERT_EQ(3u, neighbors.size());
      for (auto& neighbor : neighbors) {
        std::cout << "("
                  << std::get<0>(neighbor) << ","
                  << std::get<1>(neighbor) << ","
                  << std::get<2>(neighbor) <<
                  ")" << std::endl;
      }
    }
  };
  graph->BiasedSampleNeighbor(node_ids, parent_ids, edge_types,
                              parent_edge_types, 3, 1.0, 2.0, callback);
}

TEST(RemotelGraphTest, TestSampleNode) {
  auto graph = NewRemoteGraph();
  auto node_id_vec = new NodeIDVec();
  auto callback = [node_id_vec] (const NodeIDVec& result) {
    std::swap(*node_id_vec, const_cast<NodeIDVec&>(result));
    ASSERT_EQ(10u, node_id_vec->size());
  };
  graph->SampleNode(1, 10, callback);
}

TEST(RemoteGraphTest, TestSampleEdge) {
  auto graph = NewRemoteGraph();
  auto edge_id_vec = new EdgeIDVec();
  auto callback = [edge_id_vec] (const EdgeIDVec& result) {
    std::swap(*edge_id_vec, const_cast<EdgeIDVec&>(result));
    ASSERT_EQ(10u, edge_id_vec->size());
  };
  graph->SampleEdge(1, 10, callback);
}

#define PREPARE_GET_NODE_FEATURE()              \
  std::vector<NodeID> node_ids;                 \
  for (int i = 1; i <= 20; ++i) {               \
    node_ids.push_back(i);                      \
  }                                             \
  std::vector<int> fids = {0, 1};               \
  auto graph = NewRemoteGraph()


TEST(RemoteGraphTest, TestGetNodeFloat32Feature) {
  PREPARE_GET_NODE_FEATURE();
  GET_FEATURE(FloatFeatureVec, node_ids, fids, GetNodeFloat32Feature);
}

TEST(RemoteGraphTest, TestGetNodeUint64Feature) {
  PREPARE_GET_NODE_FEATURE();
  GET_FEATURE(UInt64FeatureVec, node_ids, fids, GetNodeUint64Feature);
}

TEST(RemoteGraphTest, TestGetNodeBinaryFeature) {
  PREPARE_GET_NODE_FEATURE();
  GET_FEATURE(BinaryFatureVec, node_ids, fids, GetNodeBinaryFeature);
}

#define PREPARE_GET_EDGE_FEATURE()              \
  EdgeIDVec edge_ids;                           \
  for (int i = 1; i <= 20; ++i) {               \
    edge_ids.push_back(EdgeID{i, i + 1, 0});    \
  }                                             \
  std::vector<int> fids = {0, 1};               \
  auto graph = NewRemoteGraph()

TEST(RemoteGraphTest, TestGetEdgeFloat32Feature) {
  PREPARE_GET_EDGE_FEATURE();
  GET_FEATURE(FloatFeatureVec, edge_ids, fids, GetEdgeFloat32Feature);
}

TEST(RemoteGraphTest, TestGetEdgeUint64Feature) {
  PREPARE_GET_EDGE_FEATURE();
  GET_FEATURE(UInt64FeatureVec, edge_ids, fids, GetEdgeUint64Feature);
}

TEST(RemoteGraphTest, TestGetEdgeBinaryFeature) {
  PREPARE_GET_EDGE_FEATURE();
  GET_FEATURE(BinaryFatureVec, edge_ids, fids, GetEdgeBinaryFeature);
}

TEST(RemoteGraphTest, TestGetFullNeighbor) {
  PREPARE_GET_NODE_FEATURE();
  std::vector<int> edge_types = {1};
  GET_NEIGHBOR(GetFullNeighbor);
}

TEST(RemoteGraphTest, TestGetSortedFullNeighbor) {
  PREPARE_GET_NODE_FEATURE();
  std::vector<int> edge_types = {1};
  GET_NEIGHBOR(GetSortedFullNeighbor);
}

TEST(RemoteGraphTest, TestGetTopKNeighbor) {
  PREPARE_GET_NODE_FEATURE();
  std::vector<int> edge_types = {1};
  GET_NEIGHBOR(GetTopKNeighbor, 2);
}

TEST(RemoteGraphTest, TestSampleNeighbor) {
  PREPARE_GET_NODE_FEATURE();
  std::vector<int> edge_types = {1};
  GET_NEIGHBOR(SampleNeighbor, 4);
}

}  // namespace client
}  // namespace euler
