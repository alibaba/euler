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

#include <algorithm>
#include <iostream>

#include "gtest/gtest.h"

#include "euler/common/logging.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/graph/graph.h"
#include "euler/core/graph/graph_builder.h"
#include "euler/core/index/index_manager.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/api/api.h"
namespace euler {

namespace {

template <typename T>
void PrintVector(T* data, size_t n, int end_line) {
  for (size_t i = 0; i < n; ++i) {
    std::cout << data[i] << " ";
    if (end_line > 0 && (i + 1) % end_line == 0) {
      std::cout << "\n";
    }
  }
  std::cout << "\n";
}

}  // namespace

class OpKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string data_path = "/tmp/euler";
    auto& graph = Graph::Instance();
    ASSERT_TRUE(graph.Init(0, 1, "node", data_path, "all").ok());

    auto& index_manager = IndexManager::Instance();
    std::string index_dir = JoinPath(data_path, "Index");
    ASSERT_TRUE(index_manager.Deserialize(index_dir).ok());
  }

  void TearDown() override {
  }
};

TEST_F(OpKernelTest, GetNode) {
  OpKernelContext ctx;

  OpKernel* op = nullptr;
  ASSERT_TRUE(CreateOpKernel("API_GET_NODE", &op).ok());
  ASSERT_NE(nullptr, op);

  DAGNodeProto proto;
  proto.set_name("get_node");
  proto.set_op("API_GET_NODE");

  std::vector<uint64_t> node_ids({1, 3, 5});

  Tensor* node_ids_t = nullptr;
  ASSERT_TRUE(ctx.Allocate(
      "node_ids", TensorShape({3}), kUInt64, &node_ids_t).ok());
  std::copy(node_ids.begin(), node_ids.end(), node_ids_t->Raw<uint64_t>());

  proto.mutable_inputs()->Add()->assign("node_ids");

  // Without DNF
  {
    EULER_LOG(INFO) << proto.DebugString();

    op->Compute(proto, &ctx);

    Tensor* ids_t = nullptr;
    ASSERT_TRUE(ctx.tensor(OutputName(proto.name(), 0), &ids_t).ok());
    ASSERT_NE(nullptr, ids_t);
    ASSERT_EQ(kUInt64, ids_t->Type());
    ASSERT_EQ(1, ids_t->Shape().Size());
    ASSERT_EQ(3, ids_t->Shape().Dims()[0]);
    ASSERT_EQ(1, ids_t->Raw<uint64_t>()[0]);
    ASSERT_EQ(3, ids_t->Raw<uint64_t>()[1]);
    ASSERT_EQ(5, ids_t->Raw<uint64_t>()[2]);

    EULER_LOG(INFO) << "Result shape {" << ids_t->Shape().Dims()[0] << "}";

    PrintVector(ids_t->Raw<uint64_t>(), ids_t->Shape().Dims()[0], -1);
  }

  // With DNF
  {
    proto.mutable_dnf()->Add()->assign("price gt 2.0");
    EULER_LOG(INFO) << proto.DebugString();

    ctx.Deallocate(OutputName(proto.name(), 0));
    op->Compute(proto, &ctx);

    Tensor* ids_t = nullptr;
    ASSERT_TRUE(ctx.tensor(OutputName(proto.name(), 0), &ids_t).ok());
    ASSERT_NE(nullptr, ids_t);
    ASSERT_EQ(kUInt64, ids_t->Type());
    ASSERT_EQ(1, ids_t->Shape().Size());
    ASSERT_EQ(2, ids_t->Shape().Dims()[0]);
    ASSERT_EQ(3, ids_t->Raw<uint64_t>()[0]);
    ASSERT_EQ(5, ids_t->Raw<uint64_t>()[1]);

    EULER_LOG(INFO) << "Result shape {" << ids_t->Shape().Dims()[0] << "}";

    PrintVector(ids_t->Raw<uint64_t>(), ids_t->Shape().Dims()[0], -1);
  }
}

TEST_F(OpKernelTest, GetEdge) {
  OpKernelContext ctx;

  OpKernel* op = nullptr;
  ASSERT_TRUE(CreateOpKernel("API_GET_EDGE", &op).ok());
  ASSERT_NE(nullptr, op);

  DAGNodeProto proto;
  proto.set_name("get_edge");
  proto.set_op("API_GET_EDGE");

  std::vector<std::vector<uint64_t>> edge_ids({
      {1, 2, 0}, {2, 5, 1}, {3, 4, 0}});

  Tensor* edge_ids_t = nullptr;
  ASSERT_TRUE(ctx.Allocate(
      "edge_ids", TensorShape({3, 3}), kUInt64, &edge_ids_t).ok());
  auto data = edge_ids_t->Raw<uint64_t>();
  for (auto& eid : edge_ids) {
    data[0] = eid[0];
    data[1] = eid[1];
    data[2] = eid[2];
    data += 3;
  }

  proto.mutable_inputs()->Add()->assign("edge_ids");

  // Without DNF
  {
    EULER_LOG(INFO) << proto.DebugString();

    op->Compute(proto, &ctx);

    Tensor* ids_t = nullptr;
    ASSERT_TRUE(ctx.tensor(OutputName(proto.name(), 0), &ids_t).ok());
    ASSERT_NE(nullptr, ids_t);
    ASSERT_EQ(kUInt64, ids_t->Type());
    ASSERT_EQ(2, ids_t->Shape().Size());
    ASSERT_EQ(3, ids_t->Shape().Dims()[0]);
    ASSERT_EQ(3, ids_t->Shape().Dims()[1]);

    EULER_LOG(INFO) << "Result shape {"
                    << ids_t->Shape().Dims()[0] << ", "
                    <<ids_t->Shape().Dims()[1] << "}";

    PrintVector(ids_t->Raw<uint64_t>(), ids_t->Shape().NumElements(), 3);
  }

  // With DNF
  {
    proto.mutable_dnf()->Add()->assign("edge_value gt 20.0");
    EULER_LOG(INFO) << proto.DebugString();

    ctx.RemoveAlias(OutputName(proto.name(), 0));
    op->Compute(proto, &ctx);

    Tensor* ids_t = nullptr;
    ASSERT_TRUE(ctx.tensor(OutputName(proto.name(), 0), &ids_t).ok());
    ASSERT_NE(nullptr, ids_t);
    ASSERT_EQ(kUInt64, ids_t->Type());
    ASSERT_EQ(2, ids_t->Shape().Size());
    ASSERT_EQ(2, ids_t->Shape().Dims()[0]);
    ASSERT_EQ(3, ids_t->Shape().Dims()[1]);

    EULER_LOG(INFO) << "Result shape {"
                    << ids_t->Shape().Dims()[0] << ", "
                    <<ids_t->Shape().Dims()[1] << "}";

    PrintVector(ids_t->Raw<uint64_t>(), ids_t->Shape().NumElements(), 3);
  }
}

TEST_F(OpKernelTest, GetFeature) {
  OpKernel* op = nullptr;
  ASSERT_TRUE(CreateOpKernel("API_GET_P", &op).ok());
  ASSERT_NE(nullptr, op);

#define CHECK_FEATURE()                                       \
  for (size_t j = 0; j < features.size(); ++j) {              \
      Tensor* index_t = nullptr;                              \
      ASSERT_TRUE(ctx.tensor(OutputName(proto.name(), 2 * j), \
                             &index_t).ok());                 \
      ASSERT_NE(nullptr, index_t);                            \
      ASSERT_EQ(kInt32, index_t->Type());                     \
      ASSERT_EQ(2, index_t->Shape().Size());                  \
      ASSERT_EQ(3, index_t->Shape().Dims()[0]);               \
      ASSERT_EQ(2, index_t->Shape().Dims()[1]);               \
                                                              \
      Tensor* values_t = nullptr;                             \
      ASSERT_TRUE(                                            \
          ctx.tensor(OutputName(proto.name(), 2 * j + 1),     \
                     &values_t).ok());                        \
      ASSERT_NE(nullptr, values_t);                           \
      ASSERT_EQ(1, values_t->Shape().Size());                 \
                                                              \
      EULER_LOG(INFO) << "Values type: " << values_t->Type(); \
                                                              \
      switch (values_t->Type()) {                             \
        case kInt32:                                          \
          PrintVector(values_t->Raw<int32_t>(),               \
                      values_t->Shape().Dims()[0], 2);        \
          break;                                              \
        case kInt64:                                          \
          PrintVector(values_t->Raw<int64_t>(),               \
                      values_t->Shape().Dims()[0], 2);        \
          break;                                              \
        case kUInt32:                                         \
          PrintVector(values_t->Raw<uint32_t>(),              \
                      values_t->Shape().Dims()[0], 2);        \
          break;                                              \
        case kUInt64:                                         \
          PrintVector(values_t->Raw<uint64_t>(),              \
                      values_t->Shape().Dims()[0], 2);        \
          break;                                              \
        case kInt8:                                           \
          PrintVector(values_t->Raw<char>(),                  \
                      values_t->Shape().Dims()[0], 3);        \
          break;                                              \
        default:                                              \
          PrintVector(values_t->Raw<float>(),                 \
                      values_t->Shape().Dims()[0], 2);        \
          break;                                              \
      }                                                       \
  }

  // Get Node Feature
  {
    OpKernelContext ctx;

    DAGNodeProto proto;
    proto.set_name("get_node_feature");
    proto.set_op("API_GET_P");

    std::vector<uint64_t> node_ids({1, 3, 5});
    std::vector<std::string> features({"sparse_f1", "sparse_f2", "dense_f3",
                                      "binary_f5"});

    Tensor* node_ids_t = nullptr;
    ASSERT_TRUE(ctx.Allocate(
        "node_ids", TensorShape({3}), kUInt64, &node_ids_t).ok());
    std::copy(node_ids.begin(), node_ids.end(), node_ids_t->Raw<uint64_t>());

    proto.mutable_inputs()->Add()->assign("node_ids");

    for (auto& feature : features) {
      Tensor* feature_t = nullptr;
      ASSERT_TRUE(ctx.Allocate(
          feature, TensorShape({1}), kString, &feature_t).ok());
      ASSERT_NE(nullptr, feature_t);
      auto data = feature_t->Raw<std::string*>();
      *(data[0]) = feature;
      proto.mutable_inputs()->Add()->assign(feature);
    }

    EULER_LOG(INFO) << proto.DebugString();

    op->Compute(proto, &ctx);

    CHECK_FEATURE();
  }

  // Get Edge Feature
  {
    OpKernelContext ctx;

    DAGNodeProto proto;
    proto.set_name("get_edge_feature");
    proto.set_op("API_GET_P");

    std::vector<std::vector<uint64_t>> edge_ids({
        {1, 2, 0}, {2, 5, 1}, {3, 4, 0}});
    std::vector<std::string> features({"sparse_f1", "sparse_f2", "dense_f3",
                                      "binary_f5"});

    Tensor* edge_ids_t = nullptr;
    ASSERT_TRUE(ctx.Allocate(
        "edge_ids", TensorShape({3, 3}), kUInt64, &edge_ids_t).ok());
    auto data = edge_ids_t->Raw<uint64_t>();
    for (auto& eid : edge_ids) {
      data[0] = eid[0];
      data[1] = eid[1];
      data[2] = eid[2];
      data += 3;
    }

    proto.mutable_inputs()->Add()->assign("edge_ids");

    for (auto& feature : features) {
      Tensor* feature_t = nullptr;
      ASSERT_TRUE(ctx.Allocate(
          feature, TensorShape({1}), kString, &feature_t).ok());
      ASSERT_NE(nullptr, feature_t);
      auto data = feature_t->Raw<std::string*>();
      *(data[0]) = feature;
      proto.mutable_inputs()->Add()->assign(feature);
    }

    EULER_LOG(INFO) << proto.DebugString();

    op->Compute(proto, &ctx);

    CHECK_FEATURE();
  }

#undef CHECK_FEATUR
}

#define CHECK_NEIGHBOR(COUNT, LINE)                                     \
  {                                                                     \
    Tensor* ids_t = nullptr;                                            \
    ASSERT_TRUE(ctx.tensor(OutputName(proto, 1), &ids_t).ok());         \
    ASSERT_NE(nullptr, ids_t);                                          \
    ASSERT_EQ(kUInt64, ids_t->Type());                                  \
    ASSERT_EQ(1, ids_t->Shape().Size());                                \
    ASSERT_EQ((COUNT), ids_t->NumElements());                           \
    PrintVector(ids_t->Raw<uint64_t>(), ids_t->NumElements(), (LINE));  \
  }                                                                     \
                                                                        \
  {                                                                     \
    Tensor* weights_t = nullptr;                                        \
    ASSERT_TRUE(ctx.tensor(OutputName(proto, 2), &weights_t).ok());     \
    ASSERT_NE(nullptr, weights_t);                                      \
    ASSERT_EQ(kFloat, weights_t->Type());                               \
    ASSERT_EQ(1, weights_t->Shape().Size());                            \
    ASSERT_EQ((COUNT), weights_t->NumElements());                       \
    PrintVector(weights_t->Raw<float>(),                                \
                weights_t->NumElements(), (LINE));                      \
  }                                                                     \
                                                                        \
  {                                                                     \
    Tensor* types_t = nullptr;                                          \
    ASSERT_TRUE(ctx.tensor(OutputName(proto, 3), &types_t).ok());       \
    ASSERT_NE(nullptr, types_t);                                        \
    ASSERT_EQ(kInt32, types_t->Type());                                 \
    ASSERT_EQ(1, types_t->Shape().Size());                              \
    ASSERT_EQ((COUNT), types_t->NumElements());                         \
    PrintVector(types_t->Raw<int32_t>(),                                \
                types_t->NumElements(), (LINE));                        \
  }                                                                     \

TEST_F(OpKernelTest, SampleNeighbor) {
  OpKernelContext ctx;

  OpKernel* op = nullptr;
  ASSERT_TRUE(CreateOpKernel("API_SAMPLE_NB", &op).ok());
  ASSERT_NE(nullptr, op);

  DAGNodeProto proto;
  proto.set_name("sample_neighbor");
  proto.set_op("API_SAMPLE_NB");

  std::vector<uint64_t> node_ids({1, 3, 5});
  std::vector<int> edge_types({0, 1});
  int count = 5;

  Tensor* node_ids_t = nullptr;
  ASSERT_TRUE(ctx.Allocate(
      "node_ids", TensorShape({node_ids.size()}), kUInt64, &node_ids_t).ok());
  std::copy(node_ids.begin(), node_ids.end(), node_ids_t->Raw<uint64_t>());

  Tensor* edge_types_t = nullptr;
  ASSERT_TRUE(ctx.Allocate(
      "edge_types", TensorShape({edge_types.size()}),
      kInt32, &edge_types_t).ok());
  std::copy(edge_types.begin(), edge_types.end(), edge_types_t->Raw<int32_t>());

  Tensor* count_t = nullptr;
  ASSERT_TRUE(ctx.Allocate(
      "count", TensorShape({1}), kInt32, &count_t).ok());
  *count_t->Raw<int32_t>() = count;

  Tensor* default_node_t = nullptr;
  ASSERT_TRUE(ctx.Allocate(
      "default_node", TensorShape({1}), kInt32, &default_node_t).ok());
  *default_node_t->Raw<int32_t>() = -1;

  proto.mutable_inputs()->Add()->assign("node_ids");
  proto.mutable_inputs()->Add()->assign("edge_types");
  proto.mutable_inputs()->Add()->assign("count");
  proto.mutable_inputs()->Add()->assign("default_node");

  EULER_LOG(INFO) << proto.DebugString();

  op->Compute(proto, &ctx);

  Tensor* index_t = nullptr;
  ASSERT_TRUE(ctx.tensor(OutputName(proto, 0), &index_t).ok());
  ASSERT_EQ(kInt32, index_t->Type());
  ASSERT_EQ(2, index_t->Shape().Size());
  ASSERT_EQ(node_ids.size(), index_t->Shape().Dims()[0]);
  ASSERT_EQ(2, index_t->Shape().Dims()[1]);
  EULER_LOG(INFO) << "Index shape: {" << index_t->Shape().Dims()[0]
                  << ", " << index_t->Shape().Dims()[1] << "}";
  PrintVector(index_t->Raw<int32_t>(), index_t->NumElements(), 2);

  CHECK_NEIGHBOR(node_ids.size() * count, count);
}

TEST_F(OpKernelTest, GetNodeNeighbor) {
  OpKernelContext ctx;

  OpKernel* op = nullptr;
  ASSERT_TRUE(CreateOpKernel("API_GET_NB_NODE", &op).ok());
  ASSERT_NE(nullptr, op);

  DAGNodeProto proto;
  proto.set_name("get_neighbor");
  proto.set_op("API_GET_NB_NODE");

  std::vector<uint64_t> node_ids({1, 3, 5});
  std::vector<int> edge_types({0, 1});

  Tensor* node_ids_t = nullptr;
  ASSERT_TRUE(ctx.Allocate(
      "node_ids", TensorShape({node_ids.size()}), kUInt64, &node_ids_t).ok());
  std::copy(node_ids.begin(), node_ids.end(), node_ids_t->Raw<uint64_t>());

  Tensor* edge_types_t = nullptr;
  ASSERT_TRUE(ctx.Allocate(
      "edge_types", TensorShape({edge_types.size()}),
      kInt32, &edge_types_t).ok());
  std::copy(edge_types.begin(), edge_types.end(), edge_types_t->Raw<int32_t>());

  proto.mutable_inputs()->Add()->assign("node_ids");
  proto.mutable_inputs()->Add()->assign("edge_types");

  // Without DNF
  {
    EULER_LOG(INFO) << proto.DebugString();
    op->Compute(proto, &ctx);
    CHECK_NEIGHBOR(6, 1);
  }

  // With DNF
  {
    proto.set_name("get_neighbor_with_dnf");

    proto.mutable_dnf()->Add()->assign("price gt 4.0");
    EULER_LOG(INFO) << proto.DebugString();

    op->Compute(proto, &ctx);

    CHECK_NEIGHBOR(3, 1);
  }
}

#undef CHECK_NEIGHBOR

TEST_F(OpKernelTest, GetEdgeNeighbor) {
  OpKernelContext ctx;

  OpKernel* op = nullptr;
  ASSERT_TRUE(CreateOpKernel("API_GET_NB_EDGE", &op).ok());
  ASSERT_NE(nullptr, op);

  DAGNodeProto proto;
  proto.set_name("get_neighbor");
  proto.set_op("API_GET_NB_EDGE");

  std::vector<uint64_t> node_ids({1, 3, 5});
  std::vector<int> edge_types({0, 1});

  Tensor* node_ids_t = nullptr;
  ASSERT_TRUE(ctx.Allocate(
      "node_ids", TensorShape({node_ids.size()}), kUInt64, &node_ids_t).ok());
  std::copy(node_ids.begin(), node_ids.end(), node_ids_t->Raw<uint64_t>());

  Tensor* edge_types_t = nullptr;
  ASSERT_TRUE(ctx.Allocate(
      "edge_types", TensorShape({edge_types.size()}),
      kInt32, &edge_types_t).ok());
  std::copy(edge_types.begin(), edge_types.end(), edge_types_t->Raw<int32_t>());

  proto.mutable_inputs()->Add()->assign("node_ids");
  proto.mutable_inputs()->Add()->assign("edge_types");
  proto.mutable_dnf()->Add()->assign("edge_value gt 13");
  proto.mutable_post_process()->Add()->assign("order_by id asc");
  op->Compute(proto, &ctx);

  Tensor* eid_idx_t = nullptr;
  Tensor* eid_t = nullptr;
  ctx.tensor("get_neighbor:0", &eid_idx_t);
  ctx.tensor("get_neighbor:1", &eid_t);
  std::vector<int32_t> eids_idx = {0, 2, 2, 3, 3, 5};
  std::vector<uint64_t> eids = {1, 3, 1, 1, 4, 0, 3, 4, 0, 5, 2, 0, 5, 6, 0};
  ASSERT_EQ(eids.size(), eid_t->NumElements());
  for (int32_t i = 0; i < eid_t->NumElements(); ++i) {
    ASSERT_EQ(eids[i], eid_t->Raw<uint64_t>()[i]);
  }
  ASSERT_EQ(eids_idx.size(), eid_idx_t->NumElements());
  for (int32_t i = 0; i < eid_idx_t->NumElements(); ++i) {
    ASSERT_EQ(eids_idx[i], eid_idx_t->Raw<int32_t>()[i]);
  }
}

TEST_F(OpKernelTest, SampleNWithTypes) {
  OpKernelContext ctx;

  OpKernel* op = nullptr;
  ASSERT_TRUE(CreateOpKernel("API_SAMPLE_N_WITH_TYPES", &op).ok());
  ASSERT_NE(nullptr, op);

  DAGNodeProto proto;
  proto.set_name("sample_n");
  proto.set_op("API_SAMPLE_N_WITH_TYPES");
  proto.mutable_inputs()->Add()->assign("types");
  proto.mutable_inputs()->Add()->assign("counts");

  Tensor* types_t = nullptr;
  Tensor* counts_t = nullptr;
  ASSERT_TRUE(ctx.Allocate(
      "types", TensorShape({2}), kInt32, &types_t).ok());
  ASSERT_TRUE(ctx.Allocate(
      "counts", TensorShape({2}), kInt32, &counts_t).ok());
  std::vector<int32_t> types = {0, 1};
  GetNodeType("0", &types[0]);
  GetNodeType("1", &types[1]);
  std::vector<int32_t> counts = {4, 8};
  std::copy(types.begin(), types.end(), types_t->Raw<int32_t>());
  std::copy(counts.begin(), counts.end(), counts_t->Raw<int32_t>());

  op->Compute(proto, &ctx);

  Tensor* nid_idx_t = nullptr;
  Tensor* nid_t = nullptr;
  ctx.tensor("sample_n:0", &nid_idx_t);
  ctx.tensor("sample_n:1", &nid_t);

  std::unordered_set<uint64_t> type0_nodes = {2, 4, 6};
  std::unordered_set<uint64_t> type1_nodes = {1, 3, 5};
  for (size_t i = 0; i < types.size(); ++i) {
    int32_t begin = nid_idx_t->Raw<int32_t>()[i * 2];
    int32_t end = nid_idx_t->Raw<int32_t>()[i * 2 + 1];
    for (int32_t j = begin; j < end; ++j) {
      if (i == 0) {
        ASSERT_TRUE(type0_nodes.find(nid_t->Raw<uint64_t>()[j]) !=
                    type0_nodes.end());
      } else {
        ASSERT_TRUE(type1_nodes.find(nid_t->Raw<uint64_t>()[j]) !=
                    type1_nodes.end());
      }
    }
  }
}
}  // namespace euler
