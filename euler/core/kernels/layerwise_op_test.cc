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

#include "euler/core/graph/graph.h"
#include "euler/core/graph/graph_builder.h"
#include "euler/core/index/index_manager.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/types.pb.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/common/logging.h"

namespace euler {

class LayerwiseSampleTest : public ::testing::Test {
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

TEST_F(LayerwiseSampleTest, GetEdgeSumWeightOpTest) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("API_GET_EDGE_SUM_WEIGHT,0");
  node_proto.set_op("API_GET_EDGE_SUM_WEIGHT");
  node_proto.add_inputs("roots");
  node_proto.add_inputs("edge_types");

  std::vector<uint64_t> roots = {1, 2, 3, 4, 5, 6};
  std::vector<int32_t> edges = {0, 1};
  TensorShape roots_shape({roots.size()});
  TensorShape edge_types_shape({edges.size()});
  Tensor* roots_t = nullptr;
  Tensor* edge_types_t = nullptr;
  ctx.Allocate("roots", roots_shape, DataType::kUInt64, &roots_t);
  ctx.Allocate("edge_types", edge_types_shape, DataType::kInt32, &edge_types_t);
  std::copy(roots.begin(), roots.end(), roots_t->Raw<uint64_t>());
  std::copy(edges.begin(), edges.end(), edge_types_t->Raw<int32_t>());

  // run
  OpKernel* get_edge_sum_weight = nullptr;
  CreateOpKernel("API_GET_EDGE_SUM_WEIGHT", &get_edge_sum_weight);
  get_edge_sum_weight->Compute(node_proto, &ctx);

  // check
  std::vector<float> w = {9.0, 8.0, 4.0, 5.0, 8.0, 9.0};
  Tensor* o_r_t = nullptr;
  Tensor* o_w_t = nullptr;
  ctx.tensor("API_GET_EDGE_SUM_WEIGHT,0:0", &o_r_t);
  ctx.tensor("API_GET_EDGE_SUM_WEIGHT,0:1", &o_w_t);
  ASSERT_EQ(roots.size(), o_r_t->Shape().Dims()[0]);
  ASSERT_EQ(1, o_r_t->Shape().Dims()[1]);
  ASSERT_EQ(roots.size(), o_w_t->NumElements());
  for (size_t i = 0; i < roots.size(); ++i) {
    ASSERT_EQ(roots[i], o_r_t->Raw<uint64_t>()[i]);
    ASSERT_EQ(w[i], o_w_t->Raw<float>()[i]);
  }
}

TEST_F(LayerwiseSampleTest, SampleRootOpTest) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("API_SAMPLE_ROOT,0");
  node_proto.set_op("API_SAMPLE_ROOT");
  node_proto.add_inputs("roots");
  node_proto.add_inputs("weights");
  node_proto.add_inputs("n");
  node_proto.add_inputs("m");
  node_proto.add_inputs("0");

  int32_t batch = 2;
  int32_t n = 4;
  int32_t m = 1000000;
  std::vector<uint64_t> roots = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> weights = {1, 2, 4, 8, 1, 3, 9, 9};

  size_t bn = batch * n;
  TensorShape roots_shape({bn, 1});
  TensorShape weights_shape({bn, 1});
  TensorShape scalar_shape({1});
  Tensor* roots_t = nullptr;
  Tensor* weights_t = nullptr;
  Tensor* n_t = nullptr;
  Tensor* m_t = nullptr;
  ctx.Allocate("roots", roots_shape, DataType::kUInt64, &roots_t);
  ctx.Allocate("weights", weights_shape, DataType::kFloat, &weights_t);
  ctx.Allocate("n", scalar_shape, DataType::kInt32, &n_t);
  ctx.Allocate("m", scalar_shape, DataType::kInt32, &m_t);
  std::copy(roots.data(), roots.data() + batch * n,
            roots_t->Raw<uint64_t>());
  std::copy(weights.data(), weights.data() + batch * n,
            weights_t->Raw<float>());
  n_t->Raw<int32_t>()[0] = n;
  m_t->Raw<int32_t>()[0] = m;

  OpKernel* sample_root = nullptr;
  CreateOpKernel("API_SAMPLE_ROOT", &sample_root);
  sample_root->Compute(node_proto, &ctx);

  Tensor* output = nullptr;
  ctx.tensor("API_SAMPLE_ROOT,0:0", &output);
  int32_t cnt[8] = {0};
  for (int32_t i = 0; i < output->NumElements(); ++i) {
    ++cnt[output->Raw<uint64_t>()[i]];
  }
  ASSERT_TRUE(cnt[1] * 1.0 / cnt[0] > 1.9 && cnt[1] * 1.0 / cnt[0] < 2.1);
  ASSERT_TRUE(cnt[2] * 1.0 / cnt[1] > 1.9 && cnt[2] * 1.0 / cnt[1] < 2.1);
  ASSERT_TRUE(cnt[3] * 1.0 / cnt[2] > 1.9 && cnt[3] * 1.0 / cnt[2] < 2.1);
  ASSERT_TRUE(cnt[5] * 1.0 / cnt[4] > 2.9 && cnt[5] * 1.0 / cnt[4] < 3.1);
  ASSERT_TRUE(cnt[6] * 1.0 / cnt[5] > 2.9 && cnt[6] * 1.0 / cnt[5] < 3.1);
  ASSERT_TRUE(cnt[7] * 1.0 / cnt[6] > 0.9 && cnt[7] * 1.0 / cnt[6] < 1.1);
}

TEST_F(LayerwiseSampleTest, SampleLayer) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("API_SAMPLE_L,0");
  node_proto.set_op("API_SAMPLE_L");
  node_proto.add_inputs("l_root");
  node_proto.add_inputs("edge_types");
  node_proto.add_inputs("0");

  std::vector<uint64_t> l_root = {1, 2, 3, 4, 5, 6};
  std::vector<int32_t> edge_types = {0, 1};
  TensorShape l_root_shape({l_root.size()});
  TensorShape edge_types_shape({edge_types.size()});
  Tensor* l_root_t = nullptr;
  Tensor* edge_types_t = nullptr;
  ctx.Allocate("l_root", l_root_shape, DataType::kUInt64, &l_root_t);
  ctx.Allocate("edge_types", edge_types_shape, DataType::kInt32, &edge_types_t);
  std::copy(l_root.begin(), l_root.end(), l_root_t->Raw<uint64_t>());
  std::copy(edge_types.begin(), edge_types.end(), edge_types_t->Raw<int32_t>());

  // run
  OpKernel* sample_layer = nullptr;
  CreateOpKernel("API_SAMPLE_L", &sample_layer);
  sample_layer->Compute(node_proto, &ctx);

  // check
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> nb;
  nb[1].insert(2); nb[1].insert(3); nb[1].insert(4);
  nb[2].insert(3); nb[2].insert(5);
  nb[3].insert(4);
  nb[4].insert(5);
  nb[5].insert(2); nb[5].insert(6);
  nb[6].insert(1); nb[6].insert(3); nb[6].insert(5);
  std::unordered_map<uint64_t, float> nbs_w;
  std::unordered_map<uint64_t, int32_t> nbs_t;
  nbs_w[1] = 1; nbs_w[2] = 2; nbs_w[3] = 3;
  nbs_w[4] = 4; nbs_w[5] = 5; nbs_w[6] = 6;
  nbs_t[1] = 1; nbs_t[2] = 0; nbs_t[3] = 1;
  nbs_t[4] = 0; nbs_t[5] = 1; nbs_t[6] = 0;
  Tensor* o_l_nb = nullptr;
  Tensor* o_l_w = nullptr;
  Tensor* o_l_t = nullptr;
  ctx.tensor("API_SAMPLE_L,0:0", &o_l_nb);
  ctx.tensor("API_SAMPLE_L,0:1", &o_l_w);
  ctx.tensor("API_SAMPLE_L,0:2", &o_l_t);
  ASSERT_EQ(l_root.size(), o_l_nb->Shape().Dims()[0]);
  ASSERT_EQ(1, o_l_nb->Shape().Dims()[1]);
  ASSERT_EQ(l_root.size(), o_l_w->Shape().Dims()[0]);
  ASSERT_EQ(1, o_l_w->Shape().Dims()[1]);
  ASSERT_EQ(l_root.size(), o_l_t->Shape().Dims()[0]);
  ASSERT_EQ(1, o_l_t->Shape().Dims()[1]);
  for (size_t i = 0; i < l_root.size(); ++i) {
    uint64_t root_id = l_root[i];
    uint64_t nb_id = o_l_nb->Raw<uint64_t>()[i];
    float nb_w = o_l_w->Raw<float>()[i];
    int32_t nb_t = o_l_t->Raw<int32_t>()[i];
    ASSERT_TRUE(nb[root_id].find(nb_id) != nb[root_id].end());
    ASSERT_EQ(nbs_w[nb_id], nb_w);
    ASSERT_EQ(nbs_t[nb_id], nb_t);
  }
}

TEST_F(LayerwiseSampleTest, LocalSampleLayerTest) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("API_LOCAL_SAMPLE_L,0");
  node_proto.set_op("API_LOCAL_SAMPLE_L");
  node_proto.add_inputs("batch_nb_idx");
  node_proto.add_inputs("batch_nb_id");
  node_proto.add_inputs("batch_nb_w");
  node_proto.add_inputs("batch_nb_t");
  node_proto.add_inputs("n");
  node_proto.add_inputs("m");
  node_proto.add_inputs("sqrt");
  node_proto.add_inputs("-1");

  // batch * n * 2, batch = 2, n = 3
  std::vector<int32_t> batch_nb_idx = {0, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 14};
  std::vector<uint64_t> batch_nb_id =
    {1, 2, 3, 3, 5, 5, 7, 8, 9, 9, 11, 11, 13, 14};
  std::vector<float> batch_nb_w =
    {1, 2, 3, 3, 5, 5, 7, 8, 9, 9, 11, 11, 13, 14};
  std::vector<int32_t> batch_nb_t = {1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0};
  int32_t n = 3;
  int32_t m = 6;

  TensorShape batch_nb_idx_shape = {6, 2};
  TensorShape batch_nb_shape = {14};
  Tensor* batch_nb_idx_t = nullptr;
  Tensor* batch_nb_id_t = nullptr;
  Tensor* batch_nb_w_t = nullptr;
  Tensor* batch_nb_t_t = nullptr;
  Tensor* n_t = nullptr;
  Tensor* m_t = nullptr;
  ctx.Allocate("batch_nb_idx", batch_nb_idx_shape,
               DataType::kInt32, &batch_nb_idx_t);
  ctx.Allocate("batch_nb_id", batch_nb_shape,
               DataType::kUInt64, &batch_nb_id_t);
  ctx.Allocate("batch_nb_w", batch_nb_shape,
               DataType::kFloat, &batch_nb_w_t);
  ctx.Allocate("batch_nb_t", batch_nb_shape,
               DataType::kInt32, &batch_nb_t_t);
  ctx.Allocate("n", {1}, DataType::kInt32, &n_t);
  ctx.Allocate("m", {1}, DataType::kInt32, &m_t);
  std::copy(batch_nb_idx.begin(), batch_nb_idx.end(),
            batch_nb_idx_t->Raw<int32_t>());
  std::copy(batch_nb_id.begin(), batch_nb_id.end(),
            batch_nb_id_t->Raw<uint64_t>());
  std::copy(batch_nb_w.begin(), batch_nb_w.end(), batch_nb_w_t->Raw<float>());
  std::copy(batch_nb_t.begin(), batch_nb_t.end(), batch_nb_t_t->Raw<int32_t>());
  n_t->Raw<int32_t>()[0] = n;
  m_t->Raw<int32_t>()[0] = m;

  OpKernel* local_sample_layer = nullptr;
  CreateOpKernel("API_LOCAL_SAMPLE_L", &local_sample_layer);
  local_sample_layer->Compute(node_proto, &ctx);

  // check
  Tensor* o_l_nb = nullptr;
  ctx.tensor("API_LOCAL_SAMPLE_L,0:0", &o_l_nb);
  std::unordered_set<uint64_t> batch1_nb = {1, 2, 3, 5, 7};
  std::unordered_set<uint64_t> batch2_nb = {8, 9, 11, 13, 14};
  ASSERT_EQ(12, o_l_nb->NumElements());
  for (int32_t i = 0; i < 6; ++i) {  // batch1 nb
    ASSERT_TRUE(batch1_nb.find(o_l_nb->Raw<uint64_t>()[i]) != batch1_nb.end());
  }
  for (int32_t i = 6; i < 12; ++i) {  // batch2 nb
    ASSERT_TRUE(batch2_nb.find(o_l_nb->Raw<uint64_t>()[i]) != batch2_nb.end());
  }
}

TEST_F(LayerwiseSampleTest, SparseGenAdjOpTest) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("API_SPARSE_GEN_ADJ,0");
  node_proto.set_op("API_SPARSE_GEN_ADJ");
  node_proto.add_inputs("roots");
  node_proto.add_inputs("l_nb");
  node_proto.add_inputs("n");

  int32_t batch = 2;
  int32_t n = 4;
  int32_t m = 4;
  std::vector<uint64_t> roots = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint64_t> l_nb = {10, 11, 12, 13, 14, 15, 16, 17};
  size_t bn = batch * n;
  size_t bm = batch * m;
  TensorShape roots_shape({bn, 1});
  TensorShape l_nb_shape({bm , 1});
  TensorShape scalar_shape({1});

  Tensor* roots_t = nullptr;
  Tensor* l_nb_t = nullptr;
  Tensor* n_t = nullptr;

  ctx.Allocate("roots", roots_shape, DataType::kUInt64, &roots_t);
  ctx.Allocate("l_nb", l_nb_shape, DataType::kUInt64, &l_nb_t);
  ctx.Allocate("n", scalar_shape, DataType::kInt32, &n_t);

  std::copy(roots.data(), roots.data() + bn,
            roots_t->Raw<uint64_t>());
  std::copy(l_nb.data(), l_nb.data() + bm,
            l_nb_t->Raw<uint64_t>());
  n_t->Raw<int32_t>()[0] = n;

  OpKernel* sparse_gen_adj = nullptr;
  CreateOpKernel("API_SPARSE_GEN_ADJ", &sparse_gen_adj);
  sparse_gen_adj->Compute(node_proto, &ctx);

  std::vector<uint64_t> roots_batch =
    {0, 0, 1, 0, 2, 0, 3, 0, 4, 1, 5, 1, 6, 1, 7, 1};
  Tensor* output0 = nullptr, *output1 = nullptr;
  ctx.tensor("API_SPARSE_GEN_ADJ,0:0", &output0);
  ctx.tensor("API_SPARSE_GEN_ADJ,0:1", &output1);
  ASSERT_EQ(roots_batch.size(), output0->NumElements());
  for (int32_t i = 0; i < output0->NumElements(); ++i) {
    ASSERT_EQ(roots_batch[i], output0->Raw<uint64_t>()[i]);
  }
  ASSERT_EQ(l_nb.size(), output1->NumElements());
  for (int32_t i = 0; i < output1->NumElements(); ++i) {
    ASSERT_EQ(l_nb[i], output1->Raw<uint64_t>()[i]);
  }
}

TEST_F(LayerwiseSampleTest, SparseGetAdjOpTest) {
  OpKernelContext ctx;

  // create op proto
  DAGNodeProto node_proto;
  node_proto.set_name("API_SPARSE_GET_ADJ,0");
  node_proto.set_op("API_SPARSE_GET_ADJ");
  node_proto.add_inputs("roots_batch");
  node_proto.add_inputs("l_nb");
  node_proto.add_inputs("edge_types");
  node_proto.add_inputs("m");

  Tensor* roots_batch_t = nullptr;
  Tensor* l_nb_t = nullptr;
  Tensor* edge_types_t = nullptr;
  Tensor* m_t = nullptr;

  std::vector<uint64_t> roots_batch = {1, 0, 2, 0, 3, 0, 4, 1, 5, 1, 6, 1};
  std::vector<uint64_t> l_nb = {1, 2, 3, 4, 5, 6};
  std::vector<int32_t> edges = {0, 1};

  ctx.Allocate("roots_batch", {6, 2}, DataType::kUInt64, &roots_batch_t);
  ctx.Allocate("l_nb", {6, 1}, DataType::kUInt64, &l_nb_t);
  ctx.Allocate("edge_types", {2}, DataType::kInt32, &edge_types_t);
  ctx.Allocate("m", {1}, DataType::kInt32, &m_t);

  std::copy(roots_batch.begin(), roots_batch.end(),
            roots_batch_t->Raw<uint64_t>());
  std::copy(l_nb.begin(), l_nb.end(), l_nb_t->Raw<uint64_t>());
  std::copy(edges.begin(), edges.end(), edge_types_t->Raw<int32_t>());
  m_t->Raw<int32_t>()[0] = 3;

  std::vector<std::vector<uint64_t>> sparse_adj(6);
  sparse_adj[0] = {2, 3};  // 1->2,3
  sparse_adj[1] = {3};    // 2->3
  sparse_adj[2] = {};     // 3->
  sparse_adj[3] = {5};    // 4->5
  sparse_adj[4] = {6};    // 5->6
  sparse_adj[5] = {5};    // 6->5

  OpKernel* sparse_get_adj = nullptr;
  CreateOpKernel("API_SPARSE_GET_ADJ", &sparse_get_adj);
  sparse_get_adj->Compute(node_proto, &ctx);

  Tensor* idx = nullptr, *adj = nullptr;
  ctx.tensor("API_SPARSE_GET_ADJ,0:0", &idx);
  ctx.tensor("API_SPARSE_GET_ADJ,0:1", &adj);
  for (int32_t i = 0; i < idx->NumElements() / 2; ++i) {
    int32_t begin = idx->Raw<int32_t>()[i * 2];
    int32_t end = idx->Raw<int32_t>()[i * 2 + 1];
    for (int32_t j = begin; j < end; ++j) {
      ASSERT_EQ(sparse_adj[i][j - begin], adj->Raw<uint64_t>()[j]);
    }
  }
}

}  // namespace euler
