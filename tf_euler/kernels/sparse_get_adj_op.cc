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

#include <memory>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tf_euler/utils/sparse_tensor_builder.h"
#include "tf_euler/utils/euler_query_proxy.h"

namespace tensorflow {

class SparseGetAdj: public AsyncOpKernel {
 public:
  explicit SparseGetAdj(OpKernelConstruction* ctx): AsyncOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("N", &N_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("M", &M_));
    EULER_LOG(INFO) << "M: " << M_ << " N: " << N_;
  }
  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;
 private:
  int32_t N_;
  int32_t M_;
};

void SparseGetAdj::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto batch_nodes = ctx->input(0);
  auto batch_nodes_flat = batch_nodes.flat<int64>();
  size_t batch_nodes_size = batch_nodes_flat.size();
  auto batch_nb_nodes = ctx->input(1);
  auto batch_nb_nodes_flat = batch_nb_nodes.flat<int64>();
  size_t batch_nb_nodes_size = batch_nb_nodes_flat.size();
  auto edge_types = ctx->input(2);
  auto edge_types_flat = edge_types.flat<int32>();
  size_t edge_types_size = edge_types_flat.size();

  int32_t N = N_, M = M_;
  if (N_ == -1) N = batch_nodes_size;
  if (M_ == -1) M = batch_nb_nodes_size;
  // batch_size = batch_nb_nodes_size / M;
  int32_t batch_size = batch_nodes_size / N;

  auto query = new euler::Query(
      "API_SPARSE_GET_ADJ",
      "get_adj", 2, {"root_batch", "l_nb"}, {"edge_types", "m"});
  auto root_batch_t = query->AllocInput(
      "root_batch", {batch_nodes_size, 2}, euler::kUInt64);
  auto l_nb_t = query->AllocInput(
      "l_nb", {batch_nb_nodes_size, 1}, euler::kUInt64);
  auto edge_types_t = query->AllocInput(
      "edge_types", {edge_types_size}, euler::kInt32);
  auto m_t = query->AllocInput("m", {1}, euler::kInt32);
  for (int32_t i = 0; i < batch_size; ++i) {
    for (int32_t j = 0; j < N; ++j) {
      int32_t cnt = i * N + j;
      root_batch_t->Raw<uint64_t>()[cnt * 2] = batch_nodes_flat(cnt);
      root_batch_t->Raw<uint64_t>()[cnt * 2 + 1] = i;
    }
  }
  std::copy(batch_nb_nodes_flat.data(),
            batch_nb_nodes_flat.data() + batch_nb_nodes_size,
            l_nb_t->Raw<uint64_t>());
  std::copy(edge_types_flat.data(),
            edge_types_flat.data() + edge_types_size,
            edge_types_t->Raw<int32_t>());
  m_t->Raw<int32_t>()[0] = M;

  auto callback = [batch_nodes_flat, batch_nb_nodes_flat,
       batch_size, ctx, done, query, N, M, this]() {
    std::vector<std::string> res_names = {"get_adj:0", "get_adj:1"};
    auto results_map = query->GetResult(res_names);
    auto idx_ptr = results_map["get_adj:0"];
    auto idx_data = idx_ptr->Raw<int32_t>();
    auto val_ptr = results_map["get_adj:1"];
    auto val_data = val_ptr->Raw<int64_t>();

    SparseTensorBuilder<int64, 3> adj_builder;
    std::set<std::pair<int64_t, int64_t>> relation_set;

    for (size_t i = 0; i < batch_size; ++i) {
      relation_set.clear();
      for (size_t j = N * i; j < N * (i + 1); ++j) {
        int32_t begin = idx_data[j * 2];
        int32_t end = idx_data[j * 2 + 1];
        for (size_t k = begin; k < end; ++k) {
          relation_set.insert(
              std::make_pair(batch_nodes_flat(j), val_data[k]));
        }
      }

      for (size_t j = 0; j < N; ++j) {
        int64_t src_id = batch_nodes_flat(j + N * i);
        for (size_t k = 0; k < M; ++k) {
          int64_t dst_id = batch_nb_nodes_flat(k + M * i);
          if (relation_set.find(
                  std::make_pair(src_id, dst_id)) != relation_set.end()) {
            adj_builder.emplace(
                {static_cast<int64>(i),
                 static_cast<int64>(j),
                 static_cast<int64>(k)}, 1);
          } else if (j == N - 1 && k == M - 1) {
            adj_builder.emplace(
                {static_cast<int64>(i),
                 static_cast<int64>(j),
                 static_cast<int64>(k)}, 0);
          }
        }
      }
    }

    (void) ctx->set_output("adj_indices", adj_builder.indices());
    (void) ctx->set_output("adj_values", adj_builder.values());
    (void) ctx->set_output("adj_shape", adj_builder.dense_shape());

    delete query;
    done();
  };

  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

REGISTER_KERNEL_BUILDER(
    Name("SparseGetAdj").Device(DEVICE_CPU), SparseGetAdj);

}  // namespace tensorflow
