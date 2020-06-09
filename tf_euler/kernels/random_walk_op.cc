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

#include <math.h>

#include <memory>
#include <vector>
#include <functional>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "euler/common/compact_weighted_collection.h"
#include "euler/common/data_types.h"
#include "tf_euler/utils/euler_query_proxy.h"

namespace tensorflow {

namespace {

class RWCallback {
 public:
  RWCallback(const std::vector<std::vector<int64_t>>& parent_neighbors,
             const std::vector<int64_t>& parent_ids,
             const std::vector<std::vector<int>>& edge_types,
             int walk_len, float p, float q, int64_t default_node,
             Tensor* output, AsyncOpKernel::DoneCallback done)
      : parent_neighbors_(parent_neighbors), parent_ids_(parent_ids),
        edge_types_(edge_types), walk_len_(walk_len), p_(p), q_(q),
        default_node_(default_node), output_(output), done_(done),
        call_time_(0), query_(nullptr) {
  }

  void set_query(euler::Query* query) { query_ = query; }

  void operator() ();

 private:
  void BuildWeights(const std::vector<int64_t>& parent_neighbors,
                    const std::vector<int64_t>& child_neighbors,
                    int64_t parent_id, std::vector<float>* weights);

 private:
  std::vector<std::vector<int64_t>> parent_neighbors_;
  std::vector<int64_t> parent_ids_;
  std::vector<std::vector<int>> edge_types_;
  int walk_len_;
  float p_;
  float q_;
  int64_t default_node_;
  Tensor* output_;
  AsyncOpKernel::DoneCallback done_;
  int call_time_;
  euler::Query* query_;
};

void GetFullNeighbor(const std::vector<int64_t>& node_ids,
                     const std::vector<int32>& edge_types,
                     RWCallback done) {
  auto query = new euler::Query("v(nodes).outV(edge_types).as(nb)");
  auto nodes_t = query->AllocInput("nodes", {node_ids.size()}, euler::kUInt64);
  auto edge_types_t = query->AllocInput(
      "edge_types", {edge_types.size()}, euler::kInt32);
  std::copy(node_ids.begin(), node_ids.end(), nodes_t->Raw<uint64_t>());
  std::copy(edge_types.begin(), edge_types.end(), edge_types_t->Raw<int32_t>());
  done.set_query(query);
  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, done);
}

void RWCallback::operator() () {
  if (query_ == nullptr) {
    return;
  }

  auto idx_t = query_->GetResult("nb:0");
  auto ids_t = query_->GetResult("nb:1");
  auto weights_t = query_->GetResult("nb:2");
  std::vector<std::vector<int64_t>> neighbors(parent_neighbors_.size());
  std::vector<std::vector<float>> weights(parent_neighbors_.size());
  for (size_t i = 0; i < neighbors.size(); ++i) {
    int start = idx_t->Raw<int32_t>()[2 * i];
    int end = idx_t->Raw<int32_t>()[2 * i + 1];
    neighbors[i].reserve(end - start);
    weights[i].reserve(end - start);
    for (int j = start; j < end; ++j) {
      neighbors[i].emplace_back(ids_t->Raw<int64_t>()[j]);
      weights[i].emplace_back(weights_t->Raw<float>()[j]);
    }
  }

  // Sample neighbor
  auto data = output_->flat<int64>().data();
  std::vector<int64_t> node_ids(neighbors.size());
  for (size_t i = 0; i < neighbors.size(); ++i) {
    auto& cn = neighbors[i];
    int64_t sample_id = default_node_;
    if (!cn.empty()) {
      auto parent_id = parent_ids_[i];
      auto& pn = parent_neighbors_[i];
      auto& w = weights[i];
      BuildWeights(pn, cn, parent_id, &w);

      using CWC = euler::common::CompactWeightedCollection<int64_t>;
      CWC sampler;
      sampler.Init(cn, w);
      sample_id = sampler.Sample().first;
    }

    data[i * (walk_len_ + 1) + call_time_ + 1] = sample_id;
    node_ids[i] = sample_id;
  }

  ++call_time_;
  if (call_time_ < walk_len_) {
    parent_neighbors_ = neighbors;
    auto nodes_t = query_->GetResult("nodes");
    auto nodes_data = nodes_t->Raw<uint64_t>();
    std::copy(nodes_data, nodes_data + parent_ids_.size(), parent_ids_.begin());
    GetFullNeighbor(node_ids, edge_types_[call_time_], *this);
  } else {
    done_();
  }

  delete query_;
}

void RWCallback::BuildWeights(const std::vector<int64_t>& parent_neighbors,
                              const std::vector<int64_t>& child_neighbors,
                              int64_t parent_id, std::vector<float>* weights) {
  size_t j = 0;
  size_t k = 0;
  while (j < child_neighbors.size() && k < parent_neighbors.size()) {
    if (child_neighbors[j] < parent_neighbors[k]) {
      if (child_neighbors[j] != parent_id) {
        weights->at(j) /= q_;  // d_tx = 2
      } else {
        weights->at(j) /=  p_;  // d_tx = 0
      }
      ++j;
    } else if (child_neighbors[j] == parent_neighbors[k]) {
      ++k;
      ++j;
    } else {
      ++k;
    }
  }
  while (j < child_neighbors.size()) {
    if (child_neighbors[j] != parent_id) {
      weights->at(j) /= q_;  // d_tx = 2
    } else {
      weights->at(j) /= p_;  // d_tx = 0
    }
    ++j;
  }
}

}  // namespace

class RandomWalk: public AsyncOpKernel {
 public:
  explicit RandomWalk(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("walk_len", &walk_len_));
    OP_REQUIRES_OK(context, context->GetAttr("p", &p_));
    OP_REQUIRES_OK(context, context->GetAttr("q", &q_));
    OP_REQUIRES_OK(context, context->GetAttr("default_node", &default_node_));
    if (p_ == 1 && q_ == 1) {
      std::stringstream ss;
      ss << "v(nodes)";
      for (size_t i = 0; i < walk_len_; ++i) {
        ss << ".sampleNB(et_" << i << ", nb_count_, " << default_node_ << ")"
           << ".as(nb_" << i << ")";
        res_names_.push_back(euler::ToString("nb_", i, ":1"));
      }
      query_str_ = ss.str();
    }
  }

  void TraditionalRandomWalk(const std::vector<int64_t>& node_ids,
                             const std::vector<std::vector<int32>>& edge_types,
                             DoneCallback done,
                             Tensor* output);

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  int walk_len_;
  float p_;
  float q_;
  int default_node_;
  std::string query_str_;
  std::vector<std::string> res_names_;
};

void RandomWalk::TraditionalRandomWalk(
    const std::vector<int64_t>& node_ids,
    const std::vector<std::vector<int32>>& edge_types,
    DoneCallback done,
    Tensor* output) {
  auto query = new euler::Query(query_str_);
  auto t_nodes = query->AllocInput("nodes", {node_ids.size()}, euler::kUInt64);
  std::copy(node_ids.begin(), node_ids.end(), t_nodes->Raw<uint64_t>());
  for (size_t i = 0; i < walk_len_; ++i) {
    auto t_edge_types =
        query->AllocInput(euler::ToString("et_", i),
        {edge_types[i].size()}, euler::kInt32);
    for (size_t j = 0; j < edge_types[i].size(); ++j) {
      t_edge_types->Raw<int32_t>()[j] = edge_types[i][j];
    }
  }
  auto t_count = query->AllocInput("nb_count_", {1}, euler::kInt32);
  *(t_count->Raw<int32_t>()) = 1;
  size_t nodes_size = node_ids.size();

  auto output_data = output->flat<int64>().data();
  for (size_t i = 0; i < nodes_size; ++i) {
    output_data[i * (walk_len_ + 1)] = node_ids[i];
  }

  auto callback = [output, done, nodes_size, query, this]() {
    auto output_data = output->flat<int64>().data();
    for (size_t i = 0; i < nodes_size; ++i) {
      for (size_t j = 0; j < walk_len_; ++j) {
        auto layers_t = query->GetResult(res_names_[j]);
        uint64_t nb_id = layers_t->Raw<uint64_t>()[i] ==
            euler::common::DEFAULT_UINT64 ? default_node_ :
            layers_t->Raw<uint64_t>()[i];
        output_data[i * (walk_len_ + 1) + j + 1] = nb_id;
      }
    }
    delete query;
    done();
  };
  euler::QueryProxy::GetInstance()->RunAsyncGremlin(query, callback);
}

void RandomWalk::ComputeAsync(OpKernelContext* ctx, DoneCallback done) {
  auto nodes = ctx->input(0).flat<int64>();
  OpInputList etypes;
  OP_REQUIRES_OK(ctx, ctx->input_list("edge_types", &etypes));

  OP_REQUIRES(ctx, etypes.size() == walk_len_,
              errors::InvalidArgument("edge_types must with size of walk_len"));

  std::vector<std::vector<int32>> edge_types(walk_len_);
  for (auto i = 0; i < walk_len_; ++i) {
    auto etypes_flat = etypes[i].flat<int32>();
    edge_types[i].resize(etypes_flat.size());
    memcpy(edge_types[i].data(), etypes_flat.data(),
           sizeof(edge_types[i][0]) * edge_types[i].size());
  }

  TensorShape output_shape;
  output_shape.AddDim(nodes.size());
  output_shape.AddDim(walk_len_ + 1);

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));

  std::vector<std::vector<int64_t>> parent_neighbors(nodes.size());
  std::vector<int64_t> node_ids(nodes.size(), -1);
  memcpy(node_ids.data(), nodes.data(), sizeof(node_ids[0]) * nodes.size());

  auto data = output->flat<int64>().data();
  for (auto i = 0; i < node_ids.size(); ++i) {
    data[i * (walk_len_ + 1)] = node_ids[i];
  }

  const float kEps = 1.0e-6;
  if (fabs(p_ - 1.0) <= kEps && fabs(q_ - 1.0) <= kEps) {  // randomwalk
    TraditionalRandomWalk(node_ids, edge_types, done, output);
  } else {  // node2vec
    RWCallback callback(parent_neighbors, node_ids,  edge_types, walk_len_,
                        p_, q_, default_node_, output, done);
    GetFullNeighbor(node_ids, edge_types[0], callback);
  }
}

REGISTER_KERNEL_BUILDER(Name("RandomWalk").Device(DEVICE_CPU), RandomWalk);

}  // namespace tensorflow
