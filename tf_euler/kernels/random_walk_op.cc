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

#include <memory>
#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "euler/client/graph.h"

namespace tensorflow {

extern std::unique_ptr<euler::client::Graph>& Graph();

///////////////////////// RandomWalkCallback /////////////////////////

class RandomWalkCallback {
 public:
  RandomWalkCallback(const std::vector<euler::client::NodeID>& node_ids,
                     const std::vector<std::vector<int>>& edge_types,
                     int walk_len, float p, float q,
                     euler::client::NodeID default_node,
                     Tensor* output, AsyncOpKernel::DoneCallback done)
      : node_ids_(node_ids), edge_types_(edge_types),
        walk_len_(walk_len), p_(p), q_(q), default_node_(default_node),
        output_(output), done_(done), call_time_(0) {
  }

  void operator() (const euler::client::IDWeightPairVec& result);

 private:
  std::vector<euler::client::NodeID> node_ids_;
  std::vector<std::vector<int>> edge_types_;
  int walk_len_;
  float p_;
  float q_;
  euler::client::NodeID default_node_;
  Tensor* output_;
  AsyncOpKernel::DoneCallback done_;
  int call_time_;
};

void RandomWalkCallback::operator() (
    const euler::client::IDWeightPairVec& result) {
  auto data = output_->flat<int64>().data();
  auto parent_ids = node_ids_;
  for (size_t i = 0; i < result.size(); ++i) {
    if (result[i].empty()) {
      data[i * (walk_len_ + 1) + call_time_ + 1] = default_node_;
      node_ids_[i] = default_node_;
    } else {
      data[i * (walk_len_ + 1) + call_time_ + 1] = std::get<0>(result[i][0]);
      node_ids_[i] =  std::get<0>(result[i][0]);
    }
  }

  ++call_time_;
  if (call_time_ < walk_len_) {
    auto& edge_types = edge_types_[call_time_];
    auto& parent_edge_types = edge_types_[call_time_ - 1];
    Graph()->BiasedSampleNeighbor(node_ids_, parent_ids, edge_types,
                                  parent_edge_types, 1, p_, q_, *this);
  } else {
    done_();
  }
}


///////////////////////////// RandomWalk ////////////////////////////

class RandomWalk: public AsyncOpKernel {
 public:
  explicit RandomWalk(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("walk_len", &walk_len_));
    OP_REQUIRES_OK(context, context->GetAttr("p", &p_));
    OP_REQUIRES_OK(context, context->GetAttr("q", &q_));
    OP_REQUIRES_OK(context, context->GetAttr("default_node", &default_node_));
  }

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override;

 private:
  int walk_len_;
  float p_;
  float q_;
  int default_node_;
};

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

  std::vector<euler::client::NodeID> parent_ids(nodes.size(), -1);
  std::vector<euler::client::NodeID> node_ids(nodes.size(), -1);
  memcpy(node_ids.data(), nodes.data(), sizeof(node_ids[0]) * nodes.size());

  auto data = output->flat<int64>().data();
  for (auto i = 0; i < node_ids.size(); ++i) {
    data[i * (walk_len_ + 1)] = node_ids[i];
  }

  RandomWalkCallback callback(node_ids, edge_types, walk_len_,
                              p_, q_, default_node_, output, done);
  Graph()->BiasedSampleNeighbor(node_ids, parent_ids, edge_types[0],
                                edge_types[0], 1, p_, q_, callback);
}

REGISTER_KERNEL_BUILDER(Name("RandomWalk").Device(DEVICE_CPU), RandomWalk);

}  // namespace tensorflow
