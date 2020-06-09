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

#include "euler/core/framework/executor.h"

#include <utility>

#include "euler/common/env.h"
#include "euler/common/signal.h"
#include "euler/common/logging.h"
#include "euler/core/dag/node.h"
#include "euler/core/dag/edge.h"
#include "euler/core/framework/op_kernel.h"

namespace euler {

Executor::Executor(DAG* dag, ThreadPool* thread_pool, OpKernelContext* ctx)
    : dag_(dag), thread_pool_(thread_pool), ctx_(ctx),
      ref_(dag->num_nodes()), remain_node_(dag->num_nodes()) {
  for (auto& node : dag->nodes()) {
    ref_[node->id()] = node->num_inputs();
  }
}

void Executor::Run() {
  Signal sig;
  Run([&sig] () { sig.Notify(); });
  sig.Wait();
}

void Executor::RunInternal() {
  std::vector<DAGNode*> frontier;
  for (auto& node : dag_->nodes()) {
    if (ref_[node->id()] == 0) {
      frontier.emplace_back(node);
    }
  }
  for (auto& node : frontier) {
    Run(node);
  }
}

void Executor::Run(DoneCallback callback) {
  callback_ = std::move(callback);
  RunInternal();
}

void Executor::Run(DAGNode* node) {
  OpKernel* op_base = nullptr;

  auto s = CreateOpKernel(node->op(), &op_base);
  if (!s.ok()) {
    EULER_LOG(FATAL) << "Create kernel: " << node->name() << " failed!";
  }

  AsyncOpKernel* op = dynamic_cast<AsyncOpKernel*>(op_base);
  if (op != nullptr) {  // Async op
    thread_pool_->Schedule([this, node, op] () {
        op->AsyncCompute(node->def(), ctx_, [this, node] () {
            RunDone(node);
        });
    });
  } else {
    thread_pool_->Schedule([this, node, op_base] () {
        op_base->Compute(node->def(), ctx_);
        RunDone(node);});
  }
}

void Executor::RunDone(DAGNode* node) {
  for (auto edge : node->output_edges()) {
    auto dst = edge->dst();
    if (--ref_[dst->id()] == 0) {
      Run(dst);
    }
  }

  if (--remain_node_ == 0) {
    callback_();
  }
}

}  // namespace euler
