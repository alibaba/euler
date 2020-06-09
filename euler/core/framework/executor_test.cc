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

#include <string>
#include <memory>

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "euler/core/dag/node.h"

namespace euler {
namespace {

class SimpleOp : public OpKernelAsync {
 public:
  explicit SimpleOp(const std::string &message) : message_(message) { }

  void Compute(OpKernelContext* ctx, Callback done) {
    LOG(INFO) << message_;
    done(0);
  }

 private:
  std::string message_;
};

TEST(ExecutorTest, RunSimple) {
  DAG dag("simple_graph");
  dag.add_node("a", 0, std::unique_ptr<SimpleOp>(new SimpleOp("a")));
  dag.add_node("b", 1, std::unique_ptr<SimpleOp>(new SimpleOp("b")));
  dag.add_node("c", 2, std::unique_ptr<SimpleOp>(new SimpleOp("c")));
  dag.add_edge(0, 0, 1, 0);
  dag.add_edge(1, 0, 2, 0);

  Executor executor(&dag, nullptr);
  executor.Run();
}


}  // namespace
}  // namespace euler
