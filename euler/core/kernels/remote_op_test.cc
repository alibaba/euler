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
#include "grpcpp/grpcpp.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/generic/generic_stub.h"

#include "euler/service/grpc_server.h"
#include "euler/common/logging.h"
#include "euler/common/server_register.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/types.pb.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/dag_def/dag_def.h"
#include "euler/client/graph_config.h"
#include "euler/client/client_manager.h"

namespace euler {

template <typename T>
void VectorAdd(T* op1, T* op2, T* out, int num) {
  for (int i = 0; i < num; ++i) {
    *out = *op1 + *op2;
    ++out;
    ++op1;
    ++op2;
  }
}

class Add : public AsyncOpKernel {
 public:
  explicit Add(const std::string& name): AsyncOpKernel(name) { }

  void AsyncCompute(const DAGNodeProto& node_def,
                    OpKernelContext* ctx, DoneCallback callback) override;
};

void Add::AsyncCompute(const DAGNodeProto& node_def,
                       OpKernelContext* ctx, DoneCallback callback) {
  if (node_def.inputs_size() != 2) {
    callback();
    return;
  }

  Tensor* input1 = nullptr;
  Tensor* input2 = nullptr;
  TensorShape shape;
  DataType type;
  std::string outname = node_def.name() + ":0";
  Tensor* output = nullptr;

  auto s = ctx->tensor(node_def.inputs(0), &input1);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "No input tensor '" << node_def.inputs(0) << "'";
    goto error;
  }
  s =  ctx->tensor(node_def.inputs(1), &input2);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "No input tensor '" << node_def.inputs(1) << "'";
    goto error;
  }
  if (input1->Type() != input2->Type()) {
    EULER_LOG(ERROR) << "Input tensor DataType dismatch, input1 type:"
                     << input1->Type() << ", input2 type:" << input2->Type();
    goto error;
  }
  if (input1->Shape() != input2->Shape()) {
    EULER_LOG(ERROR) << "Input tensor shape dismatch, input1 shape:"
                     << input1->Shape().DebugString() << ", input2 shape:"
                     << input2->Shape().DebugString();
    goto error;
  }

  shape = input1->Shape();
  type = input1->Type();
  s = ctx->Allocate(outname, shape, type, &output);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor '"
                     << outname << "' failed Status: " << s;
    goto error;
  }

  switch (type) {
    case DataType::kInt16:
      VectorAdd(input1->Raw<int16_t>(), input2->Raw<int16_t>(),
                output->Raw<int16_t>(), shape.NumElements());
      break;
    case DataType::kInt32:
      VectorAdd(input1->Raw<int32_t>(), input2->Raw<int32_t>(),
                output->Raw<int32_t>(), shape.NumElements());
      break;
    case DataType::kInt64:
      VectorAdd(input1->Raw<int64_t>(), input2->Raw<int64_t>(),
                output->Raw<int64_t>(), shape.NumElements());
      break;
    case DataType::kFloat:
      VectorAdd(input1->Raw<float>(), input2->Raw<float>(),
                output->Raw<float>(), shape.NumElements());
      break;
    case DataType::kDouble:
      VectorAdd(input1->Raw<double>(), input2->Raw<double>(),
                output->Raw<double>(), shape.NumElements());
      break;
    default:
      EULER_LOG(ERROR) << "Invalid data type: " << type;
      break;
  }

error:
  callback();
}

REGISTER_OP_KERNEL("ADD", Add);

class RemoteOpTest: public ::testing::Test {
 protected:
  static const char zk_path_[];

  void SetUp() override {
    // Create a grpc server and start it
    ServerDef server_def = {"grpc", 0, 1, {}};
    server_def.options.insert({"port", "9090"});
    server_def.options.insert({"data_path", "/tmp/euler"});
    server_def.options.insert({"zk_server", "127.0.0.1:2181"});
    server_def.options.insert({"zk_path", zk_path_});
    auto s = NewServer(server_def, &server_);
    ASSERT_TRUE(s.ok()) << s.DebugString();
    s = server_->Start();
    ASSERT_TRUE(s.ok()) << s;
  }

  void TearDown() override {
    auto s = server_->Stop();
    ASSERT_TRUE(s.ok()) << s;
  }

  std::unique_ptr<ServerInterface> server_;
};

const char RemoteOpTest::zk_path_[] = "/euler-2.0-test";

TEST_F(RemoteOpTest, Execute) {
  GraphConfig graph_config;
  graph_config.Add("zk_server", "127.0.0.1:2181");
  graph_config.Add("zk_path", RemoteOpTest::zk_path_);
  graph_config.Add("num_retries", 1);
  ClientManager::Init(graph_config);

  OpKernelContext ctx;

  // create remote op proto
  std::shared_ptr<NodeDef> inner_node = std::make_shared<NodeDef>(
      "ADD", 0);
  inner_node->input_edges_.push_back({"REMOTE", 1, 0});
  inner_node->input_edges_.push_back({"REMOTE", 1, 1});

  std::vector<std::shared_ptr<NodeDef>> inner_nodes = {inner_node};
  std::vector<FusionOutput> outputs;
  outputs.push_back({"ADD", 0, 0, 0});
  NodeDef* remote_node = new RemoteNodeDef(
      "REMOTE", 1, 0, inner_nodes, outputs, 1);
  remote_node->input_edges_.push_back({"a", 0, 0});
  remote_node->input_edges_.push_back({"b", 1, 0});

  // put intput tensor into context
  std::vector<int32_t> a(1, 1);
  std::vector<int32_t> b(1, 2);
  Tensor* t_a = nullptr;
  Tensor* t_b = nullptr;
  TensorShape shape({1});
  DataType type = kInt32;
  ctx.Allocate("a,0:0", shape, type, &t_a);
  ctx.Allocate("b,1:0", shape, type, &t_b);
  *(t_a->Raw<int32_t>()) = a[0];
  *(t_b->Raw<int32_t>()) = b[0];

  // create remote op and run
  DAGNodeProto remote_proto;
  remote_node->ToProto(&remote_proto);
  OpKernel* op_base;
  CreateOpKernel("REMOTE", &op_base);
  AsyncOpKernel* remote_op = dynamic_cast<AsyncOpKernel*>(op_base);
  Signal s;
  remote_op->AsyncCompute(remote_proto, &ctx, [&s](){s.Notify();});
  s.Wait();
  // check results, tensor name is REMOTE,1:0
  Tensor* output = nullptr;
  ctx.tensor("REMOTE,1:0", &output);
  EULER_LOG(INFO) << "result = " << output->Raw<int32_t>()[0];
  ASSERT_EQ(3, output->Raw<int32_t>()[0]);
}

}  // namespace euler
