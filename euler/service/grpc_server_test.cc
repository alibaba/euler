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

#include "euler/service/grpc_server.h"

#include <string>
#include <vector>
#include <utility>

#include "gtest/gtest.h"
#include "grpcpp/grpcpp.h"
#include "grpcpp/impl/codegen/proto_utils.h"
#include "grpcpp/generic/generic_stub.h"

#include "euler/common/logging.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/types.pb.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/kernels/common.h"

namespace euler {

Mutex mu;  // Global mutex to guard test run sequentialy

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
  std::string outname = OutputName(node_def, 0);
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

REGISTER_OP_KERNEL("add", Add);

class GrpcServerTest: public ::testing::Test {
 protected:
  void SetUp() override {
    mu.Lock();  // Guard for sequentialy run

    // Create a grpc server and start it
    ServerDef server_def = {"grpc", 0, 1, {}};
    auto& options = server_def.options;
    options.insert({"port", "9090"});
    options.insert({"num_threads", "2"});
    options.insert({"data_path", "/tmp/euler"});
    options.insert({"zk_server", "localhost:2181"});
    options.insert({"zk_path", "/euler_grpc_server_test"});

    auto s = NewServer(server_def, &server_);
    ASSERT_TRUE(s.ok()) << s.DebugString();
    s = server_->Start();
    ASSERT_TRUE(s.ok()) << s;
    sleep(3);
  }

  void TearDown() override {
    if (server_) {
      auto s = server_->Stop();
      ASSERT_TRUE(s.ok()) << s;
    }
    sleep(3);  // Wait for port to cleanup
    mu.Unlock();
  }

  std::unique_ptr<ServerInterface> server_;
};

template <typename Request, typename Response>
void SendRequest(const std::string& method,
                   const Request& req, Response* resp) {
  std::shared_ptr<grpc::Channel> channel =
      grpc::CreateChannel("0.0.0.0:9090", grpc::InsecureChannelCredentials());
  grpc::GenericStub stub(channel);
  grpc::CompletionQueue cq;
  grpc::ClientContext context;
  grpc::ByteBuffer request_buf;
  bool own_buffer = false;
  grpc::Status s = grpc::GenericSerialize<
    grpc::ProtoBufferWriter,
    google::protobuf::Message>(req, &request_buf, &own_buffer);
  ASSERT_TRUE(s.ok()) << "Serialize rpc request failed";
  auto resp_reader = stub.PrepareUnaryCall(&context, method, request_buf, &cq);
  resp_reader->StartCall();
  grpc::ByteBuffer response_buf;
  resp_reader->Finish(&response_buf, &s, reinterpret_cast<void*>(1));
  void* tag;
  bool ok = false;
  ASSERT_TRUE(cq.Next(&tag, &ok));
  ASSERT_TRUE(ok);
  ASSERT_EQ(reinterpret_cast<void*>(1), tag);
  ASSERT_TRUE(s.ok()) << "Server return error status";
  grpc::ProtoBufferReader reader(&response_buf);
  ASSERT_TRUE(resp->ParseFromZeroCopyStream(&reader));
}

TEST_F(GrpcServerTest, Ping) {
  PingRequest request;
  PingReply response;
  SendRequest("euler.EulerService/Ping", request, &response);
  ASSERT_EQ(std::string("Pong"), response.content());
  EULER_LOG(INFO) << response.DebugString();
}

TEST_F(GrpcServerTest, Execute) {
  ExecuteRequest request;
  ExecuteReply response;
  std::vector<int32_t> v1(5, 1);
  std::vector<int32_t> v2(5, 2);
  std::vector<int32_t> v3(5, 3);

  auto input1 = request.mutable_inputs()->Add();
  auto input2 = request.mutable_inputs()->Add();
  auto input3 = request.mutable_inputs()->Add();

  input1->set_name("A");
  input1->set_dtype(DataTypeProto::DT_INT32);
  input1->set_tensor_content(
      std::string(reinterpret_cast<const char*>(v1.data()),
                  v1.size() * sizeof(v1[0])));
  input1->mutable_tensor_shape()->mutable_dims()->Add(5);

  input2->set_name("B");
  input2->set_dtype(DataTypeProto::DT_INT32);
  input2->set_tensor_content(
      std::string(reinterpret_cast<const char*>(v2.data()),
                  v2.size() * sizeof(v2[0])));
  input2->mutable_tensor_shape()->mutable_dims()->Add(5);

  input3->set_name("C");
  input3->set_dtype(DataTypeProto::DT_INT32);
  input3->set_tensor_content(
      std::string(reinterpret_cast<const char*>(v3.data()),
                  v3.size() * sizeof(v3[0])));
  input3->mutable_tensor_shape()->mutable_dims()->Add(5);

  auto node1 = request.mutable_graph()->mutable_nodes()->Add();
  auto node2 = request.mutable_graph()->mutable_nodes()->Add();

  node1->set_name("add1");
  node1->set_op("add");
  node1->mutable_inputs()->Add("A");
  node1->mutable_inputs()->Add("B");

  node2->set_name("add2");
  node2->set_op("add");
  node2->mutable_inputs()->Add("C");
  node2->mutable_inputs()->Add("add1:0");

  request.mutable_outputs()->Add("add2:0");
  request.mutable_outputs()->Add("add1:0");
  SendRequest("euler.EulerService/Execute", request, &response);

  auto& proto = response.outputs(0);
  std::vector<int32_t> ov(5, 0);
  ASSERT_EQ(ov.size() * sizeof(ov[0]), proto.tensor_content().size());
  memcpy(&ov[0], proto.tensor_content().c_str(), proto.tensor_content().size());
  for (auto& oov : ov) {
    ASSERT_EQ(6, oov);
  }

  auto& proto1 = response.outputs(1);
  std::vector<int32_t> ov1(5, 0);
  ASSERT_EQ(ov1.size() * sizeof(ov1[0]), proto1.tensor_content().size());
  memcpy(&ov1[0], proto1.tensor_content().c_str(),
         proto1.tensor_content().size());
  for (auto& oov : ov1) {
    ASSERT_EQ(3, oov);
  }

  EULER_LOG(INFO) << response.DebugString();
}

TEST_F(GrpcServerTest, SampleNode) {
  auto graph = EulerGraph();
  EULER_LOG(INFO) << graph->graph_meta().ToString();

  ExecuteRequest request;
  ExecuteReply response;

  int32_t node_type = 0;
  int32_t count = 10;

  auto input1 = request.mutable_inputs()->Add();
  auto input2 = request.mutable_inputs()->Add();

  input1->set_name("node_type");
  input1->set_dtype(DataTypeProto::DT_INT32);
  input1->set_tensor_content(
      std::string(reinterpret_cast<const char*>(&node_type),
                  sizeof(node_type)));
  input1->mutable_tensor_shape()->mutable_dims()->Add(1);

  input2->set_name("count");
  input2->set_dtype(DataTypeProto::DT_INT32);
  input2->set_tensor_content(
      std::string(reinterpret_cast<const char*>(&count), sizeof(count)));
  input2->mutable_tensor_shape()->mutable_dims()->Add(1);

  auto node = request.mutable_graph()->mutable_nodes()->Add();
  node->set_name("sample_node");
  node->set_op("API_SAMPLE_NODE");
  node->mutable_inputs()->Add("node_type");
  node->mutable_inputs()->Add("count");

  request.mutable_outputs()->Add("sample_node:0");
  SendRequest("euler.EulerService/Execute", request, &response);
  EULER_LOG(INFO) << response.DebugString();

  auto& proto = response.outputs(0);
  ASSERT_EQ("sample_node:0", proto.name());
  ASSERT_EQ(DataTypeProto::DT_INT64, proto.dtype());
  ASSERT_EQ(1, proto.tensor_shape().dims_size());
  ASSERT_EQ(count, proto.tensor_shape().dims(0));
  ASSERT_EQ(sizeof(int64_t) * count, proto.tensor_content().size());

  auto data = reinterpret_cast<const int64_t*>(proto.tensor_content().c_str());
  for (int32_t i = 0; i < count; ++i) {
    EULER_LOG(INFO) << data[i];
  }
}

TEST_F(GrpcServerTest, GetFeature) {
  ExecuteRequest request;
  ExecuteReply response;

  std::vector<uint64_t> node_ids({1, 2, 3, 4});
  std::vector<std::string> ft_names({"1", "2", "3"});

  auto input1 = request.mutable_inputs()->Add();

  input1->set_name("node_ids");
  input1->set_dtype(DataTypeProto::DT_UINT64);
  input1->set_tensor_content(
      std::string(reinterpret_cast<const char*>(node_ids.data()),
                  sizeof(node_ids[0]) * node_ids.size()));
  input1->mutable_tensor_shape()->mutable_dims()->Add(node_ids.size());

  auto node = request.mutable_graph()->mutable_nodes()->Add();
  node->set_name("get_feature");
  node->set_op("API_GET_P");
  node->mutable_inputs()->Add("node_ids");
  for (auto& ft_name : ft_names) {
    auto feature_input = request.mutable_inputs()->Add();

    std::string buffer;
    uint32_t len = ft_name.size();
    buffer.append(reinterpret_cast<char*>(&len), sizeof(len));
    buffer.append(ft_name);

    std::string feature_name = "feature_name_" + ft_name;
    feature_input->set_name(feature_name);
    feature_input->set_dtype(DataTypeProto::DT_STRING);
    feature_input->set_tensor_content(buffer);
    feature_input->mutable_tensor_shape()->mutable_dims()->Add(1);
    node->mutable_inputs()->Add(std::move(feature_name));
  }

  request.mutable_outputs()->Add("get_feature:0");
  request.mutable_outputs()->Add("get_feature:1");
  request.mutable_outputs()->Add("get_feature:2");
  request.mutable_outputs()->Add("get_feature:3");
  request.mutable_outputs()->Add("get_feature:4");
  request.mutable_outputs()->Add("get_feature:5");

  SendRequest("euler.EulerService/Execute", request, &response);
  EULER_LOG(INFO) << response.DebugString();

  for (size_t j = 0; j < ft_names.size(); ++j) {
    OpKernelContext context;
    auto& proto = response.outputs(2 * j);
    ASSERT_TRUE(context.Allocate("indexs", proto).ok());
    Tensor* indexs_t = nullptr;
    ASSERT_TRUE(context.tensor("indexs", &indexs_t).ok());
    ASSERT_EQ(DataType::kInt32, indexs_t->Type());
    ASSERT_EQ(2, indexs_t->Shape().Size());
    ASSERT_EQ(4, indexs_t->Shape().Dims()[0]);
    ASSERT_EQ(2, indexs_t->Shape().Dims()[1]);

    auto& proto1 = response.outputs(2 * j + 1);
    ASSERT_TRUE(context.Allocate("values", proto1).ok());
    Tensor* values_t = nullptr;
    ASSERT_TRUE(context.tensor("values", &values_t).ok());
    ASSERT_EQ(DataType::kFloat, values_t->Type());
    ASSERT_EQ(1, values_t->Shape().Size());
    ASSERT_EQ(40, values_t->Shape().Dims()[0]);

    auto indexs = indexs_t->Raw<int32_t>();
    auto values = values_t->Raw<float>();
    for (size_t i = 0; i < node_ids.size(); ++i) {
      for (int k = indexs[0]; k < indexs[1]; ++k) {
        ASSERT_EQ(atoi(ft_names[j].c_str()) + 0.5, values[k]);
      }
      indexs += 2;
    }
  }
}


}  // namespace euler
