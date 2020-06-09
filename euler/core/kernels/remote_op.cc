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

#include <unordered_map>
#include <unordered_set>

#include "euler/common/logging.h"
#include "euler/common/str_util.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/core/framework/tensor_util.h"
#include "euler/core/dag/dag.h"
#include "euler/proto/worker.pb.h"
#include "euler/client/client_manager.h"

namespace euler {

class Remote: public AsyncOpKernel {
 public:
  explicit Remote(const std::string& name) : AsyncOpKernel(name) { }
  void AsyncCompute(const DAGNodeProto& node_def,
                    OpKernelContext* ctx, DoneCallback callback) override;
};

int32_t MapToRemoteInputIdx(std::string inner_input_name) {
  std::vector<std::string> inner_input_info = Split(inner_input_name, ":");
  if (inner_input_info.size() == 2) {
    std::vector<std::string> src_name_id = Split(inner_input_info[0], ",");
    if (src_name_id.size() == 2 && src_name_id[0] == "REMOTE") {
      return atoi(inner_input_info[1].c_str());
    }
    return -1;
  } else {
    return -1;
  }
}

#define TRY_ENCODE(NAME, TENSOR) {                           \
  if (dedup.find(NAME) == dedup.end()) {                     \
    TensorProto* input_pb = request.mutable_inputs()->Add(); \
    input_pb->set_name(NAME);                                \
    Encode(*TENSOR, input_pb);                               \
    dedup.insert(NAME);                                      \
  }                                                          \
}

void Remote::AsyncCompute(const DAGNodeProto& node_def,
                          OpKernelContext* ctx, DoneCallback callback) {
  // get server shard id
  int32_t shard_id = node_def.shard_idx();
  // prepare request
  ExecuteRequest request;
  // 1. add inputs
  std::unordered_set<std::string> dedup;
  for (int32_t i = 0; i < node_def.inner_nodes_size(); ++i) {
    const DAGNodeProto& inner_node = node_def.inner_nodes(i);
    for (int32_t j = 0; j < inner_node.inputs_size(); ++j) {
      std::string inner_input_name = inner_node.inputs(j);
      Tensor* inner_input_tensor = nullptr;
      int32_t remote_input_idx = MapToRemoteInputIdx(inner_input_name);
      // input from outsize
      if (remote_input_idx != -1 &&
          ctx->tensor(node_def.inputs(remote_input_idx),
                      &inner_input_tensor).ok()) {
        TRY_ENCODE(inner_input_name, inner_input_tensor);
      } else if (remote_input_idx == -1 &&
                 ctx->tensor(inner_input_name, &inner_input_tensor).ok()) {
        TRY_ENCODE(inner_input_name, inner_input_tensor);
      }
    }
    // udf params
    for (int32_t j = 0; j < inner_node.udf_str_params_size(); ++j) {
      std::string udf_params_name = inner_node.udf_str_params(j);
      Tensor* udf_params_tensor = nullptr;
      ctx->tensor(udf_params_name, &udf_params_tensor);
      TRY_ENCODE(udf_params_name, udf_params_tensor);
    }
    for (int32_t j = 0; j < inner_node.udf_num_params_size(); ++j) {
      std::string udf_params_name = inner_node.udf_num_params(j);
      Tensor* udf_params_tensor = nullptr;
      ctx->tensor(udf_params_name, &udf_params_tensor);
      TRY_ENCODE(udf_params_name, udf_params_tensor);
    }
    // dnf value params
    for (int32_t j = 0; j < inner_node.dnf_size(); ++j) {
      std::vector<std::string> conj = Split(inner_node.dnf(j), ",");
      for (const std::string& term : conj) {
        std::vector<std::string> index_op_value = Split(term, " ");
        Tensor* v = nullptr;
        if (ctx->tensor(index_op_value[2], &v).ok()) {
          TRY_ENCODE(index_op_value[2], v);
        }
      }
    }
  }
  // 2. add dag
  DAGProto* dag = request.mutable_graph();
  dag->mutable_nodes()->CopyFrom(node_def.inner_nodes());
  // 3. add output
  std::unordered_map<std::string, std::string> io_name_2_ro_name;
  for (int32_t i = 0; i < node_def.output_list_size(); ++i) {
    request.add_outputs(node_def.output_list(i));
    io_name_2_ro_name[node_def.output_list(i)] =
        node_def.remote_output_list(i);
  }

  // call rpc
  ClientManager* client_manager = ClientManager::GetInstance();
  std::shared_ptr<RpcClient> rpc_client;
  if (client_manager != nullptr &&
      (rpc_client = client_manager->GetClient(shard_id)) != nullptr) {
    std::string method = "euler.EulerService/Execute";
    ExecuteReply* response = new ExecuteReply();
    auto done =
        [ctx, response, io_name_2_ro_name, callback] (const Status& status) {
      if (status.ok()) {
        // put output tensor in response into ctx
        for (int32_t i = 0; i < response->outputs_size(); ++i) {
          std::string io_name = response->outputs(i).name();
          ctx->Allocate(io_name_2_ro_name.at(io_name),
                        response->outputs(i));
        }
      } else {
        EULER_LOG(FATAL) << "rpc error: " << status.error_message();
      }
      delete response;
      callback();
    };
    rpc_client->IssueRpcCall(method, request, response, done);
  } else {
    EULER_LOG(ERROR) << "client manager error or shard id error";
  }
}

REGISTER_OP_KERNEL("REMOTE", Remote);

}  // namespace euler
