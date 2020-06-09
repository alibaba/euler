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

#include <stdlib.h>

#include <string>
#include <unordered_set>
#include <unordered_map>

#include "euler/core/kernels/common.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/udf.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/common/logging.h"
#include "euler/core/api/api.h"
#include "euler/core/graph/graph.h"

namespace euler {

namespace {

template <typename Result, typename T>
void Fill(const Result& result, size_t feature_idx,
          const std::string& name, DataType type, OpKernelContext* ctx) {
  std::string idx_output = OutputName(name, 2 * feature_idx);
  TensorShape idx_shape({result.size(), 2});
  Tensor* idx_t = nullptr;
  auto s = ctx->Allocate(idx_output, idx_shape, DataType::kInt32, &idx_t);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor: "
                     << idx_output << " failed!";
    return;
  }
  size_t offset = 0;
  for (size_t j = 0; j < result.size(); ++j) {
    idx_t->Raw<int32_t>()[j * 2] = offset;
    idx_t->Raw<int32_t>()[j * 2 + 1] = offset + result[j][0].size();
    offset += result[j][0].size();
  }

  std::string data_output = OutputName(name, 2 * feature_idx + 1);
  TensorShape data_shape({offset});
  Tensor* data_t = nullptr;
  s = ctx->Allocate(data_output, data_shape, type, &data_t);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Allocate output tensor: "
                     << idx_output << " failed!";
    return;
  }
  size_t base_addr = 0;
  for (size_t j = 0; j < result.size(); ++j) {
    std::copy(result[j][0].begin(), result[j][0].end(),
              data_t->Raw<T>() + base_addr);
    base_addr += result[j][0].size();
  }
}

}  // namespace

#define UDF_FILL(ID_VEC, FEATURE_FUNC0, FEATURE_FUNC1,             \
                 FEATURE_FUNC2, FEATURE_TYPE_FUNC) {               \
  if (node_def.udf_name() != "") {                                 \
    std::vector<NodesFeature> feature_vec;                         \
    for (int32_t i = 0; i < node_def.udf_str_params_size(); ++i) { \
      auto feature_type = FEATURE_TYPE_FUNC(*udf_fids[i]);         \
      switch (feature_type) {                                      \
        case euler::kSparse:                                       \
          feature_vec.push_back(                                   \
              NodesFeature(FEATURE_FUNC0(ID_VEC, {udf_fids[i]}))); \
          break;                                                   \
        case euler::kBinary:                                       \
          feature_vec.push_back(                                   \
              NodesFeature(FEATURE_FUNC1(ID_VEC, {udf_fids[i]}))); \
          break;                                                   \
        default:                                                   \
          feature_vec.push_back(                                   \
              NodesFeature(FEATURE_FUNC2(ID_VEC, {udf_fids[i]}))); \
      }                                                            \
    }                                                              \
    Udf* udf = nullptr;                                            \
    CreateUdf(node_def.udf_name(), &udf);                          \
    static_cast<ValuesUdf*>(udf)->                                 \
        Compute(                                                   \
            node_def.name(), feature_ids, udf_fids,                \
            udf_params, feature_vec, ctx);                         \
  }                                                                \
}

class GetFeatureOp : public OpKernel {
 public:
  explicit GetFeatureOp(const std::string& name): OpKernel(name) { }

  void Compute(const DAGNodeProto& node_def, OpKernelContext* ctx);
};

void GetFeatureOp::Compute(const DAGNodeProto& node_def, OpKernelContext* ctx) {
  if (node_def.inputs_size() < 2) {
    EULER_LOG(ERROR) << "Invalid argments for GetFeatureOp";
    return;
  }

  Tensor* ids_t = nullptr;
  auto s = ctx->tensor(node_def.inputs(0), &ids_t);
  if (!s.ok()) {
    EULER_LOG(ERROR) << "Retrieve argment node_ids/edge_ids failed!";
    return;
  }

  std::vector<std::string*> feature_ids(node_def.inputs_size() - 1);
  for (int32_t i = 1; i < node_def.inputs_size(); ++i) {
    Tensor* fid_t = nullptr;
    auto s = ctx->tensor(node_def.inputs(i), &fid_t);
    if (!s.ok() || fid_t->NumElements() == 0) {
      EULER_LOG(ERROR) << "Argment feature_ids must be specified!";
      return;
    }
    feature_ids[i - 1] = fid_t->Raw<std::string*>()[0];
  }

  std::unordered_set<std::string> feature_ids_set;
  for (size_t i = 0; i < feature_ids.size(); ++i) {
    feature_ids_set.insert(*(feature_ids[i]));
  }

  std::vector<std::string*> udf_fids(node_def.udf_str_params_size());
  for (int32_t i = 0; i < node_def.udf_str_params_size(); ++i) {
    Tensor* fid_t = nullptr;
    auto s = ctx->tensor(node_def.udf_str_params(i), &fid_t);
    if (!s.ok() || fid_t->NumElements() == 0) {
      EULER_LOG(ERROR) << "Argment feature_ids must be specified!";
      return;
    }
    std::string* u = fid_t->Raw<std::string*>()[0];
    if (feature_ids_set.find(*u) == feature_ids_set.end()) {
      EULER_LOG(ERROR) << "udf: " << node_def.udf_name() << " params error!";
      return;
    }
    udf_fids[i] = fid_t->Raw<std::string*>()[0];
  }

  std::unordered_set<std::string> udf_fids_set;
  for (size_t i = 0; i < udf_fids.size(); ++i) {
    udf_fids_set.insert(*(udf_fids[i]));
  }

  std::vector<Tensor*> udf_params(node_def.udf_num_params_size());
  for (int32_t i = 0; i < node_def.udf_num_params_size(); ++i) {
    Tensor* udf_p_t = nullptr;
    auto s = ctx->tensor(node_def.udf_num_params(i), &udf_p_t);
    if (!s.ok()) {
      EULER_LOG(ERROR) << "udf params tensor error!";
      return;
    }
    udf_params[i] = udf_p_t;
  }

  const auto& outname = node_def.name();
  auto& graph = Graph::Instance();
  auto shape = ids_t->Shape();
  if (shape.Size() == 1) {  // GetNodeFeature
    NodeIdVec node_ids(ids_t->NumElements());
    memcpy(node_ids.data(), ids_t->Raw<NodeId>(),
           node_ids.size() * sizeof(node_ids[0]));
    for (size_t i = 0; i < feature_ids.size(); ++i) {
      if (udf_fids_set.find(*feature_ids[i]) == udf_fids_set.end()) {
        auto feature_type = graph.GetNodeFeatureType(*feature_ids[i]);
        switch (feature_type) {
          case euler::kSparse:
            Fill<UInt64FeatureVec, uint64_t>(
                GetNodeUint64Feature(node_ids, {feature_ids[i]}), i,
                outname, DataType::kUInt64, ctx);
            break;
          case euler::kBinary:
            Fill<BinaryFatureVec, char>(
                GetNodeBinaryFeature(node_ids, {feature_ids[i]}), i,
                outname, DataType::kInt8, ctx);
            break;
          default:
            Fill<FloatFeatureVec, float>(
                GetNodeFloat32Feature(node_ids, {feature_ids[i]}), i,
                outname, DataType::kFloat, ctx);
        }
      }
    }
    UDF_FILL(node_ids, GetNodeUint64Feature, GetNodeBinaryFeature,
             GetNodeFloat32Feature, graph.GetNodeFeatureType);
  } else if (shape.Size() == 2) {  // GetEdgeFeature
    auto& dims = shape.Dims();
    if (dims[1] != 3) {
      EULER_LOG(ERROR) << "Invalid edge_ids specified, dim 1 with shape:"
                       << dims[1] << ", expected: 3";
      return;
    }
    EdgeIdVec edge_ids(dims[0]);
    auto data = ids_t->Raw<int64_t>();
    for (auto& edge_id : edge_ids) {
      edge_id = std::make_tuple(data[0], data[1], data[2]);
      data += 3;
    }

    for (size_t i = 0; i < feature_ids.size(); ++i) {
      if (udf_fids_set.find(*feature_ids[i]) == udf_fids_set.end()) {
        auto feature_type = graph.GetEdgeFeatureType(*feature_ids[i]);
        switch (feature_type) {
          case euler::kSparse:
            Fill<UInt64FeatureVec, uint64_t>(
                GetEdgeUint64Feature(edge_ids, {feature_ids[i]}), i,
                outname, DataType::kUInt64, ctx);
            break;
          case euler::kBinary:
            Fill<BinaryFatureVec, char>(
                GetEdgeBinaryFeature(edge_ids, {feature_ids[i]}), i,
                outname, DataType::kInt8, ctx);
            break;
          default:
            Fill<FloatFeatureVec, float>(
                GetEdgeFloat32Feature(edge_ids, {feature_ids[i]}), i,
                outname, DataType::kFloat, ctx);
        }
      }
    }
    UDF_FILL(edge_ids, GetEdgeUint64Feature, GetEdgeBinaryFeature,
             GetEdgeFloat32Feature, graph.GetEdgeFeatureType);
  } else {  // Invalid ids shape
    EULER_LOG(ERROR) << "Invalid node_ids/edge_ids specified!";
  }
}

REGISTER_OP_KERNEL("API_GET_P", GetFeatureOp);

}  // namespace euler
