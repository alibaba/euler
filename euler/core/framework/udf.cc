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

#include "euler/core/framework/udf.h"

#include <algorithm>
#include <vector>
#include <unordered_map>
#include <string>

#include "euler/common/logging.h"
#include "euler/common/str_util.h"

namespace euler {

namespace {

template <typename Result, typename T>
void Output(const Result& result, std::string idx_output,
            std::string data_output, DataType type, OpKernelContext* ctx) {
  TensorShape idx_shape({result.size(), 2});
  Tensor* idx_t = nullptr;
  auto s = ctx->Allocate(idx_output, idx_shape, DataType::kInt32, &idx_t);
  if (!s.ok()) {
    EULER_LOG(FATAL) << "tensor " << idx_output << " allocate error";
  }
  size_t offset = 0;
  for (size_t i = 0; i < result.size(); ++i) {
    idx_t->Raw<int32_t>()[i * 2] = offset;
    idx_t->Raw<int32_t>()[i * 2 + 1] = offset + result[i][0].size();
    offset += result[i][0].size();
  }
  TensorShape data_shape({offset});
  Tensor* data_t = nullptr;
  s = ctx->Allocate(data_output, data_shape, type, &data_t);
  if (!s.ok()) {
    EULER_LOG(FATAL) << "tensor " << idx_output << " allocate error";
  }
  size_t base_addr = 0;
  for (size_t i = 0; i < result.size(); ++i) {
    std::copy(result[i][0].begin(), result[i][0].end(),
              data_t->Raw<T>() + base_addr);
    base_addr += result[i][0].size();
  }
}

}  // namespace

void ValuesUdf::Compute(
    const std::string& node_name,
    const std::vector<std::string*>& fids,
    const std::vector<std::string*>& udf_fids,
    const std::vector<Tensor*>& udf_params,
    const std::vector<NodesFeature>& feature_vec,
    OpKernelContext* ctx) {
  std::vector<NodesFeature> processed_fv =
      Process(udf_fids, udf_params, feature_vec);
  std::unordered_map<std::string, int32_t> fid2oid;
  for (std::string* ufid : udf_fids) {
    fid2oid[*ufid] = 0;
  }
  for (size_t i = 0; i < fids.size(); ++i) {
    auto it = fid2oid.find(*fids[i]);
    if (it != fid2oid.end()) {
      it->second = i;
    }
  }
  // output to ctx
  for (size_t i = 0; i < processed_fv.size(); ++i) {
    int32_t oid = fid2oid[*(udf_fids[i])];
    std::string idx_output = OutputName(node_name, oid * 2);
    std::string data_output = OutputName(node_name, oid * 2 + 1);
    switch (processed_fv[i].feature_type_) {
      case euler::kSparse:
        Output<UInt64FeatureVec, uint64_t>(
            processed_fv[i].uv_, idx_output, data_output,
            DataType::kUInt64, ctx);
        break;
      case euler::kBinary:
        Output<BinaryFatureVec, char>(
            processed_fv[i].bv_, idx_output, data_output,
            DataType::kInt8, ctx);
        break;
      default:
        Output<FloatFeatureVec, float>(
            processed_fv[i].fv_, idx_output, data_output,
            DataType::kFloat, ctx);
    }
  }
}

UdfRegistry* GlobalUdfRegistry() {
  static UdfRegistry* registry = new UdfRegistry;
  return registry;
}

void UdfRegistar::Register(const std::string& name, Factory factory) {
  if (!GlobalUdfRegistry()->insert({name, factory}).second) {
    EULER_LOG(FATAL) << "Register ValuesUdf '" << name << "' failed!";
  }
}

Status CreateUdf(const std::string& name, Udf** udf) {
  static UdfCache instance;
  instance.Get(name, udf);
  if (*udf == nullptr) {
    auto& registry = *GlobalUdfRegistry();
    auto it = registry.find(name);
    if (it == registry.end()) {
      return Status::NotFound("No OpKernel '", name, "' registered");
    }
    *udf = it->second(name);
    instance.Cache(name, *udf);
  }
  return Status::OK();
}

}  // namespace euler
