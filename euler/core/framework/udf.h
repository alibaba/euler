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


#ifndef EULER_CORE_FRAMEWORK_UDF_H_
#define EULER_CORE_FRAMEWORK_UDF_H_

#include <vector>
#include <string>
#include <unordered_map>

#include "euler/common/signal.h"
#include "euler/common/status.h"
#include "euler/common/logging.h"
#include "euler/core/graph/graph.h"
#include "euler/core/api/api.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/op_kernel.h"

namespace euler {

class Udf {
 public:
  explicit Udf(const std::string& name): name_(name) {}
  virtual ~Udf() {}
 protected:
  std::string name_;
};

class UdfCache {
 public:
  ~UdfCache() {
    MutexLock lock(&mu_);
    for (auto& item : cache_) {
      delete item.second;
    }
    cache_.clear();
  }

  void Get(const std::string& udf_name, Udf** udf) {
    *udf = nullptr;
    MutexLock lock(&mu_);
    auto it = cache_.find(udf_name);
    if (it != cache_.end()) {
      *udf = it->second;
    }
  }

  void Cache(const std::string& udf_name, Udf* udf) {
    MutexLock lock(&mu_);
    cache_.insert({udf_name, udf});
  }

 private:
  Mutex mu_;
  std::unordered_map<std::string, Udf*> cache_;
};

Status CreateUdf(const std::string& name, Udf** udf);

struct NodesFeature {
  FeatureType feature_type_;  // kSparse, kDense, kBinary
  UInt64FeatureVec uv_;
  FloatFeatureVec fv_;
  BinaryFatureVec bv_;

  explicit NodesFeature(const UInt64FeatureVec& uv) {
    feature_type_ = kSparse;
    uv_ = uv;
  }

  explicit NodesFeature(const FloatFeatureVec& fv) {
    feature_type_ = kDense;
    fv_ = fv;
  }

  explicit NodesFeature(const BinaryFatureVec& bv) {
    feature_type_ = kBinary;
    bv_ = bv;
  }
};

class ValuesUdf: public Udf {
 public:
  explicit ValuesUdf(const std::string& name): Udf(name) {}
  virtual ~ValuesUdf() {}
  virtual void Compute(
      const std::string& node_name,
      const std::vector<std::string*>& fids,
      const std::vector<std::string*>& udf_fids,
      const std::vector<Tensor*>& udf_params,
      const std::vector<NodesFeature>& feature_vec,
      OpKernelContext* ctx);

 protected:
  virtual std::vector<NodesFeature> Process(
      const std::vector<std::string*>& udf_fids,
      const std::vector<Tensor*>& udf_params,
      const std::vector<NodesFeature>& feature_vec) = 0;
};

#define REGISTER_UDF(name, cls)                    \
  REGISTER_UDF_UNIQ_HELPER(__COUNTER__, name, cls) \

#define REGISTER_UDF_UNIQ_HELPER(counter, name, cls) \
  REGISTER_UDF_UNIQ(counter, name, cls)              \

#define REGISTER_UDF_UNIQ(counter, name, cls)           \
  static ::euler::UdfRegistar                           \
      registrar__##counter##__obj(                      \
          name,                                         \
          [] (const std::string& udf) -> euler::Udf* {  \
            Udf* udf_ptr = new cls(udf);                \
            return udf_ptr;                             \
          });                                           \

class UdfRegistar {
 public:
  typedef Udf* (*Factory)(const std::string& name);
  UdfRegistar(const std::string& name, Factory factory) {
    EULER_LOG(INFO) << "udf: " << name << " upload";
    Register(name, factory);
  }

 private:
  void Register(const std::string& name, Factory factory);
};

typedef std::unordered_map<std::string,
                           UdfRegistar::Factory> UdfRegistry;

UdfRegistry* GlobalUdfRegistry();
}  // namespace euler

#endif  // EULER_CORE_FRAMEWORK_UDF_H_
