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

#include <vector>
#include <string>
#include <atomic>

#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/common/str_util.h"
#include "euler/common/logging.h"
#include "euler/common/env.h"
#include "euler/common/signal.h"
#include "euler/common/data_types.h"

namespace euler {
class DataMerge: public OpKernel {
 public:
  explicit DataMerge(const std::string& name) : OpKernel(name) {
    env_ = Env::Default();
    tp_ = env_->StartThreadPool("merge_thread_pool", 8);
  }
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;

 private:
  Env* env_;
  ThreadPool* tp_;
};

#define CONCURRENT_MERGE(DATA_TYPE) {                            \
  std::atomic<int32_t> cnt;                                      \
  cnt = static_cast<int32_t>(datas.size());                      \
  Signal sig;                                                    \
  for (size_t i = 0; i < datas.size(); ++i) {                    \
    Tensor* data_t = datas[i];                                   \
    Tensor* data_idx_t = data_idxs[i];                           \
    Tensor* merge_idx_t = merge_idxs[i];                         \
    tp_->Schedule([data_t, data_idx_t, merge_idx_t, data_result, \
                   &merge_addr, &cnt, &sig]() {                  \
      for (int32_t j = 0; j < merge_idx_t->NumElements(); ++j) { \
        int32_t merge_idx = merge_idx_t->Raw<int32_t>()[j];      \
        int32_t seg_begin = data_idx_t->Raw<int32_t>()[j * 2];   \
        int32_t seg_end = data_idx_t->Raw<int32_t>()[j * 2 + 1]; \
        std::copy(data_t->Raw<DATA_TYPE>() + seg_begin,          \
                  data_t->Raw<DATA_TYPE>() + seg_end,            \
                  data_result->Raw<DATA_TYPE>() +                \
                  merge_addr[merge_idx]);                        \
      }                                                          \
      if (--cnt == 0) sig.Notify();                              \
    });                                                          \
  }                                                              \
  sig.Wait();                                                    \
}

void DataMerge::Compute(const DAGNodeProto& node_def,
                        OpKernelContext* ctx) {
  /* get input tensor and merge index */
  std::vector<Tensor*> datas;
  datas.reserve(node_def.inputs_size() / 3);
  std::vector<Tensor*> data_idxs;
  data_idxs.reserve(node_def.inputs_size() / 3);
  std::vector<Tensor*> merge_idxs;
  merge_idxs.reserve(node_def.inputs_size() / 3);
  int32_t ids_num = 0;
  int32_t datas_num = 0;
  DataType data_type = kFloat;
  std::vector<size_t> shape;
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    Tensor* t = nullptr;
    ctx->tensor(node_def.inputs(i), &t);
    if (i % 3 == 0) {  // data
      datas.push_back(t);
      datas_num += t->NumElements();
      data_type = t->Type();
      shape = t->Shape().Dims();
    } else if (i % 3 == 1) {  // idx
      data_idxs.push_back(t);
      ids_num += t->Shape().Dims()[0];
    } else {  // merge idx
      merge_idxs.push_back(t);
    }
  }

  // calculate shape
  int32_t other_dim = 1;
  for (size_t i = 1; i < shape.size(); ++i) {
    other_dim *= shape[i];
  }
  shape[0] = datas_num / other_dim;
  TensorShape data_shape(shape);

  // allocate memory
  std::vector<int32_t> merge_addr(ids_num);
  for (size_t i = 0; i < datas.size(); ++i) {
    Tensor* data_idx_t = data_idxs[i];
    Tensor* merge_idx_t = merge_idxs[i];
    for (int32_t j = 0; j < merge_idx_t->NumElements(); ++j) {
      int32_t merge_idx = merge_idx_t->Raw<int32_t>()[j];
      int32_t seg_begin = data_idx_t->Raw<int32_t>()[j * 2] * other_dim;
      int32_t seg_end = data_idx_t->Raw<int32_t>()[j * 2 + 1] * other_dim;
      merge_addr[merge_idx] = seg_end - seg_begin;
    }
  }
  int32_t pre_size = 0, cur_size = 0;
  for (int32_t i = 0; i < ids_num; ++i) {
    cur_size = merge_addr[i];
    merge_addr[i] = pre_size;
    pre_size += cur_size;
  }
  Tensor* data_result = nullptr;

  std::string output_name = OutputName(node_def, 0);
  ctx->Allocate(output_name, data_shape, data_type, &data_result);
  if (data_type == DataType::kUInt64) {
    CONCURRENT_MERGE(uint64_t);
  } else if (data_type == DataType::kFloat) {
    CONCURRENT_MERGE(float);
  } else if (data_type == DataType::kInt8) {
    CONCURRENT_MERGE(char);
  } else if (data_type == DataType::kInt32) {
    CONCURRENT_MERGE(int32_t);
  } else {
    EULER_LOG(ERROR) << "error data type";
  }
}

class GPDataMerge: public OpKernel {
 public:
  explicit GPDataMerge(const std::string& name) : OpKernel(name) { }
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
};

struct MergeInfoValue {
  int32_t segment_size_;
  int32_t shard_idx_;  // the segment belong to which shard
  int32_t merge_addr_;
  MergeInfoValue(int32_t seg_size,
                 int32_t shard_idx,
                 int32_t merge_addr) {
    segment_size_ = seg_size;
    shard_idx_ = shard_idx;
    merge_addr_ = merge_addr;
  }
  MergeInfoValue() {}
};

#define REPLACE(DATA, DEFAULT_VALUE) {                      \
  if (merge_info.find(merge_idx) == merge_info.end()) {     \
    merge_info[merge_idx] = MergeInfoValue(                 \
        segment_size, shard_idx, 0);                        \
  } else if (merge_info.at(merge_idx).segment_size_ == 0) { \
    merge_info.at(merge_idx).segment_size_ = segment_size;  \
    merge_info.at(merge_idx).shard_idx_ = shard_idx;        \
  } else if (merge_info.at(merge_idx).segment_size_ ==      \
             segment_size) {                                \
    if (DATA != DEFAULT_VALUE) {                            \
      merge_info.at(merge_idx).shard_idx_ = shard_idx;      \
    }                                                       \
  } else if (merge_info.at(merge_idx).segment_size_ > 0 &&  \
             segment_size > 0) {                            \
    EULER_LOG(FATAL) << "data error";                       \
  }                                                         \
}

#define MERGE(DATA_TYPE) {                               \
  std::copy(datas[i]->Raw<DATA_TYPE>() + seg_begin,      \
            datas[i]->Raw<DATA_TYPE>() + seg_end,        \
            data_result->Raw<DATA_TYPE>() + merge_addr); \
}

void GPDataMerge::Compute(const DAGNodeProto& node_def,
                          OpKernelContext* ctx) {
  /* get input tensor and merge index */
  std::vector<Tensor*> datas;
  datas.reserve(node_def.inputs_size() / 3);
  std::vector<Tensor*> data_idxs;
  data_idxs.reserve(node_def.inputs_size() / 3);
  std::vector<Tensor*> merge_idxs;
  merge_idxs.reserve(node_def.inputs_size() / 3);
  DataType data_type = kFloat;
  std::vector<size_t> shape;
  size_t bucket_num = 0;
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    Tensor* t = nullptr;
    ctx->tensor(node_def.inputs(i), &t);
    if (i % 3 == 0) {  // data
      datas.push_back(t);
      data_type = t->Type();
      shape = t->Shape().Dims();
    } else if (i % 3 == 1) {  // idx
      data_idxs.push_back(t);
      bucket_num += t->Shape().Dims()[0];
    } else {  // merge idx
      merge_idxs.push_back(t);
    }
  }
  size_t shard_num = datas.size();

  /* get merge info
   * merge info: map
   *
   * key=merge_idx
   * value={segment_size, shard_idx, merge_addr}
   */
  std::unordered_map<int32_t, MergeInfoValue> merge_info(bucket_num);
  for (size_t i = 0; i < shard_num; ++i) {
    for (int32_t j = 0; j < merge_idxs[i]->NumElements(); ++j) {
      int32_t merge_idx = merge_idxs[i]->Raw<int32_t>()[j];
      int32_t segment_size = data_idxs[i]->Raw<int32_t>()[j * 2 + 1]
          - data_idxs[i]->Raw<int32_t>()[j * 2];
      int32_t shard_idx = static_cast<int32_t>(i);
      if (data_type == DataType::kUInt64) {
        REPLACE(datas[i]->Raw<uint64_t>()[
                data_idxs[i]->Raw<int32_t>()[j * 2]],
                euler::common::DEFAULT_UINT64);
      } else if (data_type == DataType::kFloat) {
        REPLACE(datas[i]->Raw<float>()[
                data_idxs[i]->Raw<int32_t>()[j * 2]],
                euler::common::DEFAULT_FLOAT);
      } else if (data_type == DataType::kInt8) {
        REPLACE(datas[i]->Raw<char>()[
                data_idxs[i]->Raw<int32_t>()[j * 2]],
                euler::common::DEFAULT_CHAR);
      } else if (data_type == DataType::kInt32) {
        REPLACE(datas[i]->Raw<int32_t>()[
                data_idxs[i]->Raw<int32_t>()[j * 2]],
                euler::common::DEFAULT_INT32);
      } else {
        EULER_LOG(ERROR) << "error data type";
      }
    }
  }
  int32_t ids_num = merge_info.size();
  int32_t datas_num = 0;
  for (int32_t i = 0; i < ids_num; ++i) {
    merge_info.at(i).merge_addr_ = datas_num;
    datas_num += merge_info.at(i).segment_size_;
  }

  // calculate shape
  int32_t other_dim = 1;
  for (size_t i = 1; i < shape.size(); ++i) {
    other_dim *= shape[i];
  }
  shape[0] = datas_num / other_dim;
  TensorShape data_shape(shape);

  /* merge data into data_result */
  Tensor* data_result = nullptr;
  std::string output_name = OutputName(node_def, 0);
  ctx->Allocate(output_name, data_shape, data_type, &data_result);
  for (size_t i = 0; i < shard_num; ++i) {
    /* generate merge index for each shard */
    std::string output_name = OutputName(node_def, i + 1);
    Tensor* merge_idx_tensor = nullptr;
    ctx->Allocate(output_name, datas[i]->Shape(),
        DataType::kInt32, &merge_idx_tensor);
    for (int32_t j = 0; j < merge_idxs[i]->NumElements(); ++j) {
      int32_t merge_idx = merge_idxs[i]->Raw<int32_t>()[j];
      int32_t seg_begin = data_idxs[i]->Raw<int32_t>()[j * 2];
      int32_t seg_end = data_idxs[i]->Raw<int32_t>()[j * 2 + 1];
      int32_t merge_addr = merge_info.at(merge_idx).merge_addr_;
      if (merge_info.at(merge_idx).shard_idx_ ==
          static_cast<int32_t>(i)) {
        if (data_type == DataType::kUInt64) {
          MERGE(uint64_t);
        } else if (data_type == DataType::kFloat) {
          MERGE(float);
        } else if (data_type == DataType::kInt8) {
          MERGE(char);
        } else if (data_type == DataType::kInt32) {
          MERGE(int32_t);
        } else {
          EULER_LOG(ERROR) << "error data type";
        }
      }
      for (int32_t k = seg_begin, cnt = 0; k < seg_end; ++k, ++cnt) {
        merge_idx_tensor->Raw<int32_t>()[k] = merge_addr + cnt;
      }
    }
  }
}

REGISTER_OP_KERNEL("DATA_MERGE", DataMerge);
REGISTER_OP_KERNEL("GP_DATA_MERGE", GPDataMerge);

}  // namespace euler
