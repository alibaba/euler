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
#include <algorithm>
#include <queue>

#include "euler/common/logging.h"
#include "euler/core/framework/op_kernel.h"
#include "euler/core/framework/dag_node.pb.h"
#include "euler/core/framework/tensor.h"
#include "euler/common/str_util.h"

namespace euler {

namespace {
bool IdBigger(const uint64_t* a, const uint64_t* b, int32_t len) {
  int32_t i = 0; for (; i < len && *(a + i) == *(b + i); ++i);
  return (i < len && *(a + i) > *(b + i));
}

bool IdSmaller(const uint64_t* a, const uint64_t* b, int32_t len) {
  int32_t i = 0; for (; i < len && *(a + i) == *(b + i); ++i);
  return (i < len && *(a + i) < *(b + i));
}
}  // namespace

class MinHeapIdComparison {
 public:
  bool operator()(const std::pair<std::vector<uint64_t>, int32_t>& a,
                  const std::pair<std::vector<uint64_t>, int32_t>& b) {
    return IdBigger(a.first.data(), b.first.data(),
                    static_cast<int32_t>(a.first.size()));
  }
};

class MinHeapWeightComparison {
 public:
  bool operator()(const std::pair<float, int32_t>& a,
                  const std::pair<float, int32_t>& b) {
    return a.first > b.first;
  }
};

class MaxHeapIdComparison {
 public:
  bool operator()(const std::pair<std::vector<uint64_t>, int32_t>& a,
                  const std::pair<std::vector<uint64_t>, int32_t>& b) {
    return IdSmaller(a.first.data(), b.first.data(),
                     static_cast<int32_t>(a.first.size()));
  }
};

class MaxHeapWeightComparison {
 public:
  bool operator()(const std::pair<float, int32_t>& a,
                  const std::pair<float, int32_t>& b) {
    return a.first < b.first;
  }
};

class PostProcess: public OpKernel {
 public:
  explicit PostProcess(const std::string& name) : OpKernel(name) {}
  void Compute(const DAGNodeProto& node_def,
               OpKernelContext* ctx) override;
 private:
  void MergeSort(Tensor* ids, int32_t offset, Tensor* weights,
                 bool order_with_weight, bool asc,
                 Tensor* output_ids, Tensor* output_weights);
};

#define INIT_WEIGHT_HEAP(HEAP) {                                     \
  for (int32_t seg_id = 0; seg_id <                                  \
       static_cast<int32_t>(ptr_list.size()); ++seg_id) {            \
    int32_t ptr = ptr_list[seg_id];                                  \
    std::pair<float, int32_t> p(weights->Raw<float>()[ptr], seg_id); \
    HEAP.push(p);                                                    \
  }                                                                  \
}

#define UPDATE_WEIGHT_HEAP(HEAP) {                                   \
  if (ptr_list[seg_id] < ends_pos[seg_id]) {                         \
    int32_t ptr = ptr_list[seg_id];                                  \
    std::pair<float, int32_t> p(weights->Raw<float>()[ptr], seg_id); \
    HEAP.push(p);                                                    \
  }                                                                  \
}

#define INIT_ID_HEAP(HEAP) {                                   \
  for (int32_t seg_id = 0; seg_id <                            \
       static_cast<int32_t>(ptr_list.size()); ++seg_id) {      \
    int32_t ptr = ptr_list[seg_id];                            \
    std::vector<uint64_t> id(offset == 3 ? 2 : 1);             \
    std::copy(ids->Raw<uint64_t>() + ptr * offset,             \
              ids->Raw<uint64_t>() + ptr * offset + id.size(), \
              id.begin());                                     \
    std::pair<std::vector<uint64_t>, int32_t> p(id, seg_id);   \
    HEAP.push(p);                                              \
  }                                                            \
}

#define UPDATE_ID_HEAP(HEAP) {                                 \
  if (ptr_list[seg_id] < ends_pos[seg_id]) {                   \
    int32_t ptr = ptr_list[seg_id];                            \
    std::vector<uint64_t> id(offset == 3 ? 2 : 1);             \
    std::copy(ids->Raw<uint64_t>() + ptr * offset,             \
              ids->Raw<uint64_t>() + ptr * offset + id.size(), \
              id.begin());                                     \
    std::pair<std::vector<uint64_t>, int32_t> p(id, seg_id);   \
    HEAP.push(p);                                              \
  }                                                            \
}

#define OUTPUT(HEAP, PAIR) {                          \
  seg_id = PAIR.second;                               \
  int32_t ptr = ptr_list[seg_id]++;                   \
  for (int32_t i = 0; i < offset; ++i) {              \
    output_ids->Raw<uint64_t>()[o_idx * offset + i] = \
        ids->Raw<uint64_t>()[ptr * offset + i];       \
  }                                                   \
  if (output_weights != nullptr) {                    \
    output_weights->Raw<float>()[o_idx] =             \
        weights->Raw<float>()[ptr];                   \
  }                                                   \
  ++o_idx;                                            \
}

void PostProcess::MergeSort(Tensor* ids, int32_t offset, Tensor* weights,
                            bool order_with_weight, bool asc,
                            Tensor* output_ids, Tensor* output_weights) {
  std::vector<int32_t> ptr_list;
  std::vector<int32_t> ends_pos;
  ptr_list.push_back(0);
  if (order_with_weight) {
    if (asc) {
      std::priority_queue<std::pair<float, int32_t>,
          std::vector<std::pair<float, int32_t>>,
          MinHeapWeightComparison> min_heap;
      for (int32_t i = 0; i < weights->NumElements() - 1; ++i) {
        if (weights->Raw<float>()[i] > weights->Raw<float>()[i + 1]) {
          ends_pos.push_back(i + 1);
          ptr_list.push_back(i + 1);
        }
      }
      ends_pos.push_back(weights->NumElements());
      INIT_WEIGHT_HEAP(min_heap);
      int32_t o_idx = 0;
      while (!min_heap.empty() && o_idx < output_ids->NumElements() / offset) {
        std::pair<float, int32_t> p = min_heap.top();
        min_heap.pop();
        int32_t seg_id = 0;
        OUTPUT(min_heap, p);
        UPDATE_WEIGHT_HEAP(min_heap);
      }
    } else {
      std::priority_queue<std::pair<float, int32_t>,
          std::vector<std::pair<float, int32_t>>,
          MaxHeapWeightComparison> max_heap;
      for (int32_t i = 0; i < weights->NumElements() - 1; ++i) {
        if (weights->Raw<float>()[i] < weights->Raw<float>()[i + 1]) {
          ends_pos.push_back(i + 1);
          ptr_list.push_back(i + 1);
        }
      }
      ends_pos.push_back(weights->NumElements());
      INIT_WEIGHT_HEAP(max_heap);
      int32_t o_idx = 0;
      while (!max_heap.empty() && o_idx < output_ids->NumElements() / offset) {
        std::pair<float, int32_t> p = max_heap.top();
        max_heap.pop();
        int32_t seg_id = 0;
        OUTPUT(max_heap, p);
        UPDATE_WEIGHT_HEAP(max_heap);
      }
    }
  } else {
    if (asc) {
      std::priority_queue<std::pair<std::vector<uint64_t>, int32_t>,
          std::vector<std::pair<std::vector<uint64_t>, int32_t>>,
          MinHeapIdComparison> min_heap;
      for (int32_t i = 0; i < ids->NumElements() - offset; i += offset) {
        if (IdBigger(ids->Raw<uint64_t>() + i * offset,
                     ids->Raw<uint64_t>() + (i + 1) * offset,
                     offset == 3 ? 2 : 1)) {
          ends_pos.push_back(i + 1);
          ptr_list.push_back(i + 1);
        }
      }
      ends_pos.push_back(weights->NumElements());
      INIT_ID_HEAP(min_heap);
      int32_t o_idx = 0;
      while (!min_heap.empty() && o_idx < output_ids->NumElements() / offset) {
        std::pair<std::vector<uint64_t>, int32_t> p = min_heap.top();
        min_heap.pop();
        int32_t seg_id = 0;
        OUTPUT(min_heap, p);
        UPDATE_ID_HEAP(min_heap);
      }
    } else {
      std::priority_queue<std::pair<std::vector<uint64_t>, int32_t>,
          std::vector<std::pair<std::vector<uint64_t>, int32_t>>,
          MaxHeapIdComparison> max_heap;
      for (int32_t i = 0; i < ids->NumElements() - offset; i += offset) {
        if (IdSmaller(ids->Raw<uint64_t>() + i * offset,
                      ids->Raw<uint64_t>() + (i + 1) * offset,
                      offset == 3 ? 2 : 1)) {
          ends_pos.push_back(i + 1);
          ptr_list.push_back(i + 1);
        }
      }
      ends_pos.push_back(weights->NumElements());
      INIT_ID_HEAP(max_heap);
      int32_t o_idx = 0;
      while (!max_heap.empty() && o_idx < output_ids->NumElements() / offset) {
        std::pair<std::vector<uint64_t>, int32_t> p = max_heap.top();
        max_heap.pop();
        int32_t seg_id = 0;
        OUTPUT(max_heap, p);
        UPDATE_ID_HEAP(max_heap);
      }
    }
  }
}

#define ALLOCATE_OUTPUT(ARG) {                                 \
  if (offset == 1) {                                           \
    TensorShape shape({static_cast<size_t>(ARG)});             \
    ctx->Allocate(OutputName(node_def, 0),                     \
                  shape, DataType::kUInt64, &output_ids_t);    \
  } else {                                                     \
    TensorShape shape({static_cast<size_t>(ARG),               \
                      static_cast<size_t>(offset)});           \
    ctx->Allocate(OutputName(node_def, 0),                     \
                  shape, DataType::kUInt64, &output_ids_t);    \
  }                                                            \
  if (input_weights_t != nullptr) {                            \
    TensorShape shape({static_cast<size_t>(ARG)});             \
    ctx->Allocate(OutputName(node_def, 1),                     \
                  shape, DataType::kFloat, &output_weights_t); \
  }                                                            \
}

#define PROSESS_ORDER_BY() {                                        \
  if (pp_cmds[0][1] == "id" && pp_cmds[0][2] == "asc") {            \
    MergeSort(input_ids_t, offset, input_weights_t, false, true,    \
              output_ids_t, output_weights_t);                      \
  } else if (pp_cmds[0][1] == "id" && pp_cmds[0][2] == "desc") {    \
    MergeSort(input_ids_t, offset, input_weights_t, false, false,   \
              output_ids_t, output_weights_t);                      \
  } else if (pp_cmds[0][1] == "weight" && pp_cmds[0][2] == "asc") { \
    MergeSort(input_ids_t, offset, input_weights_t, true, true,     \
              output_ids_t, output_weights_t);                      \
  } else {                                                          \
    MergeSort(input_ids_t, offset, input_weights_t, true, false,    \
              output_ids_t, output_weights_t);                      \
  }                                                                 \
}

void PostProcess::Compute(const DAGNodeProto& node_def,
                          OpKernelContext* ctx) {
  // edge_id shape=[n, 3]
  // node_id shape=[n]
  // weight shape=[n]
  std::vector<Tensor*> inputs;  // id, weight
  for (int32_t i = 0; i < node_def.inputs_size(); ++i) {
    Tensor* t = nullptr;
    ctx->tensor(node_def.inputs(i), &t);
    inputs.push_back(t);
  }

  Tensor* input_ids_t = nullptr;
  Tensor* input_weights_t = nullptr;
  ctx->tensor(node_def.inputs(0), &input_ids_t);
  if (node_def.inputs_size() == 2) {
    ctx->tensor(node_def.inputs(1), &input_weights_t);
  }
  int32_t offset = input_ids_t->Shape().Dims().size() == 2 ? 3 : 1;

  std::vector<std::vector<std::string>> pp_cmds;
  for (const std::string& pp : node_def.post_process()) {
    pp_cmds.push_back(Split(pp, " "));
  }

  Tensor* output_ids_t = nullptr;
  Tensor* output_weights_t = nullptr;
  if (pp_cmds.size() == 1) {
    if (pp_cmds[0][0] == "limit") {  // limit
      int32_t arg = atoi(pp_cmds[0][1].c_str());
      ALLOCATE_OUTPUT(arg);
      std::copy(input_ids_t->Raw<uint64_t>(),
                input_ids_t->Raw<uint64_t>() +
                output_ids_t->NumElements(),
                output_ids_t->Raw<uint64_t>());
      if (output_weights_t != nullptr) {
        std::copy(input_weights_t->Raw<float>(),
                  input_weights_t->Raw<float>() +
                  output_weights_t->NumElements(),
                  output_weights_t->Raw<float>());
      }
    } else {  // order_by
      ALLOCATE_OUTPUT(input_ids_t->Shape().Dims()[0]);
      PROSESS_ORDER_BY();
    }
  } else {  // order_by, limit
    int32_t arg = atoi(pp_cmds[1][1].c_str());
    ALLOCATE_OUTPUT(arg);
    PROSESS_ORDER_BY();
  }
}

REGISTER_OP_KERNEL("POST_PROCESS", PostProcess);

}  // namespace euler
