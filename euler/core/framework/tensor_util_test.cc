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

#include "euler/core/framework/tensor_util.h"

#include <string>

#include "gtest/gtest.h"

#include "euler/core/framework/op_kernel.h"
#include "euler/common/str_util.h"

namespace euler {

TEST(TensorUtilTest, All) {
  OpKernelContext context;
  Tensor* tensor = nullptr;
  ASSERT_TRUE(context.Allocate("tensor", TensorShape({2, 4}),
                               DataType::kString, &tensor).ok());

  auto ptr = tensor->Raw<std::string*>();
  std::string buffer;
  for (int i = 0; i < tensor->NumElements(); ++i) {
    **ptr = ToString("TensorUtilTest", i);
    uint32_t len = (*ptr)->size();
    buffer.append(reinterpret_cast<char*>(&len), sizeof(len));
    buffer.append(**ptr);
    ++ptr;
  }

  TensorProto proto;
  ASSERT_TRUE(Encode(*tensor, &proto).ok());
  ASSERT_EQ(buffer, proto.tensor_content());

  Tensor* target = nullptr;
  ASSERT_TRUE(context.Allocate("target", TensorShape({2, 4}),
                               DataType::kString, &target).ok());
  ASSERT_TRUE(Decode(proto, target).ok());

  ptr = tensor->Raw<std::string*>();
  auto tptr = target->Raw<std::string*>();
  for (int i = 0; i < tensor->NumElements(); ++i) {
    ASSERT_EQ(**ptr, **tptr);
    ++ptr;
    ++tptr;
  }
}

}  // namespace euler
