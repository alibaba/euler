/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

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

#ifndef EULER_CLIENT_IMPL_REGISTER_H_
#define EULER_CLIENT_IMPL_REGISTER_H_

#include <functional>
#include <memory>

namespace euler {
namespace client {

template <typename T>
using ImplCreator = std::function<std::unique_ptr<T>()>;

template <typename T>
class ImplFactory {
 public:
  static std::unique_ptr<T> New();
  static bool Register(ImplCreator<T> creator);

 private:
  static ImplCreator<T> &GetCreator();
};

template <typename T>
std::unique_ptr<T> ImplFactory<T>::New() {
  ImplCreator<T> &the_creator = GetCreator();
  if (the_creator) {
    return the_creator();
  } else {
    return nullptr;
  }
}

template <typename T>
bool ImplFactory<T>::Register(ImplCreator<T> creator) {
  GetCreator() = creator;
  return true;
}

template <typename T>
ImplCreator<T> &ImplFactory<T>::GetCreator() {
  static ImplCreator<T> the_creator;
  return the_creator;
}

#define REGISTER_IMPL(INTERFACE, IMPL)                   \
  static bool IMPL##_registerd __attribute__((unused)) ( \
      ImplFactory<INTERFACE>::Register([] {              \
        return std::unique_ptr<INTERFACE>(new IMPL);     \
      }));

}  // namespace client
}  // namespace euler


#endif  // EULER_CLIENT_IMPL_REGISTER_H_
