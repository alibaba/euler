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


#ifndef EULER_COMMON_SINGLETON_H_
#define EULER_COMMON_SINGLETON_H_

#include <utility>
#include <memory>

#include "euler/common/mutex.h"

namespace euler {

template<typename T>
class SingletonBase {
 public:
  friend class std::unique_ptr<T>;

  static T *Get() {
    return Instance();
  }

  template<typename... Args>
  static T *Instance(Args &&... args) {
    if (instance_ == nullptr) {
      MutexLock lock(&mu_);
      if (instance_ == nullptr) {
        instance_.reset(new T(std::forward<Args>(args)...));
      }
    }
    return instance_.get();
  }

 protected:
  SingletonBase() {}
  virtual ~SingletonBase() {}

 private:
  static Mutex mu_;
  static std::unique_ptr<T> instance_;
};

template<typename T> Mutex SingletonBase<T>::mu_;
template<typename T> std::unique_ptr<T> SingletonBase<T>::instance_;

template <typename T>
class Singleton : public SingletonBase<T>{ };

}  // namespace euler

#endif  // EULER_COMMON_SINGLETON_H_
