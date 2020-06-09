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

#ifndef EULER_COMMON_ENV_H_
#define EULER_COMMON_ENV_H_

#include <functional>
#include <string>
#include <memory>

#include "euler/common/status.h"
#include "euler/common/slice.h"

namespace euler {

typedef std::function<void()> ThreadFunc;

class Thread {
 public:
  Thread() { }
  virtual ~Thread() { }

  virtual bool Joinable() = 0;
  virtual void Join() = 0;
};

class ThreadPool {
 public:
  ThreadPool() { }
  virtual ~ThreadPool() { }

  virtual void Schedule(ThreadFunc fn) = 0;
  virtual void Join() = 0;
  virtual void Shutdown() = 0;
};

class FileIO;

class Env {
 public:
  static Env* Default();

  virtual Thread* StartThread(const std::string& name, ThreadFunc fn) = 0;

  virtual ThreadPool* StartThreadPool(
      const std::string& name, int num_threads) = 0;

  virtual Status LoadLibrary(const char* library_filename, void** handle) = 0;

  virtual Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                      void** symbol) = 0;

  virtual Status NewFileIO(Slice path, bool readonly,
                           std::unique_ptr<FileIO>* file_io);
};

}  // namespace euler

#endif  // EULER_COMMON_ENV_H_
