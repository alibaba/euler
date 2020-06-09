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


#include "euler/service/server_interface.h"

#include <memory>
#include <unordered_map>

#include "euler/common/logging.h"
#include "euler/common/mutex.h"
#include "euler/common/status.h"

namespace euler {

namespace {

Mutex* get_server_factory_lock() {
  static Mutex server_factory_lock;
  return &server_factory_lock;
}

typedef std::unordered_map<std::string, ServerFactory*> ServerFactories;

ServerFactories* server_factories() {
  static ServerFactories* factories = new ServerFactories;
  return factories;
}

}  // namespace

std::string ServerDef::DebugString() const {
  std::string result;
  result += "protocol: " + protocol + "\n";
  result += "shard_index: " + std::to_string(shard_index) + "\n";
  result += "shard_number: " + std::to_string(shard_number) + "\n";
  result += "options:\n";
  for (auto it = options.begin(); it != options.end(); ++it) {
    result += it->first + ": " + it -> second + "\n";
  }
  return result;
}

/* static */
void ServerFactory::Register(const std::string& server_type,
                             ServerFactory* factory) {
  MutexLock l(get_server_factory_lock());
  if (!server_factories()->insert({server_type, factory}).second) {
    EULER_LOG(ERROR) << "Two server factories are being registered under "
                     << server_type;
  }
}

/* static */
Status ServerFactory::GetFactory(const ServerDef& server_def,
                                 ServerFactory** out_factory) {
  MutexLock l(get_server_factory_lock());
  for (const auto& server_factory : *server_factories()) {
    if (server_factory.second->AcceptsOptions(server_def)) {
      *out_factory = server_factory.second;
      return Status::OK();
    }
  }

  return Status::NotFound(
      "No server factory registered for the given ServerDef: ",
      server_def.DebugString());
}

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
Status NewServer(const ServerDef& server_def,
                 std::unique_ptr<ServerInterface>* out_server) {
  ServerFactory* factory;
  RETURN_IF_ERROR(ServerFactory::GetFactory(server_def, &factory));
  return factory->NewServer(server_def, out_server);
}

}  // namespace euler
