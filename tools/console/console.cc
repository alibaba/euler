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

#include <stdlib.h>
#include <string.h>

#include <iostream>

#include <functional>
#include <map>
#include <vector>
#include <string>
#include <atomic>

#include "linenoise/linenoise.h"

#include "euler/client/graph.h"
#include "euler/common/string_util.h"

static std::unique_ptr<euler::client::Graph> graph;
static std::atomic<int> console_var(0);

static const char* kCommands[][4] = {
  {
    "help",  // commnad
    "Command help message",  // commond description
    "help [command]",  // command help
    "help con"  // command example
  },
  {
    "con",
    "Connect to graph server",
    "con <config>",
    "con \"mode=Remote;zk_server=localhost;zk_path=/euler\""
  },
  {
    "nf",
    "Get features for nodes",
    "nf <type> <nids> <fids>",
    "nf dense \"1, 2, 3\" \"0, 1\""
  },
  {
    "ef",
    "Get features for edges",
    "ef <type> <nids> <fids>",
    "ef sparse \"1:2:0, 2:3:1, 3:5:1\" \"0, 1\""
  },
  {
    "nb",
    "Get neighbors for nodes",
    "nb <nids> <etypes>",
    "nb \"1, 2, 3, 4\" \"0, 1, 2\""
  }
};


bool IsWhiteSpace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

const char* Read(const char* ptr, std::function<bool(char)> func) {
  while (func(*ptr)) {
    ++ptr;
  }
  return ptr;
}

std::string Trim(const std::string& str) {
  if (str.empty()) {
    return str;
  }
  auto p = Read(str.c_str(), IsWhiteSpace);
  auto q = str.c_str() + str.size();
  while (q > p && IsWhiteSpace(*(q - 1))) {
    --q;
  }
  return std::string(p, q);
}

std::vector<std::string> ParseCommand(const std::string& cmd) {
  std::vector<std::string> cmd_and_args;

  // Parse command
  auto p = Read(cmd.c_str(), IsWhiteSpace);
  auto q = Read(p, [] (char c) { return c != '\0' && !IsWhiteSpace(c); });
  std::string command(p, q);
  bool valid = false;
  for (auto& CMD : kCommands) {
    if (command == CMD[0])  {
      valid = true;
      break;
    }
  }

  if (!valid) {
    return cmd_and_args;
  }

  cmd_and_args.push_back(command);

  // Parse argments
  p = q;
  while (*p != '\0') {
    p = Read(p, IsWhiteSpace);
    if (*p == '"') {
      ++p;
      q = Read(p, [] (char c) { return c != '\0' && c != '"'; });
      if (*q == '\0') {
        std::cerr << "Invalid command, expect \", but "
                  << int(*q) << " got" << std::endl;
        cmd_and_args.clear();
        return cmd_and_args;
      }
    } else {
      q = Read(p, [] (char c) { return c != '\0' && !IsWhiteSpace(c); });
    }
    cmd_and_args.push_back(std::string(p, q));
    p = q;
    if (*p == '"') {
      ++p;
    }
  }

  return cmd_and_args;
}

void PrintMainBoard() {
  std::cout << "-----------------------------------------------------------\n"
            << "                       Euler Console                       \n"
            << "\n";
  for (auto& CMD : kCommands) {
    std::cout << CMD[0] << "\t" << CMD[1]
              <<  "\t" << CMD[2] << std::endl;
  }
  std::cout << "-----------------------------------------------------------\n";
  std::cout << std::endl;
}

void Help(const std::vector<std::string>& cmd_and_args) {
  if (cmd_and_args.size() == 1) {
    PrintMainBoard();
  } else if (cmd_and_args.size() == 2) {
    for (auto& CMD : kCommands) {
      if (cmd_and_args[1] == CMD[0]) {
        std::cout << CMD[0] << "\t" << CMD[1] << "\n"
                  << "\t command: " << CMD[2] << "\n"
                  << "\t example: " << CMD[3] << "\n"
                  << std::endl;
      }
    }
  } else {
    std::cerr << kCommands[0][2] << std::endl;
  }
}

void Connect(const std::vector<std::string>& cmd_and_args) {
  if (cmd_and_args.size() != 2) {
    std::cerr << "Invalid command got!" << std::endl;
    return Help({"help", cmd_and_args[0]});
  }

  auto& conf = cmd_and_args[1];
  euler::client::GraphConfig config;
  std::vector<std::string> vec;
  euler::common::split_string(conf, ';', &vec);
  if (vec.empty()) {
    return;
  }

  for (auto it = vec.begin(); it != vec.end(); ++it) {
    std::vector<std::string> key_value;
    euler::common::split_string(*it, '=', &key_value);
    if (key_value.size() != 2 || key_value[0].empty() || key_value[1].empty()) {
      return;
    }
    config.Add(key_value[0], key_value[1]);
  }

  graph = euler::client::Graph::NewGraph(config);
}

std::vector<euler::client::EdgeID> BuildEdgeIds(const std::string& arg) {
  std::vector<euler::client::EdgeID> eids;
  if (arg.empty()) {
    return eids;
  }

  std::vector<std::string> values;
  euler::common::split_string(arg, ',', &values);
  for (auto& v : values) {
    std::vector<std::string> arr;
    euler::common::split_string(v, ':', &arr);
    if (arr.size() != 3) {
      eids.clear();
      return eids;
    }
    auto v1 = Trim(arr[0]);
    auto v2 = Trim(arr[1]);
    auto v3 = Trim(arr[2]);
    if (!v1.empty() && !v2.empty() && !v3.empty()) {
      euler::client::NodeID sid = strtoll(v1.c_str(), NULL, 10);
      euler::client::NodeID did = strtoll(v2.c_str(), NULL, 10);
      int type = atoi(v3.c_str());
      eids.push_back(std::make_tuple(sid, did, type));
    }
  }
  return eids;
}

template <typename T>
void BuildIntVec(const std::string& arg, std::vector<T>* vec) {
  if (arg.empty()) {
    return;
  }
  std::vector<std::string> values;
  euler::common::split_string(arg, ',', &values);
  for (auto& v : values) {
    v = Trim(v);
    if (!v.empty()) {
      vec->push_back(strtoll(v.c_str(), NULL, 10));
    }
  }
}

#define PrintResult()                           \
  for (auto& n : result) {                      \
    std::cout << "[\n";                         \
    for (auto& f : n) {                         \
      std::cout << "  [";                       \
      int i = 0;                                \
      for (auto& v : f) {                       \
        if (i > 0) {                            \
          std::cout <<  ", ";                   \
        }                                       \
        std::cout << v;                         \
        ++i;                                    \
      }                                         \
      std::cout << "]\n";                       \
    }                                           \
    std::cout << "]\n";                         \
  }                                             \
  --console_var;                                \


void DenseCallback(const euler::client::FloatFeatureVec& result) {
  PrintResult();
}

void SparseCallback(const euler::client::UInt64FeatureVec& result) {
  PrintResult();
}

void BinaryCallback(const euler::client::BinaryFatureVec& result) {
  std::cout << std::hex;
  PrintResult();
  std::cout << std::dec;
}

#undef PrintResult  // PrintResult

void GetNodeFeature(const std::vector<std::string>& cmd_and_args) {
  if (cmd_and_args.size() != 4) {
    std::cerr << "Please input valid arguments!" << std::endl;
    return Help({"help", cmd_and_args[0]});
  }

  auto type = cmd_and_args[1];
  euler::client::NodeIDVec node_ids;
  BuildIntVec(cmd_and_args[2], &node_ids);
  std::vector<int> feature_ids;
  BuildIntVec(cmd_and_args[3], &feature_ids);

  if (node_ids.empty()) {
    std::cerr << "Invalid command, you must specify valid node ids"
              << std::endl;
    return;
  }

  if (feature_ids.empty()) {
    std::cerr << "Invalid command, you must specify valid feature ids"
              << std::endl;
    return;
  }

  if (graph == nullptr) {
    std::cerr << "Please connect graph first" << std::endl;
    return;
  }
  if (type == "dense") {
    ++console_var;
    graph->GetNodeFloat32Feature(node_ids, feature_ids, DenseCallback);
  } else if (type == "sparse") {
    ++console_var;
    graph->GetNodeUint64Feature(node_ids, feature_ids, SparseCallback);
  } else if (type == "binary") {
    ++console_var;
    graph->GetNodeBinaryFeature(node_ids, feature_ids, BinaryCallback);
  } else {
    std::cerr << "Invalid feature type: " << type
              << ", value must be in [dense|sparse|binary]"
              << std::endl;
  }
}

void GetEdgeFeature(const std::vector<std::string>& cmd_and_args) {
  if (cmd_and_args.size() != 4) {
    std::cerr << "Please input valid arguments!" << std::endl;
    return Help({"help", cmd_and_args[0]});
  }

  auto type = cmd_and_args[1];
  auto edge_ids = BuildEdgeIds(cmd_and_args[2]);
  std::vector<int> feature_ids;
  BuildIntVec(cmd_and_args[3], &feature_ids);

  if (edge_ids.empty()) {
    std::cerr << "Invalid command, you must specify valid edge ids"
              << std::endl;
    return;
  }

  if (feature_ids.empty()) {
    std::cerr << "Invalid command, you must specify valid feature ids"
              << std::endl;
    return;
  }

  if (graph == nullptr) {
    std::cerr << "Please connect graph first" << std::endl;
    return;
  }

  if (type == "dense") {
    ++console_var;
    graph->GetEdgeFloat32Feature(edge_ids, feature_ids, DenseCallback);
  } else if (type == "sparse") {
    ++console_var;
    graph->GetEdgeUint64Feature(edge_ids, feature_ids, SparseCallback);
  } else if (type == "binary") {
    ++console_var;
    graph->GetEdgeBinaryFeature(edge_ids, feature_ids, BinaryCallback);
  } else {
    std::cerr << "Invalid feature type: " << type
              << ", value must be in [dense|sparse|binary]"
              << std::endl;
  }
}

void NeighborCallback(const euler::client::IDWeightPairVec& result) {
  for (auto& n : result) {
    std::cout << "[\n";
    for (auto& iw : n) {
      std::cout << "  {\n"
                << "    \"id\": " << std::get<0>(iw) << ",\n"
                << "    \"weight\": " << std::get<1>(iw) << ",\n"
                << "    \"type\": " << std::get<2>(iw) << "\n"
                << "  }\n";
    }
    std::cout << "]\n";
  }
  --console_var;
}

void GetNeighbor(const std::vector<std::string>& cmd_and_args) {
  if (cmd_and_args.size() != 3) {
    std::cerr << "Please input valid arguments!" << std::endl;
    return Help({"help", cmd_and_args[0]});
  }

  euler::client::NodeIDVec node_ids;
  BuildIntVec(cmd_and_args[1], &node_ids);
  std::vector<int> edge_types;
  BuildIntVec(cmd_and_args[2], &edge_types);

  if (node_ids.empty()) {
    std::cerr << "Invalid command, please input valid ndoe ids"
              << std::endl;
    return;
  }

  if (edge_types.empty()) {
    std::cout << "Invalid command, please input valid edge types"
              << std::endl;
    return;
  }

  if (graph == nullptr) {
    std::cerr << "Please connect graph first" << std::endl;
    return;
  }
  ++console_var;
  graph->GetFullNeighbor(node_ids, edge_types, NeighborCallback);
}

void Execute(const std::vector<std::string>& cmd_and_args) {
  if (cmd_and_args.empty()) {
    return;
  }

  using CallFunc = std::function<void(const std::vector<std::string>&)>;
  const std::map<std::string, CallFunc> kCallMap = {
    {"help", Help},
    {"con", Connect},
    {"nf", GetNodeFeature},
    {"ef", GetEdgeFeature},
    {"nb", GetNeighbor}
  };

  auto& cmd = cmd_and_args[0];
  auto it = kCallMap.find(cmd);
  it->second(cmd_and_args);
}

void EulerCompletion(const char* buf, linenoiseCompletions* lc) {
  switch (buf[0]) {
    case 'h': {
      linenoiseAddCompletion(lc, kCommands[0][2]);
      break;
    }
    case 'c': {
      linenoiseAddCompletion(lc, kCommands[1][2]);
      break;
    }
    case 'n': {
      linenoiseAddCompletion(lc, kCommands[2][2]);
      linenoiseAddCompletion(lc, kCommands[4][2]);
      break;
    }
    case 'e': {
      linenoiseAddCompletion(lc, kCommands[3][2]);
      break;
    }
  }
}

int main(int argc, char** argv) {
  PrintMainBoard();

  if (argc > 1) {
    Connect({"con", argv[1]});
  }

  linenoiseSetCompletionCallback(EulerCompletion);

  linenoiseHistorySetMaxLen(20);
  const char* kHistoryFile = "/tmp/.euler.txt";
  linenoiseHistoryLoad(kHistoryFile);

  char* command = nullptr;
  while ((command = linenoise("euler> ")) != nullptr) {
    Execute(ParseCommand(command));

    while (console_var) {
      // busy loop
    }

    if (command[0] != '\0') {
      linenoiseHistoryAdd(command);
      linenoiseHistorySave(kHistoryFile);
    }

    free(command);
  }

  return 0;
}
