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

#ifndef EULER_PARSER_COMPILER_H_
#define EULER_PARSER_COMPILER_H_

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <memory>

#include "euler/common/logging.h"
#include "euler/common/mutex.h"
#include "euler/parser/translator.h"
#include "euler/parser/optimizer.h"
#include "euler/parser/optimize_rule.h"
#include "euler/parser/tree.h"
#include "euler/core/dag_def/dag_def.h"
#include "euler/core/dag/dag.h"

namespace euler {
class Compiler {
 public:
  static void Init(int32_t shard_num, OptimizerType type,
                   std::string index_info) {
    static Compiler* temp = new Compiler(shard_num, type, index_info);
    instance_ = temp;
  }

  static Compiler* GetInstance() {
    if (instance_ == nullptr) {
      EULER_LOG(ERROR) << "compiler need init first!";
    }
    return instance_;
  }

  Compiler(Compiler const&) = delete;

  void operator=(Compiler const&) = delete;

  // debug
  DAGDef* CompileToDAGDef(const std::string& gremlin, bool optmz) {
    DAGDef* dag_def = new DAGDef();
    Tree tree = BuildGrammarTree(gremlin);
    translator_.Translate(tree, dag_def);
    if (!optmz) return dag_def;
    if (optimizer_.Optimize(dag_def)) {
      return dag_def;
    } else {
      EULER_LOG(ERROR) << "compile fail! " << gremlin;
      return nullptr;
    }
  }

  // debug
  DAGDef* Op2DAGDef(const std::string& op_name,
                    const std::string& alias,
                    int32_t output_num,
                    const std::vector<std::string>& input_tensor_names,
                    const std::vector<std::string>& norm_attr_names) {
    std::string key = op_name;
    for (const std::string& name : input_tensor_names) {
      key += " " + name;
    }
    for (const std::string& name : norm_attr_names) {
      key += " " + name;
    }
    DAGDef* dag_def = new DAGDef();
    std::shared_ptr<NodeDef> node_def =
        dag_def->ProduceNodeDef(op_name, output_num);
    for (const std::string& name : input_tensor_names) {
      node_def->input_edges_.push_back({name, -1, -1});
    }
    for (const std::string& name : norm_attr_names) {
      node_def->attrs_.push_back(
          std::make_shared<NormAttrDef>(name));
    }
    std::unordered_set<int32_t> pre, succ;
    dag_def->AddNodeDef(node_def, pre, succ);
    // add as op
    std::shared_ptr<NodeDef> as_node =
        dag_def->ProduceNodeDef("AS", output_num);
    as_node->op_alias_ = alias;
    for (int32_t i = 0; i < output_num; ++i) {
      as_node->input_edges_.push_back({node_def->name_, node_def->id_, i});
    }
    pre.insert(node_def->id_);
    dag_def->AddNodeDef(as_node, pre, succ);

    if (optimizer_.Optimize(dag_def)) {
      return dag_def;
    } else {
      EULER_LOG(FATAL) << "compile fail! " << key;
      return nullptr;
    }
  }


  DAG* Compile(const std::string& gremlin) {
    MutexLock l(&mu_);
    if (dag_cache_.find(gremlin) == dag_cache_.end()) {
      EULER_LOG(INFO) << "compiling: " << gremlin;
      DAGDef dag_def;
      Tree tree = BuildGrammarTree(gremlin);
      translator_.Translate(tree, &dag_def);
      if (optimizer_.Optimize(&dag_def)) {
        dag_cache_[gremlin] = DAG::NewFromProto(dag_def.ToProto()).release();
      } else {
        EULER_LOG(FATAL) << "compile fail! " << gremlin;
      }
    }
    return dag_cache_[gremlin];
  }

  DAG* Op2DAG(const std::string& op_name,
              const std::string& alias,
              int32_t output_num,
              const std::vector<std::string>& input_tensor_names,
              const std::vector<std::string>& norm_attr_names) {
    std::string key = op_name;
    for (const std::string& name : input_tensor_names) {
      key += " " + name;
    }
    for (const std::string& name : norm_attr_names) {
      key += " " + name;
    }
    MutexLock l(&mu_);
    if (dag_cache_.find(key) == dag_cache_.end()) {
      EULER_LOG(INFO) << "processing single op: " << key;
      DAGDef dag_def;
      std::shared_ptr<NodeDef> node_def =
          dag_def.ProduceNodeDef(op_name, output_num);
      for (const std::string& name : input_tensor_names) {
        node_def->input_edges_.push_back({name, -1, -1});
      }
      for (const std::string& name : norm_attr_names) {
        node_def->attrs_.push_back(
            std::make_shared<NormAttrDef>(name));
      }
      // add op
      std::unordered_set<int32_t> pre, succ;
      dag_def.AddNodeDef(node_def, pre, succ);
      // add as op
      std::shared_ptr<NodeDef> as_node =
          dag_def.ProduceNodeDef("AS", output_num);
      as_node->op_alias_ = alias;
      for (int32_t i = 0; i < output_num; ++i) {
        as_node->input_edges_.push_back({node_def->name_, node_def->id_, i});
      }
      pre.insert(node_def->id_);
      dag_def.AddNodeDef(as_node, pre, succ);

      if (optimizer_.Optimize(&dag_def)) {
        dag_cache_[key] = DAG::NewFromProto(dag_def.ToProto()).release();
      } else {
        EULER_LOG(FATAL) << "compile fail! " << key;
      }
    }
    return dag_cache_[key];
  }

  std::unordered_map<std::string, std::vector<std::string>>
  GetIndexInfo() {
    return index_info_;
  }

  ~Compiler() {
    for (auto it = dag_cache_.begin(); it != dag_cache_.end(); ++it) {
      delete it->second;
    }
  }

 private:
  Mutex mu_;
  std::unordered_map<std::string, DAG*> dag_cache_;
  int32_t shard_num_;
  Optimizer optimizer_;
  Translator translator_;
  std::unordered_map<std::string, std::vector<std::string>> index_info_;
  static Compiler* instance_;
  explicit Compiler(int32_t shard_num, OptimizerType type,
                    std::string index_info);
};

}  // namespace euler

#endif  // EULER_PARSER_COMPILER_H_
