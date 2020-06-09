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

#include "gtest/gtest.h"

#include "euler/common/logging.h"
#include "euler/parser/compiler.h"

namespace euler {

void PrintNodeProto(const DAGNodeProto& node_proto) {
  std::cout << "name: " << node_proto.name() << std::endl;
  std::cout << "op: " << node_proto.op() << std::endl;
  std::cout << "inputs: " << std::endl;
  for (int32_t i = 0; i < node_proto.inputs_size(); ++i) {
    std::cout << node_proto.inputs(i) << std::endl;
  }
  std::cout << "cond: " << std::endl;
  std::cout << "(" << std::endl;
  for (int32_t i = 0; i < node_proto.dnf_size(); ++i) {
    std::cout << node_proto.dnf(i) << std::endl;
  }
  std::cout << ")" << std::endl;
  std::cout << "post process: " << std::endl;
  for (int32_t i = 0; i < node_proto.post_process_size(); ++i) {
    std::cout << node_proto.post_process(i) << std::endl;
  }
  std::cout << "output_num: " << node_proto.output_num() << std::endl;
  std::cout << "outputs: " << std::endl;
  for (int32_t i = 0; i < node_proto.output_list_size(); ++i) {
    std::cout << node_proto.output_list(i) << std::endl;
  }
  std::cout << "op_alias: " << node_proto.op_alias() << std::endl;
  if (node_proto.op() == "REMOTE") {
    std::cout << "inner nodes: " << std::endl;
    std::cout << "[" << std::endl;
    for (int32_t i = 0; i < node_proto.inner_nodes_size(); ++i) {
      std::cout << "--------------" << std::endl;
      PrintNodeProto(node_proto.inner_nodes(i));
    }
    std::cout << "]" << std::endl;
  }
  if (node_proto.udf_name() != "") {
    std::cout << "udf: " << node_proto.udf_name() << std::endl;
    for (int32_t i = 0; i < node_proto.udf_str_params_size(); ++i) {
      std::cout << node_proto.udf_str_params(i) << std::endl;
    }
    for (int32_t i = 0; i < node_proto.udf_num_params_size(); ++i) {
      std::cout << node_proto.udf_num_params(i) << std::endl;
    }
  }
}

TEST(CompilerTest, Simple) {
  /* parse */
  Compiler::Init(2, distribute, "att:hash_range_index,price:range_index");
  // Compiler::Init(1, local, "att:hash_range_index,price:range_index");
  Compiler* compiler = Compiler::GetInstance();

  {
    std::string gremlin =
        "v(nodes).sampleLNB(edge_types, n, m, sqrt, 0).as(layer)";
    DAGDef* dag_def = compiler->CompileToDAGDef(gremlin, true);
    if (dag_def != nullptr) {
      std::vector<int32_t> sorted_ids = dag_def->TopologicSort();
      for (int32_t id : sorted_ids) {
        DAGNodeProto node_proto;
        dag_def->GetNodeById(id)->ToProto(&node_proto);
        std::cout << "===============" << std::endl;
        PrintNodeProto(node_proto);
      }
    } else {
      EULER_LOG(ERROR) << "error";
    }
  }

  std::cout << "**************************" << std::endl;

  {
    std::string gremlin =
        "v(nodes).outE(edge_types).has(p gt 3).as(oe)";
    DAGDef* dag_def = compiler->CompileToDAGDef(gremlin, true);
    if (dag_def != nullptr) {
      std::vector<int32_t> sorted_ids = dag_def->TopologicSort();
      for (int32_t id : sorted_ids) {
        DAGNodeProto node_proto;
        dag_def->GetNodeById(id)->ToProto(&node_proto);
        std::cout << "===============" << std::endl;
        PrintNodeProto(node_proto);
      }
    } else {
      EULER_LOG(ERROR) << "error";
    }
  }

  std::cout << "**************************" << std::endl;

  {
    std::string gremlin =
        "v(nodes).label().as(l)";
    DAGDef* dag_def = compiler->CompileToDAGDef(gremlin, true);
    if (dag_def != nullptr) {
      std::vector<int32_t> sorted_ids = dag_def->TopologicSort();
      for (int32_t id : sorted_ids) {
        DAGNodeProto node_proto;
        dag_def->GetNodeById(id)->ToProto(&node_proto);
        std::cout << "===============" << std::endl;
        PrintNodeProto(node_proto);
      }
    } else {
      EULER_LOG(ERROR) << "error";
    }
  }

  std::cout << "**************************" << std::endl;

  {
    std::string gremlin =
        "v(nodes).inV().has(p gt 2).as(l)";
    DAGDef* dag_def = compiler->CompileToDAGDef(gremlin, true);
    if (dag_def != nullptr) {
      std::vector<int32_t> sorted_ids = dag_def->TopologicSort();
      for (int32_t id : sorted_ids) {
        DAGNodeProto node_proto;
        dag_def->GetNodeById(id)->ToProto(&node_proto);
        std::cout << "===============" << std::endl;
        PrintNodeProto(node_proto);
      }
    } else {
      EULER_LOG(ERROR) << "error";
    }
  }

  std::cout << "**************************" << std::endl;

  {
    // op2DAG
    DAGDef* dag_def = compiler->Op2DAGDef(
        "API_SPARSE_GET_ADJ", "sparse_get_adj", 2,
        {"root_batch", "l_nb"}, {"edge_types", "m"});
    if (dag_def != nullptr) {
      std::vector<int32_t> sorted_ids = dag_def->TopologicSort();
      for (int32_t id : sorted_ids) {
        DAGNodeProto node_proto;
        dag_def->GetNodeById(id)->ToProto(&node_proto);
        std::cout << "===============" << std::endl;
        PrintNodeProto(node_proto);
      }
    } else {
      EULER_LOG(ERROR) << "error";
    }
  }

  std::cout << "**************************" << std::endl;

  {
    // op2DAG
    DAGDef* dag_def = compiler->Op2DAGDef(
        "API_GET_GRAPH_BY_LABEL", "get_graph", 2,
        {"labels"}, {});
    if (dag_def != nullptr) {
      std::vector<int32_t> sorted_ids = dag_def->TopologicSort();
      for (int32_t id : sorted_ids) {
        DAGNodeProto node_proto;
        dag_def->GetNodeById(id)->ToProto(&node_proto);
        std::cout << "===============" << std::endl;
        PrintNodeProto(node_proto);
      }
    } else {
      EULER_LOG(ERROR) << "error";
    }
  }
}

TEST(CompilerTest, MacroFusion) {
  Compiler::Init(2, graph_partition, "att:hash_range_index,price:range_index");
  // std::string gremlin = "sampleN(node_type, count).as(node).
  // sampleNB(edge_types, count, 0).as(nb)";

  // std::string gremlin = "v(nodes).has(price gt 3).
  // limit(10).as(n).sampleNB(e_types, nb_cnt, -1).as(nb)";

  // std::string gremlin = "v(nodes).as(n).
  // sampleNB(e_types, nb_cnt, -1).as(nb)";

  std::string gremlin = R"(v(nodes).sampleNB(edge_types, n, 0).as(n1).sampleNB(edge_types, n, 0).as(n2).v_select(n1).values(fid).as(n1_f))";
  DAGDef* dag_def = new DAGDef();
  Tree tree = BuildGrammarTree(gremlin);
  Translator translator(graph_partition);
  translator.Translate(tree, dag_def);

  /*std::vector<int32_t> sorted_ids = dag_def->TopologicSort();
  for (int32_t id : sorted_ids) {
    DAGNodeProto node_proto;
    dag_def->GetNodeById(id)->ToProto(&node_proto);
    std::cout << "===============" << std::endl;
    PrintNodeProto(node_proto);
  }*/

  Optimizer optimizer(graph_partition, 2);
  if (optimizer.Optimize(dag_def)) {
    std::vector<int32_t> sorted_ids = dag_def->TopologicSort();
    for (int32_t id : sorted_ids) {
      DAGNodeProto node_proto;
      dag_def->GetNodeById(id)->ToProto(&node_proto);
      std::cout << "===============" << std::endl;
      PrintNodeProto(node_proto);
    }
  }
}

}  // namespace euler
