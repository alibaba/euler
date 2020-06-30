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

#include <string>
#include <vector>

#include "euler/parser/attribute_calculator.h"
#include "euler/common/logging.h"

namespace euler {

bool SimpleCondition(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  std::string op = children[0]->GetValue();
  std::string p = children[1]->GetValue();
  t->GetProp()->AddValue(op);  // op
  t->GetProp()->AddValue(p);  // num
  return true;
}

bool HasKey(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  t->GetProp()->AddValue(children[0]->GetValue());  // has_key
  t->GetProp()->AddValue(children[1]->GetValue());  // p
  return true;
}

bool HasLabel(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  t->GetProp()->AddValue(children[0]->GetValue());  // has_label
  t->GetProp()->AddValue(children[1]->GetValue());  // p
  return true;
}

bool Has(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  t->GetProp()->AddValue(children[1]->GetValue());  // p
  std::vector<std::string> simple_condition =
      children[2]->GetProp()->GetValues();
  t->GetProp()->AddValue(simple_condition[0]);  // op
  t->GetProp()->AddValue(simple_condition[1]);  // p
  return true;
}

bool Term(TreeNode* t) {
  TreeNode* child = (t->GetChildren())[0];
  std::vector<std::string> values = child->GetProp()->GetValues();
  for (const std::string& v : values) {
    t->GetProp()->AddValue(v);
  }
  return true;
}

bool CONJ(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  if (children.size() == 3) {  // TERM and_ CONJ
    Prop* term = children[0]->GetProp();
    std::vector<Prop*> terms = children[2]->GetProp()->GetNestingValues();
    t->GetProp()->AddNestingValue(term);
    for (Prop* te : terms) {
      t->GetProp()->AddNestingValue(te);
    }
  } else {  // TERM
    Prop* term = children[0]->GetProp();
    t->GetProp()->AddNestingValue(term);
  }
  return true;
}

bool DNF(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  Prop* conj = children[0]->GetProp();
  t->GetProp()->AddNestingValue(conj);
  if (children.size() == 3) {  // CONJ or_ DNF
    std::vector<Prop*> conjs = children[2]->GetProp()->GetNestingValues();
    for (Prop* c : conjs) {
      t->GetProp()->AddNestingValue(c);
    }
  }
  return true;
}

bool Limit(TreeNode* t) {
  TreeNode* child = (t->GetChildren())[1];
  t->GetProp()->AddValue("limit");
  t->GetProp()->AddValue(child->GetValue());
  return true;
}

bool OrderBy(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  t->GetProp()->AddValue("order_by");
  t->GetProp()->AddValue(children[1]->GetValue());
  t->GetProp()->AddValue(children[2]->GetValue());
  return true;
}

bool As(TreeNode* t) {
  TreeNode* child = (t->GetChildren())[1];
  t->GetProp()->AddValue(child->GetValue());
  return true;
}

// 中把后处理的命令按顺序排列存储在NestingValue中
bool PostProcess(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  for (TreeNode* node : children) {
    t->GetProp()->AddNestingValue(node->GetProp());
  }
  return true;
}

// NestingValues第一个域是DNF，第二个域是PostProcess
bool Condtition(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  t->GetProp()->AddNestingValue(nullptr);
  t->GetProp()->AddNestingValue(nullptr);
  for (TreeNode* node : children) {
    if (node->GetType() == "DNF") {
      t->GetProp()->SetNestingValue(node->GetProp(), 0);
    } else {  // POST_PROCESS
      t->GetProp()->SetNestingValue(node->GetProp(), 1);
    }
  }
  return true;
}

bool Params(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  t->GetProp()->AddValue(children[0]->GetValue());  // p
  if (children.size() == 2) {  // p PARAMS
    for (const std::string& p : children[1]->GetProp()->GetValues()) {
      t->GetProp()->AddValue(p);
    }
  }
  return true;
}

bool Va(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  for (const std::string& p : children[1]->GetProp()->GetValues()) {  // fids
    t->GetProp()->AddValue(p);
  }
  if (children.size() > 2) {
    t->GetProp()->AddValue(children[2]->GetValue());  // udf name
    for (const std::string& p :
         children[3]->GetProp()->GetValues()) {  // udf process fids
      t->GetProp()->AddValue(p);
    }
    if (children.size() > 4) {  // contains udf params list
      t->GetProp()->AddValue("[");
      for (const std::string& num : children[5]->GetProp()->GetValues()) {
        t->GetProp()->AddValue(num);
      }
      t->GetProp()->AddValue("]");
    }
  }
  return true;
}

bool SampleNB(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  for (const std::string& p : children[1]->GetProp()->GetValues()) {
    t->GetProp()->AddValue(p);
  }  // params
  t->GetProp()->AddValue(children[2]->GetValue());  // num
  return true;
}

bool SampleLNB(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  for (const std::string& p : children[1]->GetProp()->GetValues()) {
    t->GetProp()->AddValue(p);
  }  // params
  t->GetProp()->AddValue(children[2]->GetValue());  // num
  return true;
}

bool SampleEdge(TreeNode* t) {
  TreeNode* child = (t->GetChildren())[1];
  for (const std::string& p : child->GetProp()->GetValues()) {
    t->GetProp()->AddValue(p);
  }
  return true;
}

bool SampleNode(TreeNode* t) {
  TreeNode* child = (t->GetChildren())[1];
  for (const std::string& p : child->GetProp()->GetValues()) {
    t->GetProp()->AddValue(p);
  }
  return true;
}

bool SampleNWithTypes(TreeNode* t) {
  TreeNode* child = (t->GetChildren())[1];
  for (const std::string& p : child->GetProp()->GetValues()) {
    t->GetProp()->AddValue(p);
  }
  return true;
}

bool E(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  if (children.size() == 2) {
    t->GetProp()->AddValue(children[1]->GetValue());
  }
  return true;
}

bool V(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  if (children.size() == 2) {
    t->GetProp()->AddValue(children[1]->GetValue());
  }
  return true;
}

bool APISampleNB(TreeNode* t) {
  // SAMPLE_NB
  std::vector<TreeNode*> children = t->GetChildren();
  for (const std::string& v : children[0]->GetProp()->GetValues()) {
    t->GetProp()->AddValue(v);
  }
  // SAMPLE_NB CONDITION
  if (children.size() == 2 && children[1]->GetType() == "CONDITION") {
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
  } else if (children.size() == 2 &&
             children[1]->GetType() == "AS") {  // SAMPLE_NB AS
    t->SetOpAlias(children[1]->GetProp()->GetValues()[0]);
  } else if (children.size() == 3) {  // SAMPLE_NB CONDITION AS
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
    t->SetOpAlias(children[2]->GetProp()->GetValues()[0]);
  }
  return true;
}

bool APISampleLNB(TreeNode* t) {
  TreeNode* child = (t->GetChildren())[0];
  for (const std::string& p : child->GetProp()->GetValues()) {
    t->GetProp()->AddValue(p);
  }
  // contains AS
  if (t->GetChildren().size() == 2) {
    t->SetOpAlias((t->GetChildren())[1]->GetProp()->GetValues()[0]);
  }
  return true;
}

// NestingValues存放condition。第一个域是DNF，第二个域是PostProcess
bool APIGetNBEdge(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  t->GetProp()->AddValue(children[1]->GetValue());
  if (children.size() == 3 &&
      children[2]->GetType() == "CONDITION") {  // out_e p CONDITION
    for (Prop* p : children[2]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
  } else if (children.size() == 3 &&
             children[2]->GetType() == "AS") {  // out_e p AS
    t->SetOpAlias(children[2]->GetProp()->GetValues()[0]);
  } else if (children.size() == 4) {  // out_e p CONDITION AS
    for (Prop* p : children[2]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
    t->SetOpAlias(children[3]->GetProp()->GetValues()[0]);
  }
  return true;
}

// NestingValues存放condition。第一个域是DNF，第二个域是PostProcess
bool APIGetRNBNode(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  if (children.size() == 2 &&
      children[1]->GetType() == "CONDITION") {  // in_v CONDITION
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
  } else if (children.size() == 2 &&
             children[1]->GetType() == "AS") {  // in_v AS
    t->SetOpAlias(children[1]->GetProp()->GetValues()[0]);
  } else if (children.size() == 3) {  // in_v CONDITION AS
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
    t->SetOpAlias(children[2]->GetProp()->GetValues()[0]);
  }
  return true;
}

// NestingValues存放condition。第一个域是DNF，第二个域是postprocess
bool APIGetNBNode(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  t->GetProp()->AddValue(children[1]->GetValue());
  if (children.size() == 3 &&
      children[2]->GetType() == "CONDITION") {  // out_v p CONDITION
    for (Prop* p : children[2]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
  } else if (children.size() == 3 &&
             children[2]->GetType() == "AS") {  // out_v p AS
    t->SetOpAlias(children[2]->GetProp()->GetValues()[0]);
  } else if (children.size() == 4) {  // out_v p CONDITION AS
    for (Prop* p : children[2]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
    t->SetOpAlias(children[3]->GetProp()->GetValues()[0]);
  }
  return true;
}

bool APIGetNodeT(TreeNode* t) {
  (void) t;
  return true;
}

bool APIGetP(TreeNode* t) {
  // VA
  TreeNode* child = (t->GetChildren())[0];
  for (const std::string& v : child->GetProp()->GetValues()) {
    t->GetProp()->AddValue(v);
  }
  // contains AS
  if (t->GetChildren().size() == 2) {
    t->SetOpAlias((t->GetChildren())[1]->GetProp()->GetValues()[0]);
  }
  return true;
}

bool APISampleEdge(TreeNode* t) {
  // SAMPLE_EDGE
  std::vector<TreeNode*> children = t->GetChildren();
  for (const std::string& v : children[0]->GetProp()->GetValues()) {
    t->GetProp()->AddValue(v);
  }
  // SAMPLE_EDGE CONDITION
  if (children.size() == 2 && children[1]->GetType() == "CONDITION") {
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
  } else if (children.size() == 2 && children[1]->GetType() == "AS") {
    t->SetOpAlias(children[1]->GetProp()->GetValues()[0]);
  } else if (children.size() == 3) {
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
    t->SetOpAlias(children[2]->GetProp()->GetValues()[0]);
  }
  return true;
}

// NestingValues存放condition。第一个域是DNF，第二个域是PostProcess
// Values存放E附带的value
bool APIGetEdge(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  for (const std::string& v : children[0]->GetProp()->GetValues()) {
    t->GetProp()->AddValue(v);
  }  // E
  if (children.size() == 2 &&
      children[1]->GetType() == "CONDITION") {  // E CONDITION
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
  } else if (children.size() == 2 && children[1]->GetType() == "AS") {  // E AS
    t->SetOpAlias(children[1]->GetProp()->GetValues()[0]);
  } else if (children.size() == 3) {  // E CONDITION AS
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
    t->SetOpAlias(children[2]->GetProp()->GetValues()[0]);
  }
  return true;
}

bool APISampleNode(TreeNode* t) {
  // SAMPLE_NODE
  std::vector<TreeNode*> children = t->GetChildren();
  for (const std::string& v : children[0]->GetProp()->GetValues()) {
    t->GetProp()->AddValue(v);
  }
  // SAMPLE_NODE CONDITION
  if (children.size() == 2 && children[1]->GetType() == "CONDITION") {
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
  } else if (children.size() == 2 &&
             children[1]->GetType() == "AS") {  // SAMPLE_NODE AS
    t->SetOpAlias(children[1]->GetProp()->GetValues()[0]);
  } else if (children.size() == 3) {  // SAMPLE_NODE CONDITION AS
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
    t->SetOpAlias(children[2]->GetProp()->GetValues()[0]);
  }
  return true;
}

bool APISampleNWithTypes(TreeNode* t) {
  // SAMPLE_N_WITH_TYPES
  std::vector<TreeNode*> children = t->GetChildren();
  for (const std::string& v : children[0]->GetProp()->GetValues()) {
    t->GetProp()->AddValue(v);
  }
  // SAMPLE_N_WITH_TYPES CONDITION
  if (children.size() == 2 && children[1]->GetType() == "CONDITION") {
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
  } else if (children.size() == 2 &&
             children[1]->GetType() == "AS") {  // SAMPLE_N_WITH_TYPES AS
    t->SetOpAlias(children[1]->GetProp()->GetValues()[0]);
  } else if (children.size() == 3) {  // SAMPLE_N_WITH_TYPES CONDITION AS
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
    t->SetOpAlias(children[2]->GetProp()->GetValues()[0]);
  }
  return true;
}

// NestingValues存放condition。第一个域是DNF，第二个域是PostProcess
// Values存放V附带的value
bool APIGetNode(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  for (const std::string& v : children[0]->GetProp()->GetValues()) {
    t->GetProp()->AddValue(v);
  }  // V
  if (children.size() == 2 &&
      children[1]->GetType() == "CONDITION") {  // V CONDITION
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
  } else if (children.size() == 2 && children[1]->GetType() == "AS") {  // V AS
    t->SetOpAlias(children[1]->GetProp()->GetValues()[0]);
  } else if (children.size() == 3) {  // V CONDITION AS
    for (Prop* p : children[1]->GetProp()->GetNestingValues()) {
      t->GetProp()->AddNestingValue(p);
    }
    t->SetOpAlias(children[2]->GetProp()->GetValues()[0]);
  }
  return true;
}

bool Select(TreeNode* t) {
  std::vector<TreeNode*> children = t->GetChildren();
  t->GetProp()->AddValue(children[1]->GetValue());  // p
  return true;
}

}  // namespace euler
