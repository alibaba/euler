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

#ifndef EULER_PARSER_TREE_H_
#define EULER_PARSER_TREE_H_

#include <stdarg.h>

#include <vector>
#include <queue>
#include <string>
#include <sstream>
#include <iostream>

class Prop {
 public:
  std::vector<std::string> GetValues() const {
    return values_;
  }

  void AddValue(std::string value) {
    values_.push_back(value);
  }

  bool SetValue(std::string value, size_t idx) {
    if (values_.size() <= idx) {
      return false;
    }
    values_[idx] = value;
    return true;
  }

  std::vector<Prop*> GetNestingValues() const {
    return nesting_values_;
  }

  void AddNestingValue(Prop* value) {
    nesting_values_.push_back(value);
  }

  bool SetNestingValue(Prop* value, size_t idx) {
    if (nesting_values_.size() <= idx) {
      return false;
    }
    nesting_values_[idx] = value;
    return true;
  }

  void Print() const {
    if (!values_.empty()) {
      std::cout << "(";
    }
    for (const std::string& value : values_) {
      std::cout << value << " ";
    }
    if (!values_.empty()) {
      std::cout << ")" << std::endl;
    }
    if (!nesting_values_.empty()) {
      std::cout << "[" << std::endl;
    }
    for (Prop* p : nesting_values_) {
      if (p != nullptr) {
        p->Print();
      }
    }
    if (!nesting_values_.empty()) {
      std::cout << "]" << std::endl;
    }
  }

 private:
  std::vector<std::string> values_;
  std::vector<Prop*> nesting_values_;
};

class TreeNode {
 public:
  TreeNode(std::string type,
           std::string value) {
    type_ = type;
    value_ = value;
    property_ = new Prop();
    op_alias_ = "";
    parent_ = nullptr;
  }

  explicit TreeNode(std::string type) {
    type_ = type;
    value_ = type;
    property_ = new Prop();
    op_alias_ = "";
    parent_ = nullptr;
  }

  std::string GetValue() const {
    return value_;
  }

  std::string GetType() const {
    return type_;
  }

  std::vector<TreeNode*> GetChildren() const {
    return children_;
  }

  void SetOpAlias(std::string alias) {
    op_alias_ = alias;
  }

  std::string GetOpAlias() const {
    return op_alias_;
  }

  void AddChild(TreeNode* c) {
    children_.push_back(c);
    c->parent_ = this;
  }

  void AddChildren(int32_t num, ...) {
    va_list valist;
    va_start(valist, num);
    for (int32_t i = 0; i < num; i++) {
      AddChild(va_arg(valist, TreeNode*));
    }
    va_end(valist);
  }

  void DeleteChild(TreeNode* t) {
    for (int32_t i = children_.size() - 1; i >= 0; --i) {
      if (children_[i] == t) {
        children_.erase(children_.begin() + i);
        delete t;
        break;
      }
    }
  }

  void ClearChildren() {
    for (TreeNode* node : children_) {
      delete node;
    }
    children_.clear();
  }

  TreeNode* GetParent() const {
    return parent_;
  }

  Prop* GetProp() const {
    return property_;
  }

  TreeNode* FindLeft() {
    TreeNode* left = this;
    while (!left->children_.empty()) {
      left = left->children_.back();
    }
    return left;
  }

  virtual ~TreeNode() {
    for (TreeNode* node : children_) {
      if (node != nullptr) delete node;
    }
    delete property_;
  }

 private:
  std::string type_;

  std::string value_;

  TreeNode* parent_;

  Prop* property_;  // 综合属性

  std::string op_alias_;

  std::vector<TreeNode*> children_;
};

class Tree {
 public:
  Tree() {
    root_ = nullptr;
  }

  explicit Tree(TreeNode* root) {
    root_ = root;
  }

  void AddNode(TreeNode* p, TreeNode* c) {
    if (p != nullptr) {
      p->AddChild(c);
    } else {
      root_ = c;
    }
  }

  void DeleteNode(TreeNode* t) {
    if (t->GetParent() != nullptr) {
      t->GetParent()->DeleteChild(t);
    } else {
      delete t;
    }
  }

  void PostTraversal(TreeNode* root,
                     std::vector<TreeNode*>* results) const {
    if (root == nullptr) {
      root = root_;
    }
    std::vector<TreeNode*> children = root->GetChildren();
    for (TreeNode* node : children) {
      PostTraversal(node, results);
    }
    results->push_back(root);
  }

  std::string Deserialize() const {
    std::stringstream result;
    std::queue<TreeNode*> q1;
    std::queue<int32_t> q2;
    q1.push(root_);
    q2.push(0);
    result << "{ ";
    TreeNode* parent = nullptr;
    int32_t now_depth = 0;
    while (!q1.empty()) {
      TreeNode* t = q1.front();
      int32_t depth = q2.front();
      q1.pop();
      q2.pop();
      if (t->GetParent() != parent) {
        parent = t->GetParent();
        result << "}";
        if (depth > now_depth) {
          result << "\n";
          now_depth = depth;
        }
        result << "{ " << parent->GetValue() << " ; ";
      }
      result << t->GetValue() << " ";
      for (TreeNode* child : t->GetChildren()) {
        q1.push(child);
        q2.push(now_depth + 1);
      }
    }
    result << "}";
    return result.str();
  }

  virtual ~Tree() {
    delete root_;
  }

 private:
  TreeNode* root_;
};

Tree BuildGrammarTree(std::string gremlin);
#endif  // EULER_PARSER_TREE_H_
