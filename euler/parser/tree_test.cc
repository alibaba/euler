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

#include <iostream>

#include "gtest/gtest.h"

#include "euler/parser/tree.h"
namespace euler {

TEST(TreeTest, PostTraversalTest) {
  TreeNode* n0 = new TreeNode("0");
  Tree t(n0);
  TreeNode* n1 = new TreeNode("1");
  TreeNode* n2 = new TreeNode("2");
  TreeNode* n3 = new TreeNode("3");
  TreeNode* n4 = new TreeNode("4");
  TreeNode* n5 = new TreeNode("5");
  TreeNode* n6 = new TreeNode("6");
  TreeNode* n7 = new TreeNode("7");
  /*
       0
     /   \
    1     2
    |\  / | \
    3 4 5 6 7
  */
  t.AddNode(n0, n1);
  t.AddNode(n0, n2);
  t.AddNode(n1, n3);
  t.AddNode(n1, n4);
  t.AddNode(n2, n5);
  t.AddNode(n2, n6);
  t.AddNode(n2, n7);
  // t.DeleteNode(n1);
  std::vector<TreeNode*> results;
  t.PostTraversal(n0, &results);

  std::vector<std::string> expct =
      {"3", "4", "1", "5", "6", "7", "2", "0"};
  ASSERT_EQ(results.size(), expct.size());
  for (size_t i = 0; i < expct.size(); ++i) {
    ASSERT_EQ(results[i]->GetValue(), expct[i]);
  }
  std::cout << "=================" << std::endl;
  std::cout << t.Deserialize() << std::endl;
}

}  // namespace euler
