# 概述
图是表达能力很强的通用数据结构，可以用来刻画现实世界中的很多问题。图神经网络等基于图的学习方法在很多领域取得了非常好的效果。

Euler是大规模分布式的图学习框架，配合TensorFlow或者阿里开源的XDL等深度学习工具，它支持用户在数十亿点数百亿边的复杂异构图上进行模型训练。有关Euler系统的详细介绍请参见[系统介绍](https://github.com/alibaba/euler/wiki/系统介绍)。

Euler中已经实现了近10中图学习算法，这些算法的详情请见[论文列表](https://github.com/alibaba/euler/wiki/论文列表)。

# 使用文档

- 基础教程
  - [编译安装](https://github.com/alibaba/euler/wiki/编译安装)
  - [快速开始](https://github.com/alibaba/euler/wiki/快速开始)
  - [数据准备](https://github.com/alibaba/euler/wiki/数据准备)
  - [使用指南](https://github.com/alibaba/euler/wiki/使用指南)
- 进阶应用
  - [模型编写](https://github.com/alibaba/euler/wiki/模型编写)
  - [集群使用](https://github.com/alibaba/euler/wiki/集群使用)
- 详细接口
  - [Euler OP (based on TensorFlow)](https://github.com/alibaba/euler/wiki/Euler-OP)
  - [Euler Model Zoo (based on TensorFlow)](https://github.com/alibaba/euler/wiki/Euler-Model)
  - [Euler C++ API](https://github.com/alibaba/euler/wiki/CPP接口)
- 基准测试
  - [效果测试](https://github.com/alibaba/euler/wiki/效果测试)
  - [性能测试](https://github.com/alibaba/euler/wiki/性能测试)
  - [ScalableGCN](https://github.com/alibaba/euler/wiki/ScalableGCN)

# 联系我们
如果有任何问题，请直接提交issues，也欢迎通过Euler开源技术支持邮件组（[euler-opensource@list.alibaba-inc.com](mailto:euler-opensource@list.alibaba-inc.com)）联系我们。

# License

Euler使用[Apache-2.0](LICENSE)许可

# 致谢

Euler由阿里妈妈工程平台团队与搜索广告算法团队共同探讨与开发，也获得了阿里妈妈多个团队的大力支持。同时也特别感谢蚂蚁金服的机器学习团队，项目早期的一些技术交流，对我们帮助很大。
