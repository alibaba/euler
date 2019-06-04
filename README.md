[中文版](https://github.com/alibaba/euler/wiki/HOME-CN)

# Overview

Graph is a generic data structure with strong expressivity to characterize entities and their relationships in real world. Graph learning algorithms, such as graph neural networks (GNN), have been shown to be tremendously beneficial to many applications in both academia and industry.

Euler is a large-scale distributed graph learning system. It can work with deep learning tools like TensorFlow or X-Deep Learning and support users to train models on very complex heterogeneous graphs with billions of nodes and tens of billions of edges. For a more detailed introduction to the Euler system, please refer to [System Introduction](https://github.com/alibaba/euler/wiki/System-Introduction).

# Usage Documents
In *Tutorials*, we guide users to get started quickly with Euler. Then, in *Advanced Usage*, we show how to write a customized model and how to achieve distributed training.

Next, we introduce Euler's *Programming Interfaces*: the ready-made model package based on TensorFlow which contains a bunch of state-of-the-art graph learning models, the graph operators to be used with TensorFlow, and the C++ API for the graph engine (for users who want to adapt to other deep learning frameworks).

In *Model Introduction*, we give a brief introduction for our in-house algorithms and the links for other open-released models. In *Evaluation*, we provide effectiveness and efficiency evaluation of our system.

- Tutorials
  - [Installation](https://github.com/alibaba/euler/wiki/Installation)
  - [Getting started](https://github.com/alibaba/euler/wiki/Getting-Started)
  - [Preparing the data](https://github.com/alibaba/euler/wiki/Preparing-Data)
  - [User manual](https://github.com/alibaba/euler/wiki/User-Manual)
- Advanced Usage
  - [Writing a model](https://github.com/alibaba/euler/wiki/Writing-Model)
  - [Distributed training](https://github.com/alibaba/euler/wiki/Distributed-Training)
- Programming Interfaces
  - [Euler OP (based on TensorFlow)](https://github.com/alibaba/euler/wiki/Euler-OP-En)
  - [Euler Model Zoo (based on TensorFlow)](https://github.com/alibaba/euler/wiki/Euler-Model-En)
  - [Euler C++ API](https://github.com/alibaba/euler/wiki/CPP-API)
- Model Introduction
  - [LsHNE](https://github.com/alibaba/euler/wiki/LsHNE)
  - [LasGNN](https://github.com/alibaba/euler/wiki/LasGNN)
  - [ScalableGCN](https://github.com/alibaba/euler/wiki/ScalableGCN)
  - [Paper list](https://github.com/alibaba/euler/wiki/paper-list)
- Evaluation
  - [Effectiveness](https://github.com/alibaba/euler/wiki/Effectiveness)
  - [Efficiency](https://github.com/alibaba/euler/wiki/Efficiency)

# Contact Us
If you have any questions, please submit [issues](https://github.com/alibaba/euler/issues) or send mails to [euler-opensource@list.alibaba-inc.com](mailto:euler-opensource@list.alibaba-inc.com).

# License
Euler uses [Apache License 2.0](https://github.com/alibaba/euler/blob/master/LICENSE).

# Acknowledgement
Euler is developed collaboratively by Alimama Engineering Platform Team and Alimama Search Advertising Algorithm Team. We are very grateful to the support of other teams in Alimama and to the Ant Financial Services Group's Machine Learning Team for their help in the early technical exchange stage of the project.
