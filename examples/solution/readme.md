## SuperviseSolution

有监督训练解决方案，可供用户定义配置：

- label获取方式(get_label_fn)

- encoder编码方式(encoder_fn)
- logits函数(logit_fn)
- 损失函数(loss_fn)
- metric(metric_name)

### get_label_fn

输入：node_id (Tensor, shape: [batch_size])
输出：labels (Tensor, shape: [batch_size, label_dim])

提供的组件：

- tf_euler.solution.GetLabelFromFea(label_idx, label_dim)：从Euler图数据的label_idx中获取dim为label_dim的label。

------

### encoder_fn

输入：node_id (Tensor, shape: [batch_size])

输出：node_embedding (Tensor, shape: [batch_size, embedding_dim])

提供的组件：

- tf_euler.models.[model_name].GNN：Euler实现的算法模型的编码方式，具体参数参考examples
- tf_euler.python.mp_utils.base_gnn.BaseGNNNet：自定义的Embedding类，具体实现方式参考Message Passing接口

------

### logit_fn

输入：embedding (Tensor, shape: [batch_size, embedding_dim])

输出：logits (Tensor, shape: [batch_size, logit_dim])

提供的组件：

- tf_euler.solution.DenseLogits(logit_dim)：经过一个dim为logit_dim的全连接网络输出结果

------

###  loss_fn:

输入：

- label (Tensor, shape: [batch_size, label_dim])
- logit (Tensor, shape: [batch_size, logit_dim])

输出：loss (Tensor, shape: scalar)

提供的默认组件：

- tf_euler.python.solution.losses.sigmoid_loss：计算sigmoid loss

------

### metric_name

- f1: F1-Score
- auc: AUC
- acc: Accuracy

------

## UnsuperviseSolution

无监督训练解决方案，可供用户定义配置：

- target_encoder编码方式(target_encoder_fn)
- context_encoder编码方式(context_encoder_fn)
- 
- logits函数(logit_fn)
- 损失函数(loss_fn)
- metric(metric_name)

------

### target_encoder_fn/context_encoder_fn

输入：node_id (Tensor, shape: [batch_size])

输出：node_embedding (Tensor, shape: [batch_size, embedding_dim])

提供的组件：

- tf_euler.models.[model_name].GNN：Euler实现的算法模型的编码方式，具体参数参考examples
- tf_euler.python.mp_utils.base_gnn.BaseGNNNet：自定义的Embedding类，具体实现方式参考Message Passing接口

------

### pos_sample_fn

输入：src_node (Tensor, shape: [batch_size])

输出：pos_node (Tensor, shape: [batch_size, pos_num])

提供的组件：

- tf_euler.solution.SamplePosWithTypes(pos_edge_type, num_pos=1, max_id=-1): 采样num_pos个pos_edge_type的邻居，若没有则用max_id+1代替

------

### neg_sample_fn

输入：src_node (Tensor, shape: [batch_size])

输出：neg_node (Tensor, shape: [batch_size, neg_num])

提供的组件：

- tf_euler.solution.SampleNegWithTypes(neg_type, num_negs=5): 采样num_negs个neg_type的node

------

### logit_fn

输入：

- src_embedding (Tensor, shape: [batch_size, 1, embedding_dim])
- pos_embedding (Tensor, shape: [batch_size, pos_num, embedding_dim])
- sneg_embedding (Tensor, shape: [batch_size, neg_num, embedding_dim])

输出：

- logits (Tensor, shape: [batch_size, pos_num, logit_dim])
- neg_logits (Tensor, shape: [batch_size, neg_num, logit_dim])

提供的组件：

- tf_euler.solution.PosNegLogits()：pos/neg分别与src的embedding点乘得到的结果

------

###  loss_fn:

输入：

- logit (Tensor, shape: [batch_size, pos_num, label_dim])
- neg_logit (Tensor, shape: [batch_size, neg_num, logit_dim])

输出：loss (Tensor, shape: scalar)

提供的默认组件：

- tf_euler.python.solution.losses.xent_loss：计算xent loss

------

### metric_name

- f1: F1-Score
- auc: AUC
- acc: Accuracy
