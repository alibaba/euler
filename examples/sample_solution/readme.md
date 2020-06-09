# Sample Solution

有样本输入的训练解决方案的示例，展示如何将带有label的样本数据和图学习相结合。sample_solution_model.py文件给出了如何通过graphsage的组件通过Solution解决方案，拼装一个graphsage模型。
Sample Solution需要一个样本文件，其中包含多行样本数据，每一行为一个样本，每一个样本的格式为以逗号分隔的label,node1,node2,...
示例代码中以cora图数据为例创建模型并运行，数据没有任何意义，需要根据数据的实际需要创建数据样本及模型。

## SuperviseSampleSolution

从样本读入数据的有监督训练方案（eg: Graphsage），可供用户定义配置：

- 输入数据解析函数(parse_input_fn)：解析tf任务中，reader读出的数据
- encoder编码方式(encoder_fn)
- embedding解析函数(parse_group_emb_fn)：解析encoder_fn输出的embedding
- 负采样函数(neg_sample_fn)：可选。在需要拓展负例个数时的负采样方式
- logits函数(logit_fn)
- 损失函数(loss_fn)
- metric(metric_name)

### parse_input_fn

输入：inputs (样本文件每一条数据tf.string_split的输出)

输出：input_list (Tensor list): 包括

- 0: label  (Tensor, shape: [batch_size, label_dim])
- 1~n: node_group (Tensor)

------

### encoder_fn

输入：group_node_id (Tensor list)

输出：group_node_embedding (Tensor list)

提供的组件：

- tf_euler.python.mp_utils.group_gnn.SharedGNNNet：group之间共享卷积参数的embedding方式，具体使用参考example
- tf_euler.python.mp_utils.group_gnn.GroupGNNNet：group之间不共享卷积参数的embedding方式，具体使用参考example
- example中各个模型的GNNNet

------

### parse_group_emb_fn

输入：group_node_embedding (Tensor list)

输出：

- target_embedding (Tensor, shape: [batch_size, embedding_dim]): 目标node的embedding
- context_embedding (Tensor, shape: [batch_size, embedding_dim]): context node的embedding
- out_embedding (Tensor, shape: [batch_size, embedding_dim]): infer过程中输出的embedding

------

### neg_sample_fn(可选)

输入：src_node (Tensor, shape: [batch_size])

输出：neg_node (Tensor, shape: [batch_size, neg_nums])

------

### logit_fn

输入：

- target_embedding (Tensor, shape: [batch_size, embedding_dim])
- context_embedding (Tensor, shape: [batch_size, embedding_dim])

输出：logits (Tensor, shape: [batch_size, logit_dim])

提供的组件：

- tf_euler.solution.CosineLogits()：target_emb context_emb 计算cosine距离的结果

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

## UnsuperviseSampleSolution

从样本读入数据的无监督训练方案，除了标准无监督解决方案外，还可供用户定义配置：

- 输入数据解析函数(parse_input_fn)：解析tf任务中，reader读出的数据

### parse_input_fn

输入：inputs (Reader Tensor)

输出：input_list (Tensor list): 包括

- 0: src_node  (Tensor, shape: [batch_size])
- 1: neg_node (Tensor, shape: [batch_size, num_neg]) （可选）
- 2: pos_node (Tensor, shape: [batch_size, num_pos]) （可选）

*注*：输出list size可为1/2/3，为1时只有src，为2时只有src和neg，为3时为src neg pos。若只需要src和pos，可返回[src, None, pos]的list
