TransE / TransH / TransR / TransD
======

Paper link: 
[Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)

[Knowledge Graph Embedding by Translating on Hyperplanes](https://pdfs.semanticscholar.org/2a3f/862199883ceff5e3c74126f0c80770653e05.pdf)

[Learning Entity and Relation Embeddings forKnowledge Graph Completion](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewFile/9571/9523)

[Knowledge Graph Embedding via Dynamic Mapping Matrix](https://aclweb.org/anthology/P15-1067)

Run
------
```python
python run_transE.py [--optional_params=params]
```
Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | fb15k          |                                 |
| embedding_dim(transE/H/D) | 100    | entity/relation hidden dimension |
| entity_embedding_dim(transR) | 100 | entity hidden dimension          |
| relation_embedding_dim(transR) | 100 | relation hidden dimension      |
| num_negs          | 1              | Number of negative samplings    |
| corrupt           | both           | both/front/tail                 |
| margin            | 1.0            | Margin of loss                  |
| L1                | False           | Use l1 distance for score      |
| metric_name       | mrr            | mrr/mr/hit10                    |
| batch_size        | 128            | Mini-batch size                 |
| num_epochs        | 500            | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| run_mode          | train          | train/evaluate/infer            |
| optimizer         | adam           | run optimizer algorithm         |
| infer_type        | edge           | edge_node_src/node_dst          |

Result
------
We list the result of various methods implemented by ourselves in dateset FB15k.

| Model | MeanRank(paper) | MeanRank(ours) | Hit@10(paper) | Hit@10(ours) |
| :------: | :------:| :------: | :------: |:------: |
|TransE|243|197|34.9|39.7|
|TransH|211|179|42.5|45.4|
|TransR|211|191|43.8|46.1|
|TransD|226|163|49.4|51.3|
* The configurations are: learning rate=0.005, margin=0.5, dim=50, batch_size=1200

We compare the running time per epoch (batch_size=100) with OpenKE(Tensorflow-TransX).

| Model | Time OpenKE(s) | Time Ours(s) |
| :------: | :------: | :------: |
|TransE|11.92|9.36|
|TransH|17.12|11.87|
|TransR|31.32|26.30|
|TransD|15.11|11.71|

* CPU : Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GHz * 8

