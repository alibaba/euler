Graph Convolutional Networks (GCN)
============

Paper link: [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)

Run
-------
```python
python run_deepwalk.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | cora           | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| embedding_dim     | 32             | embedding dimension             |
| walk_len          | 3              | Length of random walk           |
| p                 | 1              | DeepWalk return parameter       |
| q                 | 1              | DeepWalk in-out parameter       |
| left_win_size     | 1              | Left window size                |
| right_win_size    | 1              | Right window size               |
| num_negs          | 5              | Negative sample number          |
| use_id            | True           | Whether use id embedding        |
| batch_size        | 32             | running batch size              |
| num_epochs        | 10             | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| run_mode          | train          | train/evaluate                  |

Result
------
| Dataset | mrr |
| ---------- | ------------------ |
| cora       | 0.905              |
| pubmed     | 0.983              |
| citeseer   | 0.976              |

