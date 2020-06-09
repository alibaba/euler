Graph Attention Networks (GAT)
============

Paper link: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)

Run
-------
```python
python run_gat.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | cora           | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| layers            | 2              | GCN layer number                |
| batch_size        | 32             | running batch size              |
| num_epochs        | 10             | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| run_mode          | train          | train/evaluate                  |
| concat            | True           | attention concat                |
| improved          | True           | attention improved              |
| num_heads         | 1              | attention head number           |

Result
------
| Dataset | F1 |
| ---------- | ------------------ |
| cora       | 0.823              |
| pubmed     | 0.876              |
| citeseer   | 0.755              |

