Genipath
============

Paper link: [GeniePath: Graph Neural Networks with Adaptive Receptive Paths](https://arxiv.org/pdf/1802.00910.pdf)

Run
-------
```python
python run_genipath.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | cora           | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| embedding_dim     | 32             | embedding dimension             |
| use_id            | False          | Whether use id embedding        |
| layers            | 2              | GCN layer number                |
| batch_size        | 32             | running batch size              |
| num_epochs        | 10             | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| run_mode          | train          | train/evaluate                  |

Result
------
| Dataset | F1 |
| ---------- | ------------------ |
| cora       | 0.742              |
| pubmed     | 0.872              |
| citeseer   | 0.735              |

