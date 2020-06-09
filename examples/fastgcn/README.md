Fast Learning with Graph Convolutional Networks (FastGCN)
============

Paper link: [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247)

Run
-------
```python
python run_fastgcn.py [--optional_params=params]
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
| fanouts           | 400,400        | total neighbor per layer        |

Result
------
| Dataset | F1 |
| ---------- | ------------------ |
| cora       | 0.803              |
| pubmed     | 0.860              |
| citeseer   | 0.740              |

