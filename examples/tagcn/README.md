Topology Adaptive Graph Convolutional Networks(TAGCN)
============

Paper link: [Topology Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1710.10370)

Run
-------
```python
python run_tagcn.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | cora           | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| layers            | 2              | GCN layer number                |
| batch_size        | 32             | running batch size              |
| num_epochs        | 15             | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| run_mode          | train          | train/evaluate                  |

Result
------
| Dataset | F1 |
| ---------- | ------------------ |
| cora       | 0.817              |
| pubmed     | 0.867              |
| citeseer   | 0.727              |

