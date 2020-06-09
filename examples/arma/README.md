ARMA
============

Paper link: [Graph Neural Networks with convolutional ARMA filters](https://arxiv.org/abs/1901.01343)

Run
-------
```python
python run_arma.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | cora           | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| layers            | 1              | GCN layer number                |
| batch_size        | 32             | running batch size              |
| num_epochs        | 10             | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| run_mode          | train          | train/evaluate                  |
| K                 | 2              | ARMA K parameter                |

Result
------
| Dataset | F1 |
| ---------- | ------------------ |
| cora       | 0.822              |
| pubmed     | 0.880              |
| citeseer   | 0.755              |

