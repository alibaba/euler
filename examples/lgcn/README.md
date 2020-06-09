Large-Scale Learnable Graph Convolutional Networks(LGCN)
============

Paper link: [Large-Scale Learnable Graph Convolutional Networks](https://arxiv.org/pdf/1808.03965.pdf)

Run
-------
```python
python run_lgcn.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | cora           | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| k                 | 3              | LGCN top k                      |
| num_neighbor      | 10              | number of neighbor              |
| batch_size        | 32             | running batch size              |
| num_epochs        | 10             | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| run_mode          | train          | train/evaluate                  |

Result
------
| Dataset | F1 |
| ---------- | ------------------ |
| cora       | 0.641              |
| pubmed     | 0.848              |
| citeseer   | 0.675              |

