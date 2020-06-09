Graph Auto Encoder / Variational Graph Auto Encoder (GAE VGAE)
============

Paper link: [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308)

Run
-------
```python
python run_gae.py [--optional_params=params]
python run_vgae.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | cora           | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| layers            | 2              | GCN layer number                |
| batch_size        | 1024           | running batch size              |
| num_epochs        | 2000           | epochs to run                   |
| log_steps         | 10             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| run_mode          | train          | train/evaluate                  |

GAE Result
------
| Dataset | acc |
| ---------- | ------------------ |
| cora       | 0.71               |

VGAE Result
------
| Dataset | acc |
| ---------- | ------------------ |
| cora       | 0.79               |
