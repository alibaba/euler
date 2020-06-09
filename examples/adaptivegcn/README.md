Adaptive Graph Convolutional Neural Networks (AdaptiveGCN)
============

Paper link: [Adaptive Graph Convolutional Neural Networks](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/download/16642/16554)

Run
-------
```python
python run_adaptivegcn.py [--optional_params=params]
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
| cora       | 0.821              |
| pubmed     | 0.859              |
| citeseer   | 0.751              |

