GatedGraph
============

Paper link: [GATED GRAPH SEQUENCE NEURAL NETWORKS](https://arxiv.org/pdf/1511.05493.pdf)

Run
-------
```python
python run_gatedgraph.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | mutag          | dataset                         |
| hidden_dim        | 32             | hidden dimension                |
| layers            | 2              | GCN layer number                |
| process_steps     | 2              | LSTM processing steps           |
| lstm_layers       | 2              | LSTM layer number               |
| num_epochs        | 1000            | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| run_mode          | train          | train/evaluate                  |

Result
------
| Dataset | Accuracy |
| ---------- | ------------------ |
| mutag      | 0.920              |

