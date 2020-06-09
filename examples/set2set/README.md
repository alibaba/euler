Set2Set
============

Paper link: [ORDER MATTERS: SEQUENCE TO SEQUENCE FOR SETS](https://arxiv.org/pdf/1511.06391.pdf)

Run
-------
```python
python run_set2set.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | mutag          | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| layers            | 5              | GCN layer number                |
| process_steps     | 4              | LSTM processing steps           |
| lstm_layers       | 2              | LSTM layer number               |
| num_epochs        | 500            | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| run_mode          | train          | train/evaluate                  |

Result
------
| Dataset | Accuracy |
| ---------- | ------------------ |
| mutag      | 0.901              |

