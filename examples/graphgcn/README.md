Higher-order Graph Neural Networks
============

Paper link: [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://arxiv.org/pdf/1810.02244.pdf)

Run
-------
```python
python run_graphgcn.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | mutag           | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| layers            | 5              | GCN layer number                |
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
| mutag      | 0.891              |

