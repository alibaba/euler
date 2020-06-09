Graph Isomorphism Network (GIN)
============

Paper link: [HOW POWERFUL ARE GRAPH NEURAL NETWORKS?](https://arxiv.org/pdf/1810.00826.pdf)

Run
-------
```python
python run_gin.py [--optional_params=params]
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
| train_eps         | True           | Whether ope train as variable   |
| eps               | 0.0            | eps initialize                  |

Result
------
| Dataset | Accuracy |
| ---------- | ------------------ |
| mutag      | 0.923              |

