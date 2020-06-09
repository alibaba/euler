LINE
============

Paper link: [Line: Large-scale information network embedding](https://arxiv.org/abs/1503.03578)

Run
-------
```python
python run_line.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | cora           | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| embedding_dim     | 32             | embedding dimension             |
| order             | 2              | line order                      |
| num_negs          | 5              | Negative sample number          |
| use_id            | True           | Whether use id embedding        |
| batch_size        | 32             | running batch size              |
| num_epochs        | 10             | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| run_mode          | train          | train/evaluate                  |

Result
------
| Dataset | mrr |
| ---------- | ------------------ |
| cora       | 0.900              |
| pubmed     | 0.987              |
| citeseer   | 0.956              |

