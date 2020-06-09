Dynamic Neighborhood Aggregation(DNA)
============

Paper link: [Just Jump: Dynamic Neighborhood Aggregation in Graph Neural Networks
](https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1904.04849)

Run
-------
```python
python run_dna.py [--optional_params=params]
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
| group_num         | 8              | DNA group number                |
| head_num          | 1              | attension head number           |

Result
------
| Dataset | F1 |
| ---------- | ------------------ |
| cora       | 0.811              |
| pubmed     | 0.867              |
| citeseer   | 0.710              |

