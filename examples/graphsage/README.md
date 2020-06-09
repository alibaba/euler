Inductive Representation Learning on Large Graphs (GraphSAGE)
============

Paper link: [Inductive Representation Learning on Large Graphs](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)

Run
-------
```python
python run_graphsage.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | cora           | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| layers            | 2             | GCN layer number                |
| batch_size        | 32            | running batch size              |
| num_epochs        | 10            | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| run_mode          | train          | train/evaluate                  |
| fanouts           | 10,10          | sample neighbor number per layer |

Result
------
| Dataset | F1 |
| ---------- | ------------------ |
| cora       | 0.774              |
| pubmed     | 0.884              |
| citeseer   | 0.731              |

