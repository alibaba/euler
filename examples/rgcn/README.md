Modeling Relational Data with Graph Convolutional Networks (RGCN)
============

Paper link: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)

Run
-------
```python
python run_rgcn.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | wn18           | wn18/fb15/fb15-237/
       |
| hidden_dim        | 32             | hidden dimension                |
| embedding_dim     | 32             | node embedding dimension        |
| num_negs          | 5              | negative sample  number         |
| layers            | 2              | RGCN layer number               |
| batch_size        | 32             | running batch size              |
| num_epochs        | 10             | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| metric            | mrr            | mrr/mr/hit1/hit3/hit10          |
| run_mode          | train          | train/evaluate                  |
