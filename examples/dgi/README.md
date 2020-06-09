Deep Graph Infomax (DGI)
============

Paper link: [Deep Graph Infomax](https://arxiv.org/abs/1809.10341)
Author's code repo (in Pytorch): [https://github.com/PetarV-/DGI](https://github.com/PetarV-/DGI)

Run
-------
```python
python run_dgi.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | ppi            | cora/pubmed/citeseer/ppi/reddit |
| hidden_dim        | 32             | hidden dimension                |
| layers            | 2              | SAGE convolution layer number   |
| fanouts           | [10, 10]       | GraphSage fanouts               |
| batch_size        | 32             | mini  batch size                |
| num_epochs        | 20             | epochs to run                   |
| num_negs          | 5              | negative sample number          |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| metric            | mrr            | mrr/mr/hit10                    |
| run_mode          | train          | train/evaluate                  |

