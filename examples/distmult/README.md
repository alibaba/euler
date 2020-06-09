Embedding Entities and Relations for Learning and Inference in Knowledge Bases(DistMult)
============

Paper link: [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575)

Run
-------
```python
python run_distmult.py [--optional_params=params]
```

Params:

| Parameter Name | Default | Note |
| ----------------- | -------------- | ------------------------------- |
| dataset           | fb15k          |                                 |
| embedding_dim     | 100            | entity/relation hidden dimension |
| num_negs          | 1              | Number of negative samplings    |
| corrupt           | both           | both/front/tail                 |
| margin            | 1.0            | Margin of loss                  |
| metric_name       | mrr            | mrr/mr/hit10                    |
| batch_size        | 128            | Mini-batch size                 |
| l2_regular        | True           | if use l2 regular               |
| regular_param     | 0.0001         | l2 regular param                |
| num_epochs        | 500            | epochs to run                   |
| log_steps         | 20             | log per steps                   |
| model_dir         | ckpt           | checkpoint dir                  |
| learning_rate     | 0.01           | run learning rate               |
| optimizer         | adam           | run optimizer algorithm         |
| run_mode          | train          | train/evaluate/infer            |
| infer_type        | edge           | edge_node_src/node_dst          |

