# EULER Estimator
Euler Estimator对tensorflow的Estimator进行了封装，为Graph Node Classification / Graph Classification / Knowledge Graph Link Prediction提供了方便的分布式实验接口。

## 模型编写
Euler Estimator的运行共需要四部分的配置：
1. Euler graph的构建
2. 算法模型的构建
3. 分布式配置/训练参数配置
4. Estimator创建


## 训练参数配置
### 通用配置项
| 配置项                   | 作用                                   |
| ------------------------ | -------------------------------------- |
| learning_rate            | 学习率                                 |
| log_steps                | 打印log信息的间隔                      |
| model_dir                | 模型checkpiont存储位置                 |
| infer_dir                | 模型预测的输出目录                     |
| total_step               | 训练的总step数量                       |

### NodeEstimator特殊配置项
| 配置项                   | 作用                                   |
| ------------------------ | -------------------------------------- |
| train_node_type          | 训练的采样node type                    |
| batch_size               | 训练和预测的batch_size                 |
| id_file                  | 预测node idx的文件地址                 |

### GraphEstimator特殊配置
| 配置项                   | 作用                                             |
| ------------------------ | ------------------------------------------------ |
| graph_file               | 训练/预测时含有[graph_idx,graph_label]的文件     |
| graph_size               | 训练/预测时的graph总数目                         |
| node_file                | 训练/预测时含有[node_idx,node所属graphidx]的文件 |
| node_size                | 训练/预测时的node总数目                          |
| num_classes              | 图分类的分类个数                                 |

### EdgeEstimator特殊配置
| 配置项                   | 作用                                                |
| ------------------------ | --------------------------------------------------- |
| train_edge_type          | 训练的采样edge type                                 |
| batch_size               | 训练和预测的batch_size                              |
| id_file                  | 预测edge信息的文件地址                              |
| infer_type               | 预测输出的embedding类型[node_src/node_dst/edge]可选 |

## 运行
- train(): 根据input_fn开始训练
- evaluate(): 根据input_fn开始验证，输出验证信息
- infer(): 根据input_fn开始infer embedding，结果通过pickle写入到infer_dir中(infer 结果为{'idx', 'embedding'}的字典)
- train_and_evalute(): 根据os.enviro['TF_CONFIG']配置，开始分布式训练或验证

## 运行示例
### 单机运行

```python
tf.logging.set_verbosity(tf.logging.INFO)

graph_dir = 'your euler graph dir'
datatype = 'node'  # 'node' or 'all'
tf_euler.initialize_embedded_graph(graph_dir, data_type=datatype)

# 算法模型的构建，可以使用example中的算法模型，自行配置各个参数
model_cls = your_model(model_params)

# 训练参数配置，提供必要的训练参数，具体所需参数，已经在上文说明
params = estimator_params
# Estimator的创建与训练/验证/预测
config = tf.estimator.RunConfig(log_step_count_steps=None)
base_estimator = NodeEstimator(model_cls, params, config)
base_estimator.train()
# base_estimator.evaluate()
# base_estimator.infer()
```

### 分布式运行
初始化Euler
```python
euler.start(
    directory='euler_graph_dir',
    shard_idx=shard_idx,
    shard_num=shard_num,
    zk_addr=zk_addr,
    zk_path=zk_path,
    module=euler.Module.DEFAULT_MODULE)
```

运行Estimator
```python
# 分布式参数配置
# 参考https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig进行配置
'''
cluster = {'chief': ['host0:2222'],
           'ps': ['host1:2222', 'host2:2222'],
           'worker': ['host3:2222', 'host4:2222']}
task_type = 'worker'
task_id = 0
os.environ['TF_CONFIG'] = json.dumps(
    {'cluster': cluster,
     'task': {'type': task_type, 'index': task_id}})
})
'''
if not task_type == 'ps':
    tf_euler.initialize_graph({
        'mode': 'remote',
        'zk_server': zk_addr,
        'zk_path': zk_path,
        'shard_num': shard_num,
        'num_retries': 1
    })

tf.logging.set_verbosity(tf.logging.INFO)

# 算法模型的构建，可以使用example中的算法模型，自行配置各个参数
model_cls = your_model(model_params)

# 训练参数配置，提供必要的训练参数，具体所需参数，已在上文说明
params = estimator_training_params

# Estimator的创建与训练/验证/预测
config = tf.estimator.RunConfig(log_step_count_steps=None)
base_estimator = NodeEstimator(model_cls, params, config)
base_estimator.train_and_evaluate()
```

*注：由于Euler Estimator中添加了自己的logger hook，建议将tf.estimator.RunConfig中log_step_count_steps设置为None，避免log信息混乱
