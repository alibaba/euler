# Load build-in dataset
You can load tf_euler build-in dataset in python.

Datasets:
- cora
- pubmed
- citeseer
- ppi
- reddit
- fb15k

Examples:
```python
import tf_euler
ppi = tf_euler.dataset.get_dataset('ppi')
ppi.load_graph()
'''

