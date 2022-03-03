# Single-trajectory map equation

This is a set of python codes for the single-trajectory map equation [1], which is a variant of the map equation [2]. 

Please check out `Example.ipynb` for a simple example. 

## Prerequisites
- numpy
- infomap


## Agglomerative heuristic for community detection

```
code_length, module_assignments = Infomap_st(
trajectories, 
vertices, 
scheme,
init_module, 
init_node,
deterministic,
n_trials,
n_itr)
```    

This is a simple agglomerative heuristic that minimizes the average description length of the trajectories. 
This function uses Infomap [3] as the initial partition of the nodes. 


### Outputs
- `code_length`: Average code length 
- `module_assignments`: Optimal module assignments


### Input parameters

| Parameter          | Default        | Description                                                                                                 | 
| ------------------ | -------------- | ----------------------------------------------------------------------------------------------------------- | 
| trajectories       | Required input | A list of lists. Each list represents a trajectory (list of visited nodes).                               | 
| vertices       | `None` | A list of node indices. If `None`, it is assumed that the node indices are a set of consecutive numbers starting from zero.                               | 
| scheme | `'Huffman'` | Coding scheme: Huffman coding (`'Huffman'`), Shannon-Fano coding (`'Shannon-Fano'`), or the lower bound of the code length based on Shannon's source codign theorem (`'lower_bound'`).                                           | 
| init_module           | `True`             | If `False`, the codeword for the initial module in a trajectory is ignored.   | 
| init_node           | `True`             | If `False`, the codeword for the initial node in a trajectory is ignored.   | 
| deterministic           | `False`             | If `True`, the algorithm merges modules deterministically from small modules.   | 
| n_trials           | 10             | Number of trials of the agglomerative heuristic. | 
| n_itr              | 1000           | Max. number of times that the algorithm merges modules.                                                     | 



## References
[1] Tatsuro Kawamoto, "Single-trajectory map equation," arxiv:***.

[2] Martin Rosvall and Carl T. Bergstrom, "Maps of random walks on complex networks reveal community structure," Proc. Natl. Acad. Sci. U. S. A. (2008).

[3] https://mapequation.github.io/infomap/python/

## Citation
Please cite [1].