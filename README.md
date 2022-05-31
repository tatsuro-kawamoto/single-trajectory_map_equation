# Single-trajectory map equation

This is a set of python codes for the single-trajectory map equation [1], which is a variant of the map equation [2]. 

Please check out `Example.ipynb` for a few examples. 

## Prerequisites
- numpy
- infomap


## Community detection for graphs/trajectories (Infomap+)

This is a simple greedy heuristic [1] that minimizes the average code length of trajectories. 
This function uses Infomap [3] as the initial partition of the nodes by default. 

### Usage
```
im = Infomap_st(trajectories)
membership = im.optimize()
```    
`trajectories` will be the edgelist when the input is a graph.

### Output
- `membership`: The optimal partition.


### Input parameters

| Parameter          | Default        | Description                                                                                                 | 
| ------------------ | -------------- | ----------------------------------------------------------------------------------------------------------- | 
| trajectories       | Required input | A list of lists. Each list represents a trajectory (list of visited nodes).                               | 
| membership       | `None` | A list inditating the group membership of each node (node partition). If `None`, Infomap with "--two-level -f rawdir" will be used as the initial partition.                               | 
| coding | `'lower_bound'` | Coding scheme: Huffman coding (`'Huffman'`), Shannon-Fano coding (`'Shannon-Fano'`), or the lower bound of the code length based on Shannon's source codign theorem (`'lower_bound'`).                                           | 
| init_module           | `True`             | If `False`, the codeword for the initial module in a trajectory is ignored.   | 
| init_node           | `True`             | If `False`, the codeword for the initial node in a trajectory is ignored.   | 
| min_size           | 3             | Returns an alert when the smallest module size is less than this value   | 
| lmbda           | 1             | The resolution parameter | 



## References
[1] Tatsuro Kawamoto, "Single-trajectory map equation," [arXiv:2203.04044](https://arxiv.org/abs/2203.04044).

[2] Martin Rosvall and Carl T. Bergstrom, "Maps of random walks on complex networks reveal community structure," [Proc. Natl. Acad. Sci. U. S. A. (2008)](https://www.pnas.org/doi/full/10.1073/pnas.0706851105).

[3] [https://mapequation.github.io/infomap/python/](https://mapequation.github.io/infomap/python/)

## Citation
Please cite [1].