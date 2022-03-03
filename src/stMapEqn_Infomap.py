import numpy as np
import src.coding.codelength as codelength
import random
import collections
from infomap import Infomap

def merge_modules(trajectories, module_assignments, scheme="Huffman", init_module=True, init_node=True, deterministic=False, n_trials=10, n_itr=1000):
	def smallest_module_pair(module_assignments_opt):
		module_histogram = collections.Counter(module_assignments_opt).most_common()
		if len(module_histogram) < 2:
			multiple_modules = False
			module1 = None
			module2 = None
		else:
			k_smallest = module_histogram.pop()
			k_next_smallest = module_histogram.pop()
			multiple_modules = True
			module1 = k_smallest[0]
			module2 = k_next_smallest[0]
		return module1, module2, multiple_modules
	
	MCL = codelength.AverageCodeLength(module_assignments, trajectories=trajectories, scheme=scheme, init_module=init_module, init_node=init_node)
	module_assignments_opt = module_assignments.copy()

	if deterministic == True: 
	# Deterministically merge smallest module pair
		merged_assignments = module_assignments.copy()
		while True:
			module1, module2, multiple_modules = smallest_module_pair(merged_assignments)
			if multiple_modules == False:
				break
			merged_assignments = [module1 if v_module == module2 else v_module for v_module in merged_assignments]
			ACL = codelength.AverageCodeLength(merged_assignments, trajectories=trajectories, scheme=scheme, init_module=init_module, init_node=init_node)
			if ACL < MCL:
				MCL = ACL
				module_assignments_opt = merged_assignments
	else:
	# Randomly merge module pairs
		for trial in range(n_trials):
			merged_assignments = module_assignments.copy()
			for t in range(n_itr):
				modules = list(set(merged_assignments))
				if len(modules) == 1:
					break
				else:
					ids = random.sample(modules, 2)
					merged_assignments = [ids[0] if v_module == ids[1] else v_module for v_module in merged_assignments]
					ACL = codelength.AverageCodeLength(merged_assignments, trajectories=trajectories, scheme=scheme, init_module=init_module, init_node=init_node)
					if ACL < MCL:
						MCL = ACL
						module_assignments_opt = merged_assignments
	
	# Rename module labels to a set of labels from zero
	module_labels = list(set(module_assignments_opt))
	module_assignments_opt_ = [module_labels.index(k) for k in module_assignments_opt]
	return MCL, module_assignments_opt_


def trajectories_to_edgelist(trajectories):
	edgelist = []
	for trajectory in trajectories:
		for i in range(len(trajectory)-1):
			edgelist.append([trajectory[i],trajectory[i+1]])
	return edgelist


def Infomap_rawdir(edgelist, vertices):
	im = Infomap("--two-level -f rawdir", num_trials=100)
	if vertices is not None:
		im.add_nodes(vertices)
	for edge in edgelist:
		im.add_link(edge[0], edge[1])
	im.run()
	
	d = {}
	for node in im.tree:
		if node.is_leaf:
			d[node.node_id] = node.module_id-1
	module_assignments_ = dict(sorted(d.items()))
	module_assignments = list(module_assignments_.values())
	return module_assignments


def Infomap_st(trajectories, vertices=None, scheme="Huffman", init_module=True, init_node=True, deterministic=False, n_trials=10, n_itr=1000):
	# Initial partition by Infomap
	edgelist = trajectories_to_edgelist(trajectories)
	module_assignments = Infomap_rawdir(edgelist, vertices)

	# Correction by the single-trajectory map equation
	MCL_stMapEqn, module_assignments_stMapEqn = MCL_ST, module_assignments = merge_modules(trajectories, module_assignments, \
		scheme=scheme, init_module=init_module, init_node=init_node, \
		deterministic=deterministic, n_trials=n_trials, n_itr=n_itr)

	return MCL_stMapEqn, module_assignments_stMapEqn
