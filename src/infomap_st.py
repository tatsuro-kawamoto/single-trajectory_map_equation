import numpy as np
from infomap import Infomap
import src.coding.codelength as clen

def Infomap_rawdir(e_trajectories):
	im = Infomap("--two-level -f rawdir", num_trials=100)
	for edge in e_trajectories:
		im.add_link(edge[0], edge[1])
	im.run()

	d = {}
	for node in im.tree:
		if node.is_leaf:
			d[node.node_id] = node.module_id-1
	membership_ = dict(sorted(d.items()))
	membership = list(membership_.values())
	return membership


class Infomap_st:
	"""
	Each trajectory must have length more than 0 (i.e., a walker must move at least one step).
	"""
	def __init__(self, trajectories, membership=None, coding="lower_bound", init_module=True, init_node=True, min_size=3, lmbda=1):
		self.coding = coding
		self.init_module = init_module
		self.init_node = init_node
		self.min_size = min_size
		self.lmbda = lmbda

		self.e_trajectories = self.trajectories_in_edgelist_format(trajectories) # Break down each trajectory into a set of edges with a starting-point flag
		self.num_trajectories = len(trajectories)
		self.trajectory_lengths = [len(trajectory) for trajectory in trajectories]
		
		# Variables for the optimization ========
		# Initial partition
		self.membership = Infomap_rawdir(self.e_trajectories) if membership is None else self.rename_membership(membership)

		self.module_list = list(set(self.membership)) # This is remain fixed (empty modules will appear because of merging. We assume that there is no empty modules at the beggining). 
		self.freq_init, self.module_adjacency = clen.module_adjacency_matrix(self.e_trajectories, self.membership)
		self.freq_nodes = clen.node_visiting_frequencies(self.e_trajectories, self.init_node, N=len(self.membership)) # This is remain fixed.
		self.freq_lv1 = clen.module_visiting_frequencies(self.e_trajectories, self.membership, self.module_list, self.init_module)

		self.codelength, self.codebooks, self.codelength_dict = self.AverageCodeLength()

	def rename_membership(self, membership):
		# Rename module labels to a set of labels from zero
		module_labels = list(set(membership))
		return [module_labels.index(k) for k in membership]

	def trajectories_in_edgelist_format(self, trajectories):
		e_trajectories = []
		for trajectory in trajectories:
			start = True
			for i in range(len(trajectory)-1):
				e_trajectories.append([trajectory[i],trajectory[i+1],start])
				start = False
		return e_trajectories

	def AverageCodeLength(self, ):
		if self.coding == 'lower_bound':
			codebooks = None
			codelength_dict = None
			average_code_length = clen.Shannon_Limit(self.membership, self.freq_nodes, self.freq_lv1, self.module_list, self.trajectory_lengths, self.lmbda)
		else:
			codebooks = clen.generate_codebooks(self.e_trajectories, self.membership, self.freq_nodes, self.module_list, self.coding, self.init_module, self.init_node, self.lmbda)
			codelength_dict = clen.encoding(codebooks, self.e_trajectories, self.trajectory_lengths, self.membership, self.module_list, self.init_module, self.init_node)
			average_code_length = sum(codelength_dict.values())/self.num_trajectories
		return average_code_length, codebooks, codelength_dict

	def evaluate_module_merge_lower_bound(self, average_code_length, freq_lv1, freq_init, module_adjacency, module_merged, module_erased):
		"""
		`freq_lv1` is not used in this function
		"""
		freq_lv1_tmp, module_adjacency_tmp = clen.update_module_visiting_frequencies(freq_lv1, module_adjacency, module_merged, module_erased)
		average_code_length_tmp, freq_init_tmp = clen.update_Shannon_Limit(average_code_length, freq_init, module_adjacency, module_adjacency_tmp, module_merged, module_erased, self.trajectory_lengths, self.lmbda)

		return average_code_length_tmp, freq_init_tmp, module_adjacency_tmp

	def evaluate_module_merge(self, membership, freq_lv1, module_adjacency, codebooks, codelength_dict, module_merged, module_erased):
		freq_lv1_tmp, module_adjacency_tmp = clen.update_module_visiting_frequencies(freq_lv1, module_adjacency, module_merged, module_erased)

		codebooks_tmp = clen.update_codebooks(codebooks, self.freq_nodes, freq_lv1_tmp, membership, self.coding, self.lmbda, module_merged, module_erased)
		codelength_dict_tmp = clen.update_encoding(codelength_dict, module_merged, module_erased, codebooks_tmp, self.e_trajectories, self.trajectory_lengths, membership, self.module_list, self.init_module, self.init_node)
		average_code_length_tmp = sum(codelength_dict_tmp.values())/self.num_trajectories

		return average_code_length_tmp, freq_lv1_tmp, module_adjacency_tmp, codebooks_tmp, codelength_dict_tmp


	def greedy_sequential(self, ):
		def module_sizes_init(membership):
			module_sizes = np.zeros(len(set(membership)))
			for m in membership:
				module_sizes[m] += 1
			return module_sizes

		def smallest_module_size(module_sizes):
			valid_idx = np.where(module_sizes > 0)[0]
			return min(module_sizes[valid_idx])

		def extract_smallest_module(arr):
			valid_idx = np.where(arr > 0)[0]
			return valid_idx[arr[valid_idx].argmin()]

		def extract_tightly_connected_module(data, module_erased):
			m = np.zeros(data.shape[0], dtype=bool)
			m[module_erased] = True
			argmax_row = np.argmax(np.ma.array(data[module_erased,:], mask=m))
			argmax_column = np.argmax(np.ma.array(data[:,module_erased], mask=m))
			return argmax_column if data[module_erased,argmax_column] > data[argmax_row,module_erased] else argmax_row

		# Initialize
		membership = self.membership
		freq_lv1 = self.freq_lv1
		freq_init = self.freq_init
		module_adjacency = self.module_adjacency
		codebooks = self.codebooks
		codelength_dict = self.codelength_dict
		average_code_length = self.codelength
		nonempty_modules = list(range(freq_lv1.shape[0]))
		module_sizes = module_sizes_init(membership)
		module_size_min = min(module_sizes)

		membership_best = membership
		min_code_length = average_code_length
		while True:
			if len(nonempty_modules) == 1:
				break
			module_erased = extract_smallest_module(arr=module_sizes)
			module_merged = extract_tightly_connected_module(data=module_adjacency, module_erased=module_erased)

			if self.coding == 'lower_bound':
				average_code_length, freq_init, module_adjacency = self.evaluate_module_merge_lower_bound(average_code_length, freq_lv1, freq_init, module_adjacency, module_merged, module_erased)
			else:
				average_code_length, freq_lv1, module_adjacency, codebooks, codelength_dict = self.evaluate_module_merge(membership, freq_lv1, module_adjacency, codebooks, codelength_dict, module_merged, module_erased)

			membership = [module_merged if v_module == module_erased else v_module for v_module in membership]
			nonempty_modules.remove(module_erased)
			module_sizes[module_merged] += module_sizes[module_erased]
			module_sizes[module_erased] = 0

			if average_code_length < min_code_length:
				min_code_length = average_code_length
				membership_best = membership
				module_size_min = smallest_module_size(module_sizes)

		return min_code_length, membership_best, module_size_min



	def optimize(self, ):
		# Sequentially merge module pairs
		min_code_length, membership_opt, module_size_min = self.greedy_sequential()
		self.codelength = min_code_length
		self.membership = self.rename_membership(membership_opt)
		if module_size_min < self.min_size:
			print("Warning: Some modules are too small (the smallest module size = "+str(int(module_size_min))+")")

		return self.membership


