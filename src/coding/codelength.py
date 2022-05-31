import numpy as np
from src.coding.huffman import encode_Huffman
from src.coding.shannonfano import encode_ShannonFano

# Uppdate coding ========================================================
def update_module_visiting_frequencies(freq_lv1,module_adjacency,module_merged,module_erased):
	"""
	This function updates `freq_lv1`, `freq_init`, and `module_adjacency`.
	"""
	# Update freq_lv1
	freq_lv1_ = freq_lv1.copy()
	freq_lv1_[module_merged,:] += freq_lv1_[module_erased,:]
	freq_lv1_[module_erased,:] = np.zeros(2)

	freq_lv1_[module_merged,0] -= module_adjacency[module_erased][module_merged] + module_adjacency[module_merged][module_erased] # overcounting of enter frequencies
	freq_lv1_[module_merged,1] -= module_adjacency[module_erased][module_merged] + module_adjacency[module_merged][module_erased] # overcounting of exit frequencies

	# Update module_adjacency_matrix
	module_adjacency_ = module_adjacency.copy()
	module_adjacency_[module_merged,:] += module_adjacency_[module_erased,:]
	module_adjacency_[module_erased,:] = np.zeros(module_adjacency_.shape[0])
	module_adjacency_[:,module_merged] += module_adjacency_[:,module_erased]
	module_adjacency_[:,module_erased] = np.zeros(module_adjacency_.shape[0])

	return freq_lv1_, module_adjacency_

def update_codebooks(codebooks, freq_nodes, freq_lv1, membership, coding, lmbda, module_merged, module_erased):
	"""
	This function updates `codebooks`.
	- The input `codebooks` and `membership` are the variables before updated.
	- `freq_lv1` is assumed to be updated already.
	- `freq_nodes` is a variable that is not updated during the optimization.
	"""
	codebooks_ = codebooks.copy()

	# Overwrite an intra-module codebook
	nodes_in_module = list(np.where((np.array(membership) == module_merged)|(np.array(membership) == module_erased))[0])
	freq_lv0_dict = {'visit_node_'+str(i):freq_nodes[i] for i in nodes_in_module}
	if freq_lv1[module_merged,1] > 0:
		freq_lv0_dict['exit_module_'+str(module_merged)] = freq_lv1[module_merged,1]
		
	if coding == 'Shannon-Fano':
		codewords = encode_ShannonFano(freq_lv0_dict,lmbda=1) # lmbda=1 for the intra-module codebook
	else: # coding == 'Huffman'
		codewords = encode_Huffman(freq_lv0_dict,lmbda=1) # lmbda=1 for the intra-module codebook
		
	for codeword in codewords:
		codebooks_[codeword[0]] = codeword[1]

	# Overwrite the inter-module codebook (updated entirely, reflectiing the updated `freq_lv1`)
	freq_lv1_dict = {'enter_module_'+str(k): freq_lv1[k,0] for k in range(freq_lv1.shape[0]) if freq_lv1[k,0]>0}
	if coding == 'Shannon-Fano':
		codewords = encode_ShannonFano(freq_lv1_dict,lmbda)
	else: # coding == 'Huffman'
		codewords = encode_Huffman(freq_lv1_dict,lmbda)
		
	for codeword in codewords:
		codebooks_[codeword[0]] = codeword[1]
		
	# Eliminate codewords (code lengths) for the module_erased (although this is not mandatory)
	codebooks_.pop('enter_module_'+str(module_erased), None)
	codebooks_.pop('exit_module_'+str(module_erased), None)

	return codebooks_

def update_encoding(codelength_dict, module_merged, module_erased, codebooks, e_trajectories, trajectory_lengths, membership, module_list, init_module, init_node):
	"""
	This function updates `codelength_dict`.
	- The input `codelength_dict` and `membership` are the variables before updated.
	- `codebooks` is assumed to be updated already.
	"""
	codelength_dict_ = codelength_dict.copy()
	codelength_dict_[module_merged] = 0
	codelength_dict_[module_erased] = 0
	codelength_dict_['inter_module'] = 0

	traj = -1
	codelength = 0
	for edge in e_trajectories:
		# Starting point of a trajectory
		traj += 1 if edge[2] == True else 0
		module_0 = module_merged if membership[edge[0]] == module_erased else membership[edge[0]]
		module_1 = module_merged if membership[edge[1]] == module_erased else membership[edge[1]]
		# module_0, module_1 = module_indices(edge[0], edge[1])

		if module_0 == module_merged or module_1 == module_merged:
			# When module_0 and/or module_1 belong to module_merged and/or module_erased, update the code lengths entirely.
			# Update the encoding for the intra-module codebook only when [target module] == module_merged (because codelength_dict remains the same otherwise).
			if edge[2] == True:
				codelength_dict_['inter_module'] += codebooks['enter_module_'+str(module_0)]/trajectory_lengths[traj] if init_module == True else 0 # module entering
				if module_0 == module_merged:
					codelength_dict_[module_0] += codebooks['visit_node_'+str(edge[0])]/trajectory_lengths[traj] if init_node == True else 0 # node visiting (source node)

			# Transition to anoter module (after modules are merged)
			if module_0 != module_1:
				if module_0 == module_merged:
					codelength_dict_[module_0] += codebooks['exit_module_'+str(module_0)]/trajectory_lengths[traj] # module exiting
				codelength_dict_['inter_module'] += codebooks['enter_module_'+str(module_1)]/trajectory_lengths[traj] # module entering

			# node visit (target node)
			if module_1 == module_merged:
				codelength_dict_[module_1] += codebooks['visit_node_'+str(edge[1])]/trajectory_lengths[traj]
		else:
			# When neither module_0 nor module_1 belongs to module_merged, update codelength_dict_['inter_module'] only
			if edge[2] == True:
				codelength_dict_['inter_module'] += codebooks['enter_module_'+str(module_0)]/trajectory_lengths[traj] if init_module == True else 0 # module entering

			# Transition to anoter module
			if module_0 != module_1:
				codelength_dict_['inter_module'] += codebooks['enter_module_'+str(module_1)]/trajectory_lengths[traj] # module entering
	
	return codelength_dict_

def update_Shannon_Limit(average_code_length, freq_init, module_adjacency, module_adjacency_tmp, module_merged, module_erased, trajectory_lengths, lmbda):
	"""
	This function updates `average_code_length`.
	- The input `average_code_length` and `freq_init` are the variable before updated.
	- `module_adjacency_tmp` is assumed to be updated already.
	"""
	def qlogq(q):
		return -q*np.log2(q) if q>0 else 0
	
	total_trajectory_length = sum(trajectory_lengths)
	# Update freq_init
	freq_init_tmp = freq_init.copy()
	freq_init_tmp[module_merged] += freq_init_tmp[module_erased]
	freq_init_tmp[module_erased] = 0

	# Module entering probabilities
	q_module_enter = [(freq_init[k] + sum(module_adjacency[k,:]) - module_adjacency[k,k])/total_trajectory_length for k in range(len(freq_init))]
	q_module_enter_tmp = [(freq_init_tmp[k] + sum(module_adjacency_tmp[k,:]) - module_adjacency_tmp[k,k])/total_trajectory_length for k in range(len(freq_init))]

	# Module exiting probabilities
	q_module_exit = [(sum(module_adjacency[:,k]) - module_adjacency[k,k])/total_trajectory_length for k in range(len(freq_init))]
	q_module_exit_tmp = [(sum(module_adjacency_tmp[:,k]) - module_adjacency_tmp[k,k])/total_trajectory_length for k in range(len(freq_init))]

	# Node visiting probability for each module. The diagonal element in module_adjacency is doubly counted because it represents the visit of two nodes within a module. 
	q_node_per_module = [(sum(module_adjacency[k,:]) + sum(module_adjacency[:,k]))/total_trajectory_length for k in range(len(freq_init))]
	q_node_per_module_tmp = [(sum(module_adjacency_tmp[k,:]) + sum(module_adjacency_tmp[:,k]))/total_trajectory_length for k in range(len(freq_init))]

	average_code_length += lmbda*(qlogq(q_module_enter_tmp[module_merged]) - qlogq(q_module_enter[module_merged]) - qlogq(q_module_enter[module_erased]))
	average_code_length -= lmbda*(qlogq(sum(q_module_enter_tmp)) - qlogq(sum(q_module_enter)))
	average_code_length += qlogq(q_module_exit_tmp[module_merged]) - qlogq(q_module_exit[module_merged]) - qlogq(q_module_exit[module_erased])
	average_code_length -= qlogq(q_module_exit_tmp[module_merged]+q_node_per_module_tmp[module_merged]) - qlogq(q_module_exit[module_merged]+q_node_per_module[module_merged]) - qlogq(q_module_exit[module_erased]+q_node_per_module[module_erased])

	return average_code_length, freq_init_tmp



# Coding of the whole system =====================================
def encoding(codebooks, e_trajectories, trajectory_lengths, membership, module_list, init_module, init_node):
	# Code length of the trajectories
	codelength_dict = {k:0 for k in module_list}
	codelength_dict['inter_module'] = 0

	traj = -1
	codelength = 0
	for edge in e_trajectories:
		module_0 = membership[edge[0]]
		module_1 = membership[edge[1]]

		# Starting point of a trajectory
		if edge[2] == True:
			traj += 1
			codelength_dict[module_0] += codebooks['visit_node_'+str(edge[0])]/trajectory_lengths[traj] if init_node == True else 0 # node visiting (source node)
			codelength_dict['inter_module'] += codebooks['enter_module_'+str(module_0)]/trajectory_lengths[traj] if init_module == True else 0 # module entering

		# Transition to anoter module
		if module_0 != module_1:
			codelength_dict[module_0] += codebooks['exit_module_'+str(module_0)]/trajectory_lengths[traj] # module exiting
			codelength_dict['inter_module'] += codebooks['enter_module_'+str(module_1)]/trajectory_lengths[traj] # module entering

		# node visit (target node)
		codelength_dict[module_1] += codebooks['visit_node_'+str(edge[1])]/trajectory_lengths[traj]
	
	return codelength_dict

def module_visiting_frequencies(e_trajectories, membership, module_list, init_module):
	n_modules = len(module_list)
	freq_lv1 = np.zeros((n_modules, 2), dtype=int) # entering_freq., exiting freq.

	# Walker's visiting frequencies
	for edge in e_trajectories:
		module_0 = membership[edge[0]]
		module_1 = membership[edge[1]]
		if edge[2] == True and init_module == True:
			# Turn the following line on to include the initial module:
			freq_lv1[module_list.index(module_0),0] += 1

		if module_0 != module_1:
			freq_lv1[module_list.index(module_0),1] += 1
			freq_lv1[module_list.index(module_1),0] += 1

	return freq_lv1

def node_visiting_frequencies(e_trajectories, init_node, N):
	freq_nodes = np.zeros(N, dtype=int) # node visiting_freq.

	# Walker's visiting frequencies
	for edge in e_trajectories:
		if edge[2] == True and init_node == True:
			freq_nodes[int(edge[0])] += 1
		freq_nodes[int(edge[1])] += 1

	return freq_nodes

def generate_codebooks(e_trajectories, membership, freq_nodes, module_list, coding='Huffman', init_module=True, init_node=True, lmbda=1):
	freq_lv1 = module_visiting_frequencies(e_trajectories, membership, module_list, init_module)

	# Create a codebook
	codebook_arr = []
	for module in module_list:
		nodes = np.where(np.array(membership) == module)[0]
		freq_lv0_dict = {'visit_node_'+str(i):freq_nodes[i] for i in nodes}
		if freq_lv1[module_list.index(module),1] > 0:
			freq_lv0_dict['exit_module_'+str(module)] = freq_lv1[module_list.index(module),1]
		if coding == 'Shannon-Fano':
			codewords = encode_ShannonFano(freq_lv0_dict,lmbda=1) # lmbda=1 for the intra-module codebook
		else: # coding == 'Huffman'
			codewords = encode_Huffman(freq_lv0_dict,lmbda=1) # lmbda=1 for the intra-module codebook
		codebook_arr.extend(codewords)

	freq_lv1_dict = {'enter_module_'+str(module): freq_lv1[module_list.index(module),0] for module in module_list if freq_lv1[module_list.index(module),0]>0}
	if len(freq_lv1_dict)>0:
		if coding == 'Shannon-Fano':
			codewords = encode_ShannonFano(freq_lv1_dict,lmbda)
		else: # coding == 'Huffman'
			codewords = encode_Huffman(freq_lv1_dict,lmbda)
		codebook_arr.extend(codewords)

	codebooks = {code[0]: code[1] for code in codebook_arr} # code length of each code (not the code itself)
	return codebooks

def Shannon_Limit(membership, freq_nodes, freq_lv1, module_list, trajectory_lengths, lmbda):
	def entropy(freqs):
		if sum(freqs) > 0:
			probs = freqs/sum(freqs)
			return sum([-p*np.log2(p) for p in probs if p>0])
		else:
			return 0
	
	total_trajectory_length = sum(trajectory_lengths)
	ShannonLimit = 0

	freq_lv0_list = [ [] for _ in module_list ] # intra-module codebook
	for i in range(len(membership)):
		freq_lv0_list[membership[i]].append(freq_nodes[i]) # node visit
	for module in module_list:
		freq_lv0_list[module].append(freq_lv1[module,1]) # module exit
		ShannonLimit += entropy(freq_lv0_list[module])*sum(freq_lv0_list[module])/total_trajectory_length

	freq_lv1_enter = [freq_lv1[module,0] for module in module_list if freq_lv1[module,0]>0]
	if len(freq_lv1_enter)>0:
		ShannonLimit += lmbda*entropy(freq_lv1_enter)*sum(freq_lv1_enter)/total_trajectory_length
	return ShannonLimit

def module_adjacency_matrix(e_trajectories, membership):
	n_modules = len(set(membership))
	freq_init = np.zeros(n_modules)
	module_adjacency = np.zeros((n_modules,n_modules))
	for edge in e_trajectories:
		module_adjacency[membership[edge[1]], membership[edge[0]]] += 1
		if edge[2] == True:
			freq_init[membership[edge[0]]] += 1

	return freq_init, module_adjacency

