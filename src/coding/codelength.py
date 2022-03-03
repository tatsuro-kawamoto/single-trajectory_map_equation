import numpy as np
from src.coding.huffman import encode_Huffman
from src.coding.shannonfano import encode_ShannonFano
"""
module_assignments: N-dim array indicating the module assignment of each node (N = Num. of nodes)
trajectories: Array of arrays indicating the set of trajectories (visited nodes)
"""

def incidence_to_trajectories(B):
    trajectories = []
    for i in range(B.shape[0]):
        if sum(B[i,:]) > 0:
            traj = np.where(B[i,:]==1)
            trajectories.append(traj[0])
    return trajectories

def encoding(codebook, module_assignments, trajectories, init_module):
    # Code of the trajectories
    codes = []
    for trajectory in trajectories:
        code = ''
        prev_module = -1
        for v in trajectory:
            module = module_assignments[v]
            if module != prev_module:
                if init_module == True:
                    code += codebook['enter_module_'+str(module)] # Count module assignment even if it is the starting point of the trajectory

                if prev_module != -1:
                    code += codebook['exit_module_'+str(prev_module)]
                    if init_module == False:
                        code += codebook['enter_module_'+str(module)] # Count module assignment only when it is NOT the starting point of the trajectory
                prev_module = module
            code += codebook['visit_node_'+str(v)]
        codes.append(code)
    
    return codes

def visiting_frequencies(module_assignments, trajectories, init_module, init_node):
    N = len(module_assignments)
    module_list = list(set(module_assignments))
    n_modules = len(module_list)
    freq_lv1 = np.zeros((n_modules, 2), dtype=int) # entering_freq., exiting freq.
    freq_nodes = np.zeros(N, dtype=int) # node visiting_freq.

    # Walker's visiting frequencies
    for trajectory in trajectories:
        prev_module = -1
        init = True
        for v in trajectory:
            if init == False or init_node == True:
                freq_nodes[int(v)] += 1
            init = False
            module = module_assignments[v]
            if prev_module == -1:
                prev_module = module
                if init_module == True:
                    # Turn the following line on to include the initial module:
                    freq_lv1[module_list.index(module),0] += 1

            elif module != prev_module:
                freq_lv1[module_list.index(prev_module),1] += 1
                freq_lv1[module_list.index(module),0] += 1
                prev_module = module

    return freq_nodes, freq_lv1

def generate_codebooks(module_assignments, trajectories, scheme='Huffman', init_module=True, init_node=True):
    """
    module_assignments: N-dim array indicating the module assignment of each node (N = Num. of nodes)
    trajectories: Array of arrays indicating the set of trajectories (visited nodes)
    """
    freq_nodes, freq_lv1 = visiting_frequencies(module_assignments, trajectories, init_module, init_node)

    # Create a codebook
    codebook_arr = []
    module_list = list(set(module_assignments))
    for module in module_list:
        nodes = np.where(np.array(module_assignments) == module)[0]
        freq_lv0_dict = {'visit_node_'+str(i):freq_nodes[i] for i in nodes}
        if freq_lv1[module_list.index(module),1] > 0:
            freq_lv0_dict['exit_module_'+str(module)] = freq_lv1[module_list.index(module),1]
        if scheme == 'Shannon-Fano':
            codewords = encode_ShannonFano(freq_lv0_dict)
        else: # scheme == 'Huffman'
            codewords = encode_Huffman(freq_lv0_dict)
        codebook_arr.extend(codewords)

    freq_lv1_dict = {'enter_module_'+str(module): freq_lv1[module_list.index(module),0] for module in module_list if freq_lv1[module_list.index(module),0]>0}
    if len(freq_lv1_dict)>0:
        if scheme == 'Shannon-Fano':
            codewords = encode_ShannonFano(freq_lv1_dict)
        else: # scheme == 'Huffman'
            codewords = encode_Huffman(freq_lv1_dict)
        codebook_arr.extend(codewords)

    codebooks = {code[0]: code[1] for code in codebook_arr}
    return codebooks

def Shannon_Limit(module_assignments, trajectories, init_module, init_node):
    def entropy(d):
        freqs = list(d.values())
        if sum(freqs) > 0:
            probs = freqs/sum(freqs)
            return sum([-p*np.log2(p) for p in probs if p>0])
        else:
            return 0
    
    total_trajectory_length = sum([len(trajectory) for trajectory in trajectories])
    freq_nodes, freq_lv1 = visiting_frequencies(module_assignments, trajectories, init_module, init_node)
    module_list = list(set(module_assignments))
    ShannonLimit = 0
    for module in module_list:
        nodes = np.where(np.array(module_assignments) == module)[0]
        freq_lv0_dict = {'visit_node_'+str(i):freq_nodes[i] for i in nodes}
        if freq_lv1[module_list.index(module),1] > 0:
            freq_lv0_dict['exit_module_'+str(module)] = freq_lv1[module_list.index(module),1]
        ShannonLimit += entropy(freq_lv0_dict)*sum(list(freq_lv0_dict.values()))/total_trajectory_length

    freq_lv1_dict = {'enter_module_'+str(module): freq_lv1[module_list.index(module),0] for module in module_list if freq_lv1[module_list.index(module),0]>0}
    if len(freq_lv1_dict)>0:
        ShannonLimit += entropy(freq_lv1_dict)*sum(list(freq_lv1_dict.values()))/total_trajectory_length
    return ShannonLimit


# All in one --------------------------
def AverageCodeLength(module_assignments, B=None, trajectories=None, scheme='Huffman', init_module=True, init_node=True):
    if trajectories is not None:
        M = len(trajectories) #len([v for trajectory in trajectories for v in trajectory])
        trajectory_lengths = [len(trajectory) for trajectory in trajectories]
    elif B is not None:
        M = np.sum(B)
        trajectory_lengths = np.sum(B, axis=1)
        trajectories = incidence_to_trajectories(B)
    else:
        print("Error: Either trajectories (list of lists) or B (binary incidence matrix) must be provided as an input.", file=sys.stderr)
        sys.exit()

    if scheme == 'lower_bound':
        average_code_length = Shannon_Limit(module_assignments, trajectories, init_module, init_node)
    else:
        codebook = generate_codebooks(module_assignments, trajectories, scheme, init_module, init_node)
        codes = encoding(codebook, module_assignments, trajectories, init_module)
        average_code_length = sum([len(codes[k])/trajectory_lengths[k] for k in range(len(codes)) if trajectory_lengths[k]>0 ])/M # Exclude length=0 case, but keep M the same.
    return average_code_length


