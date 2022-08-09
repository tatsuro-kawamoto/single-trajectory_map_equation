import numpy as np
########################
# The following code is copied from 
# https://www.geeksforgeeks.org/shannon-fano-algorithm-for-data-compression/
########################

# Python3 program for Shannon Fano Algorithm
 
# declare structure node
# codelimit = 10000
class node:
	def __init__(self) -> None:
		# for storing symbol
		self.sym=''
		# for storing probability or frequency
		self.pro=0.0
		self.arr=[0]*codelimit
		self.top=0
 
# function to find shannon code
def shannon(l, h, p):
	pack1 = 0; pack2 = 0; diff1 = 0; diff2 = 0
	if ((l + 1) == h or l == h or l > h) :
		if (l == h or l > h):
			return
		p[h].top+=1
		p[h].arr[(p[h].top)] = 0
		p[l].top+=1
		p[l].arr[(p[l].top)] = 1
		 
		return
	 
	else :
		for i in range(l,h):
			pack1 = pack1 + p[i].pro
		pack2 = pack2 + p[h].pro
		diff1 = pack1 - pack2
		if (diff1 < 0):
			diff1 = diff1 * -1
		j = 2
		while (j != h - l + 1) :
			k = h - j
			pack1 = pack2 = 0
			for i in range(l, k+1):
				pack1 = pack1 + p[i].pro
			for i in range(h,k,-1):
				pack2 = pack2 + p[i].pro
			diff2 = pack1 - pack2
			if (diff2 < 0):
				diff2 = diff2 * -1
			if (diff2 >= diff1):
				break
			diff1 = diff2
			j+=1
		 
		k+=1
		for i in range(l,k+1):
			p[i].top+=1
			p[i].arr[(p[i].top)] = 1
			 
		for i in range(k + 1,h+1):
			p[i].top+=1
			p[i].arr[(p[i].top)] = 0
			 
 
		# Invoke shannon function
		shannon(l, k, p)
		shannon(k + 1, h, p)
	 
 
 
# Function to sort the symbols
# based on their probability or frequency
def sortByProbability(n, p):
	temp=node()
	for j in range(1,n) :
		for i in range(n - 1) :
			if ((p[i].pro) > (p[i + 1].pro)) :
				temp.pro = p[i].pro
				temp.sym = p[i].sym
 
				p[i].pro = p[i + 1].pro
				p[i].sym = p[i + 1].sym
 
				p[i + 1].pro = temp.pro
				p[i + 1].sym = temp.sym
			 
		 
	 
 
 
# function to display shannon codes
def display(n, p):
	print("\n\n\n\tSymbol\tProbability\tCode",end='')
	for i in range(n - 1,-1,-1):
		print("\n\t", p[i].sym, "\t\t", p[i].pro,"\t",end='')
		for j in range(p[i].top+1):
			print(p[i].arr[j],end='')


def ShannonFano_code(symb2freq):
	global codelimit
	# Input number of symbols
	n = len(symb2freq)
	codelimit = n

	p=[node() for _ in range(codelimit)]

	i=0
	# Input symbols
	for i in range(n):
		# Insert the symbol to node
		p[i].sym += list(symb2freq.keys())[i]
	 

	# Input probability of symbols
	freqs = list(symb2freq.values())
	if np.sum(freqs) == 0:
		codewords = [[p[i].sym, '']]
	else:
		x = freqs/np.sum(freqs)
		for i in range(n):
			# Insert the value to node
			p[i].pro = x[i]

		# Sorting the symbols based on
		# their probability or frequency
		sortByProbability(n, p)

		for i in range(n):
			p[i].top = -1

		# Find the shannon code
		shannon(0, n - 1, p)
	 
		# Display the codes
		# display(n, p)
		codewords = []
		for i in range(n):
			code = ''.join([str(p[i].arr[j]) for j in range(p[i].top+1)])
			codewords.append([p[i].sym, code])
	
	return codewords

def encode_ShannonFano(symb2freq, lmbda):
    code = ShannonFano_code(symb2freq)
    return [[c[0], lmbda*len(c[1])] for c in code]

